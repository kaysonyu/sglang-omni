# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for MOSS-TTS Local."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    _STREAMING_ROPE_CACHE_DURATION_SECONDS,
)
from sglang_omni.models.moss_tts.request_builders import (
    _DATA_URI_RE,
    MOSS_TTS_DEFAULT_MAX_NEW_TOKENS,
    build_row_cache_key_ids,
    derive_moss_tts_sampling_seed,
    new_moss_tts_sampling_seed,
    normalize_moss_tts_inputs,
    reference_for_processor,
    resolve_moss_reference,
    resolve_optional_text,
    resolve_token_count,
    validate_moss_tts_generation_kwargs,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    MossTTSLocalPrompt,
    MossTTSLocalState,
)
from sglang_omni.models.moss_tts_local.rollout_trace import (
    build_moss_tts_local_rollout_trace,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.scheduling.types import ARRequestData

_MOSS_TTS_LOCAL_PREPARED_MARKER = "_moss_tts_local_prepared_request"
_MOSS_TTS_LOCAL_AUDIO_FRAME_RATE = 12.5
# note (Zhang Yiyang): Each Local AR step emits at most one audio frame;
# derive the token limit once from the fixed RoPE duration budget.
_MOSS_TTS_LOCAL_MAX_STREAMING_TOKENS = int(
    _STREAMING_ROPE_CACHE_DURATION_SECONDS * _MOSS_TTS_LOCAL_AUDIO_FRAME_RATE
)


@dataclass
class MossTTSLocalSGLangRequestData(ARRequestData):
    """Scheduler-owned request state for MOSS-TTS Local."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    # note (Yue Yin): launch-side seeded-sampling step counter (async decode); advances
    # at launch while generation_steps moves at resolve, floored so the sync path is unchanged.
    sampling_steps: int | None = None
    suppress_tokens: list[int] | None = None
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    state: MossTTSLocalState = field(default_factory=MossTTSLocalState)
    model_config: Any = None
    prompt_rows: torch.Tensor | None = None
    output_rows: list[torch.Tensor] = field(default_factory=list)
    # note (Yue Yin): checkpoint generate() defaults — the continue/stop head samples at
    # temperature 1.0 while audio channels use the model-card values (1.7 / 0.8 / 25, no rep penalty).
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0
    seed: int | None = None
    sampling_seed: int = field(default_factory=new_moss_tts_sampling_seed)
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None
    stream_pending_rows: list[torch.Tensor] = field(default_factory=list)
    stream_first_batch_sent: bool = False
    return_omni_rollout: bool = False
    admission_weight_version: str | None = None
    output_decisions: list[torch.Tensor] = field(default_factory=list)
    output_decision_logprobs: list[torch.Tensor] = field(default_factory=list)
    output_code_logprobs: list[torch.Tensor] = field(default_factory=list)


@dataclass
class MossTTSLocalPreparedRequest:
    """Heavy preprocessing output consumed by the AR scheduler."""

    state: MossTTSLocalState
    input_ids_list: list[int]
    input_ids: torch.Tensor
    prompt_rows: torch.Tensor
    gen_kwargs: dict[str, Any]


@dataclass
class PreprocessingContext:
    processor: Any
    reference_encoder: Any = None


_QUEUE: PreparedRequestQueue[PreprocessingContext, MossTTSLocalPreparedRequest] = (
    PreparedRequestQueue()
)
MOSS_STREAM_TRANSPORT_BATCH_FRAMES = 5


def set_moss_tts_local_preprocessing_context(
    *, processor: Any, reference_encoder: Any = None
) -> None:
    _QUEUE.set_context(
        PreprocessingContext(processor=processor, reference_encoder=reference_encoder)
    )


def clear_moss_tts_local_preprocessing_context() -> None:
    _QUEUE.clear_context()


def cleanup_prepared_moss_tts_local_request(request_id: str) -> None:
    """Drop any prepared handoff for an aborted request (see MOSS Delay)."""
    _QUEUE.abort(str(request_id))


def pop_prepared_moss_tts_local_request(
    payload: StagePayload,
) -> MossTTSLocalPreparedRequest | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_MOSS_TTS_LOCAL_PREPARED_MARKER)
    if marker is None:
        return None
    prepared = _QUEUE.pop(str(marker))
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS Local preprocessing state is missing for prepared payload "
            f"{marker!r}; the AR scheduler must not rebuild it"
        )
    return prepared


def build_moss_tts_local_state(payload: StagePayload) -> MossTTSLocalState:
    inputs = payload.request.inputs or {}
    params = dict(payload.request.params or {})
    params.update((params.get("stage_params") or {}).get("tts_engine") or {})
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    else:
        tts_params = dict(tts_params)
    explicit = metadata.get(EXPLICIT_GENERATION_PARAMS_KEY)
    if explicit is not None:
        tts_params["explicit_generation_params"] = explicit

    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    language = resolve_optional_text(
        tts_params.get("language") or params.get("language")
    )
    if language is not None and language.casefold() == "auto":
        language = None
    instructions = resolve_optional_text(
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    text, token_count = resolve_token_count(text, params, tts_params)
    return MossTTSLocalState(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        instructions=instructions,
        token_count=token_count,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
        return_logprob=bool(params.get("return_logprob", False)),
        return_omni_rollout=bool(params.get("return_omni_rollout", False)),
    )


def build_generation_kwargs(
    params: dict[str, Any],
    *,
    tts_params: dict[str, Any],
) -> dict[str, Any]:
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()

    raw_max_new_tokens = params.get("max_new_tokens")
    if raw_max_new_tokens is None:
        max_new_tokens = MOSS_TTS_DEFAULT_MAX_NEW_TOKENS
    elif isinstance(raw_max_new_tokens, bool):
        raise ValueError(
            f"MOSS-TTS max_new_tokens must be an integer, got {raw_max_new_tokens!r}"
        )
    else:
        max_new_tokens = int(raw_max_new_tokens)

    if params.get("stream") and max_new_tokens > _MOSS_TTS_LOCAL_MAX_STREAMING_TOKENS:
        raise ValueError(
            "MOSS-TTS Local streaming max_new_tokens must be <= "
            f"{_MOSS_TTS_LOCAL_MAX_STREAMING_TOKENS} "
            f"({_STREAMING_ROPE_CACHE_DURATION_SECONDS / 60:g} minutes at "
            f"{_MOSS_TTS_LOCAL_AUDIO_FRAME_RATE:g} audio frames/s), "
            f"got {max_new_tokens}"
        )

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "text_temperature": 1.0,
        "audio_temperature": 1.7,
        "text_top_p": 1.0,
        "audio_top_p": 0.8,
        "text_top_k": 50,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
    }

    if "temperature" in explicit_fields and params.get("temperature") is not None:
        generation_kwargs["text_temperature"] = float(params["temperature"])
        generation_kwargs["audio_temperature"] = float(params["temperature"])
    if "top_p" in explicit_fields and params.get("top_p") is not None:
        generation_kwargs["text_top_p"] = float(params["top_p"])
        generation_kwargs["audio_top_p"] = float(params["top_p"])
    if "top_k" in explicit_fields and params.get("top_k") is not None:
        generation_kwargs["text_top_k"] = int(params["top_k"])
        generation_kwargs["audio_top_k"] = int(params["top_k"])
    if (
        "repetition_penalty" in explicit_fields
        and params.get("repetition_penalty") is not None
    ):
        generation_kwargs["audio_repetition_penalty"] = float(
            params["repetition_penalty"]
        )

    for source in (tts_params, params):
        for field_name in (
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
        ):
            if source.get(field_name) is not None:
                value = source[field_name]
                generation_kwargs[field_name] = (
                    int(value) if field_name.endswith("top_k") else float(value)
                )

    seed = tts_params.get("seed")
    if seed is None:
        seed = params.get("seed")
    if seed is not None:
        generation_kwargs["seed"] = seed

    validate_moss_tts_generation_kwargs(generation_kwargs)
    return generation_kwargs


def build_processor_message(
    processor: Any,
    state: MossTTSLocalState,
    reference_encoder: Any = None,
) -> dict[str, Any]:
    protocol = getattr(
        getattr(processor, "model_config", None), "prompt_protocol", None
    )
    is_v2 = protocol == "moss_tts_v2"
    if not is_v2 and state.script is not None:
        raise ValueError("MOSS-TTS v2 prompt fields require a moss_tts_v2 processor")
    reference = []
    reference_sources = (
        [item["uri"] for item in state.references]
        if state.script is not None
        else [state.ref_audio]
    )
    for ref_audio in reference_sources:
        if reference_encoder is not None and isinstance(ref_audio, str):
            if _DATA_URI_RE.match(ref_audio) is None:
                reference.append(reference_encoder.encode(ref_audio))
            else:
                reference.append(reference_encoder.encode_data_uri(ref_audio))
        else:
            reference.extend(reference_for_processor(processor, ref_audio) or [])
    if is_v2:
        return processor.build_user_message(
            script=state.script if state.script is not None else state.text,
            reference=reference or None,
            global_instruction=(
                state.global_instruction
                if state.script is not None
                else state.instructions
            ),
            tokens=state.token_count,
        )
    else:
        return processor.build_user_message(
            text=state.text,
            reference=reference or None,
            instruction=state.instructions,
            tokens=state.token_count,
            language=state.language,
        )


def prepare_moss_tts_local_request(
    payload: StagePayload,
    *,
    processor: Any,
    reference_encoder: Any = None,
) -> MossTTSLocalPreparedRequest:
    state = build_moss_tts_local_state(payload)
    inputs = payload.request.inputs
    input_references = (
        inputs.get("references") or [] if isinstance(inputs, dict) else []
    )
    if isinstance(inputs, dict) and (
        "script" in inputs
        or "global_instruction" in inputs
        or any(
            isinstance(reference, dict) and ("id" in reference or "uri" in reference)
            for reference in input_references
        )
    ):
        prompt = MossTTSLocalPrompt.model_validate(inputs)
        params = dict(payload.request.params or {})
        params.update((params.get("stage_params") or {}).get("tts_engine") or {})
        raw_tts_params = (payload.request.metadata or {}).get("tts_params")
        tts_params = raw_tts_params if isinstance(raw_tts_params, dict) else {}
        if any(
            key in source
            for source in (params, tts_params)
            for key in ("instructions", "instruct", "ref_audio", "ref_text")
        ):
            raise ValueError(
                "MOSS-TTS v2 inputs cannot mix legacy instruction or reference parameters"
            )
        state.script = (
            prompt.script
            if isinstance(prompt.script, str)
            else [segment.model_dump(exclude_none=True) for segment in prompt.script]
        )
        state.text = state.script if isinstance(state.script, str) else ""
        state.global_instruction = (
            json.dumps(
                prompt.global_instruction,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if isinstance(prompt.global_instruction, dict)
            else prompt.global_instruction
        )
        state.references = [reference.model_dump() for reference in prompt.references]
        if len({reference["id"] for reference in state.references}) != len(
            state.references
        ):
            raise ValueError("MOSS-TTS reference ids must be unique")
        state.language = prompt.language
        state.instructions = None
        state.ref_audio, state.ref_text = None, None
    message = build_processor_message(processor, state, reference_encoder)
    batch = processor([[message]], mode="generation")
    input_rows = batch["input_ids"]
    if input_rows.ndim != 3 or int(input_rows.shape[0]) != 1:
        raise ValueError(
            "MOSS-TTS Local processor must return input_ids with shape [1, T, C]"
        )
    prompt_rows = input_rows[0].detach().to(dtype=torch.long, device="cpu")
    input_ids_list = build_row_cache_key_ids(prompt_rows)
    return MossTTSLocalPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        prompt_rows=prompt_rows,
        gen_kwargs=state.generation_kwargs,
    )


def preprocess_moss_tts_local_payload(payload: StagePayload) -> StagePayload:
    """Run prompt/reference preprocessing outside the AR scheduler."""

    rid = str(payload.request_id)
    context = _QUEUE.begin(rid)
    if context is None:
        raise RuntimeError(
            "MOSS-TTS Local preprocessing context is not initialized; "
            "create_preprocessing_executor must register it before requests run"
        )

    try:
        prepared = prepare_moss_tts_local_request(
            payload,
            processor=context.processor,
            reference_encoder=context.reference_encoder,
        )
    except BaseException:
        _QUEUE.fail_inflight(rid)
        raise
    # note (Yue Yin): publish fails closed; when it drops the handoff (aborted
    # mid-flight or context reset) skip the marker, so the AR stage never pops a
    # marker whose prepared state no longer exists.
    published = _QUEUE.publish(rid, prepared)

    data = prepared.state.to_dict()
    if published:
        data[_MOSS_TTS_LOCAL_PREPARED_MARKER] = payload.request_id
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=data
    )


def build_moss_tts_local_stream_metadata(
    payload: StagePayload,
    *,
    n_vq: int,
) -> dict[str, Any] | None:
    """Stream contract attached to every forwarded row of a streaming request."""
    params = payload.request.params if isinstance(payload.request.params, dict) else {}
    if not params.get("stream"):
        return None
    metadata: dict[str, Any] = {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": int(n_vq),
    }
    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = params[
            INITIAL_CODEC_CHUNK_FRAMES_PARAM
        ]
    return metadata


def build_sglang_moss_tts_local_request(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSLocalSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_moss_tts_local_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS Local AR request builder requires a payload prepared by "
            "preprocess_moss_tts_local_payload"
        )

    cfg = model.config
    gen_kwargs = prepared.gen_kwargs
    admission_weight_version = None
    if getattr(model, "enable_rl", False) and not prepared.state.return_omni_rollout:
        raise ValueError("RL instances require return_omni_rollout=true")
    if prepared.state.return_omni_rollout:
        if not model.enable_rl:
            raise ValueError(
                "MOSS rollout requires starting the server with enable_rl=true"
            )
        if not prepared.state.return_logprob:
            raise ValueError("MOSS rollout requires return_logprob=true")
        for name, expected in {
            "text_top_p": 1.0,
            "audio_top_p": 1.0,
            "text_top_k": -1,
            "audio_top_k": -1,
            "audio_repetition_penalty": 1.0,
        }.items():
            if gen_kwargs[name] != expected:
                raise ValueError(f"MOSS rollout requires {name}={expected}")
        if any(
            gen_kwargs[name] <= 0 for name in ("text_temperature", "audio_temperature")
        ):
            raise ValueError("MOSS rollout requires positive text/audio temperatures")
        from sglang.srt.runtime_context import get_serving

        version = get_serving().weight_version
        if version is None:
            raise RuntimeError("MOSS rollout requires a weight version")
        admission_weight_version = str(version)
    max_new_tokens = int(
        gen_kwargs.get("max_new_tokens", MOSS_TTS_DEFAULT_MAX_NEW_TOKENS)
    )
    if prepared.state.return_omni_rollout and max_new_tokens <= 0:
        raise ValueError("MOSS rollout requires max_new_tokens > 0")
    audio_end = int(cfg.audio_end_token_id)
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[audio_end],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(cfg.vocab_size_list[0]))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={audio_end},
        vocab_size=int(cfg.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    data = MossTTSLocalSGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        return_omni_rollout=prepared.state.return_omni_rollout,
        admission_weight_version=admission_weight_version,
        model_config=cfg,
        prompt_rows=prepared.prompt_rows,
        text_temperature=float(gen_kwargs.get("text_temperature", 1.0)),
        text_top_p=float(gen_kwargs.get("text_top_p", 1.0)),
        text_top_k=int(gen_kwargs.get("text_top_k", 50)),
        audio_temperature=float(gen_kwargs.get("audio_temperature", 1.7)),
        audio_top_p=float(gen_kwargs.get("audio_top_p", 0.8)),
        audio_top_k=int(gen_kwargs.get("audio_top_k", 25)),
        audio_repetition_penalty=float(gen_kwargs.get("audio_repetition_penalty", 1.0)),
        seed=gen_kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(gen_kwargs["seed"])
            if gen_kwargs.get("seed") is not None
            else new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
        stream_metadata=build_moss_tts_local_stream_metadata(
            payload, n_vq=int(prepared.prompt_rows.shape[1]) - 1
        ),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_moss_tts_local_result(
    payload: StagePayload,
    data: MossTTSLocalSGLangRequestData,
) -> StagePayload:
    state = data.state
    if not data.output_rows and not data.return_omni_rollout:
        raise RuntimeError(
            "MOSS-TTS Local generated no audio frames. Please retry the request."
        )
    if data.output_rows:
        state.audio_codes = (
            torch.stack(data.output_rows)[:, 1:]
            .detach()
            .to(dtype=torch.long, device="cpu")
        )
    else:
        state.audio_codes = torch.empty(
            (0, int(data.model_config.n_vq)), dtype=torch.long
        )
    if data.return_omni_rollout:
        n_vq = int(data.model_config.n_vq)
        if data.admission_weight_version is None or data.weight_version is None:
            raise RuntimeError("MOSS rollout is missing its weight version")
        if data.admission_weight_version != str(data.weight_version):
            raise RuntimeError("MOSS rollout crossed a weight update")
        state.finish_reason = str(data.finish_reason)
        state.weight_version = str(data.weight_version)
        rollout = build_moss_tts_local_rollout_trace(
            prompt_rows=data.prompt_rows,
            decisions=torch.stack(data.output_decisions),
            decision_logprobs=torch.stack(data.output_decision_logprobs),
            codes=state.audio_codes,
            code_logprobs=(
                torch.stack(data.output_code_logprobs)
                if data.output_code_logprobs
                else torch.empty((0, n_vq), dtype=torch.float32)
            ),
            finish_reason=state.finish_reason,
            admission_weight_version=data.admission_weight_version,
            request_id=payload.request_id,
            sampling={
                name: getattr(data, name)
                for name in (
                    "text_temperature",
                    "text_top_p",
                    "text_top_k",
                    "audio_temperature",
                    "audio_top_p",
                    "audio_top_k",
                    "audio_repetition_penalty",
                    "seed",
                )
            },
            model_config=data.model_config,
        )
        # note (Zhang Yiyang): Keep tensor routing from walking every trace scalar.
        state.omni_rollout = json.dumps(rollout, separators=(",", ":")).encode()
        if not data.output_rows:
            state.audio_codes = None

    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_rows)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_moss_tts_local_scheduler_adapters(*, model: Any):
    """Build StagePayload <-> SGLang request adapters for MOSS-TTS Local."""

    def request_builder(payload: StagePayload) -> MossTTSLocalSGLangRequestData:
        return build_sglang_moss_tts_local_request(payload, model=model)

    def result_adapter(data: MossTTSLocalSGLangRequestData) -> StagePayload:
        try:
            return apply_sglang_moss_tts_local_result(data.stage_payload, data)
        finally:
            # note (Yue Yin): release the finished request's decode-state pool row
            # (mirrors higgs_tts/request_builders.py); recycles the row for a waiter.
            model.reset_request(data.stage_payload.request_id)

    return request_builder, result_adapter
