# SPDX-License-Identifier: Apache-2.0
"""Typed prompt and scheduler request adapters for MOSS-TTS Prism."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_serving
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import ProcessorMixin

from sglang_omni.models.moss_tts.request_builders import (
    _DATA_URI_RE,
    derive_moss_tts_sampling_seed,
    new_moss_tts_sampling_seed,
)
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.moss_tts_local.request_builders import (
    MOSS_STREAM_TRANSPORT_BATCH_FRAMES,
    build_moss_tts_local_state,
    build_moss_tts_local_stream_metadata,
)
from sglang_omni.models.moss_tts_local.stages import MossLocalReferenceEncoder
from sglang_omni.models.moss_tts_prism.rollout_trace import build_prism_rollout_trace
from sglang_omni.models.moss_tts_prism.sglang_model import MossTTSPrismSGLangModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import ARRequestData, RequestOutput


class PrismReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    uri: str = Field(min_length=1)


class PrismPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    script: str = Field(min_length=1)
    global_instruction: str | None = None
    task_type: str = Field(min_length=1)
    show_language: bool
    language: str | None = None
    tokens_control: bool = False
    global_tokens: int | None = Field(default=None, gt=0, strict=True)
    references: list[PrismReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_controls(self) -> PrismPrompt:
        if not self.script.strip() or not self.task_type.strip():
            raise ValueError("Prism script and task_type must not be blank")
        elif self.show_language and (not self.language or not self.language.strip()):
            raise ValueError("Prism show_language requires language")
        elif self.tokens_control and self.global_tokens is None:
            raise ValueError("Prism tokens_control requires global_tokens")
        elif not self.tokens_control and self.global_tokens is not None:
            raise ValueError("Prism global_tokens requires tokens_control")
        else:
            return self


@dataclass
class PrismRequestData(ARRequestData):
    enforce_request_limits: bool = True
    req: Req | None = None
    synced: bool = False
    generation_steps: int = 0
    stage_payload: StagePayload | None = None
    state: MossTTSLocalState = field(default_factory=MossTTSLocalState)
    prompt: dict[str, torch.Tensor] = field(default_factory=dict)
    output_codes: list[torch.Tensor] = field(default_factory=list)
    sampling_seed: int = field(default_factory=new_moss_tts_sampling_seed)
    engine_start_s: float = 0.0
    stream_metadata: dict[str, int | float | str | bool] | None = None
    stream_pending_rows: list[torch.Tensor] = field(default_factory=list)
    stream_first_batch_sent: bool = False
    code_logprobs: list[torch.Tensor] = field(default_factory=list)
    stop_probabilities: list[torch.Tensor] = field(default_factory=list)
    admission_weight_version: str | None = None
    model_identity: dict[str, str | int | bool] | None = None


class PrismStreamOutputBuilder:
    def __call__(
        self, request_id: str, data: PrismRequestData, output: RequestOutput
    ) -> list[OutgoingMessage]:
        if (
            data.stream_metadata is None
            or output.data in data.req.sampling_params.stop_token_ids
        ):
            return []
        codes = data.output_codes[-1]
        row = torch.cat((data.prompt["input_ids"].new_zeros(1), codes))
        data.stream_pending_rows.append(row)
        threshold = (
            MOSS_STREAM_TRANSPORT_BATCH_FRAMES if data.stream_first_batch_sent else 1
        )
        if len(data.stream_pending_rows) < threshold:
            return []
        return self.flush(request_id, data)

    def flush(self, request_id: str, data: PrismRequestData) -> list[OutgoingMessage]:
        if not data.stream_pending_rows:
            return []
        rows = torch.stack(data.stream_pending_rows)
        data.stream_pending_rows.clear()
        data.stream_first_batch_sent = True
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target="vocoder",
                data=rows,
                metadata=data.stream_metadata,
            )
        ]


def build_prism_prompt(
    processor: ProcessorMixin, message: dict[str, object]
) -> dict[str, torch.Tensor]:
    """Pack a generation prompt without allocating tensors for individual tokens."""
    message = processor._normalize_user(message)
    references = message["audio_codes_list"]
    content = processor._replace_placeholders(
        message["content"], references, role="user"
    )
    content = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        chat_template=processor.chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )
    config = processor.model_config
    tokens = torch.tensor(
        processor.tokenizer.encode(content, add_special_tokens=False)
        + [config.audio_start_token_id],
        dtype=torch.long,
    )
    count = tokens.numel()
    rows = torch.zeros(count, config.n_vq + 1, dtype=torch.long)
    rows[:, 0] = tokens
    kinds = torch.zeros(count, dtype=torch.long)
    successors = torch.zeros(count, config.n_vq, dtype=torch.long)
    successor_mask = torch.zeros(count, dtype=torch.bool)
    if references:
        codes = torch.cat(references)
        audio_rows = (tokens == config.audio_user_slot_token_id).nonzero().flatten()
        rows[audio_rows, 0] = config.text_pad_idx
        rows[audio_rows, 1:] = codes
        kinds[audio_rows] = 1
        # note (Zhang Yiyang): Each reference frame conditions its preceding row.
        successors[audio_rows - 1] = codes
        successor_mask[audio_rows - 1] = True
    return {
        "input_ids": rows,
        "attention_mask": torch.ones(count, dtype=torch.bool),
        "position_ids": torch.arange(count),
        "item_kind": kinds,
        "audio_role": kinds.clone(),
        "target_history_retention_mask": torch.ones(
            count, config.n_vq, dtype=torch.bool
        ),
        "successor_audio_mask": successor_mask,
        "successor_audio_codes": successors,
    }


def preprocess_prism_payload(
    payload: StagePayload,
    *,
    processor: ProcessorMixin,
    reference_encoder: MossLocalReferenceEncoder | None,
) -> StagePayload:
    state = build_moss_tts_local_state(payload)
    params = dict(payload.request.params or {})
    params.update((params.get("stage_params") or {}).get("tts_engine") or {})
    for name in ("audio_temperature", "audio_top_p", "audio_repetition_penalty"):
        if not math.isfinite(state.generation_kwargs[name]):
            raise ValueError(f"MOSS-TTS Prism {name} must be finite")
    for name in ("text_temperature", "text_top_p", "text_top_k"):
        state.generation_kwargs.pop(name)
    tts_params = (payload.request.metadata or {}).get("tts_params") or {}
    for source in (params, tts_params):
        if any(
            source.get(name) is not None
            for name in (
                "text_temperature",
                "text_top_p",
                "text_top_k",
                "execution_rvq_channels",
                "returned_rvq_channels",
                "loop_block_prefix_counts",
            )
        ):
            raise ValueError(
                "MOSS-TTS Prism supports audio sampling and full RVQ execution only"
            )
    inputs = payload.request.inputs
    if isinstance(inputs, str) or (
        isinstance(inputs, dict)
        and "script" not in inputs
        and ("text" in inputs or "input" in inputs)
    ):
        if isinstance(inputs, dict) and len(inputs.get("references") or []) > 1:
            raise ValueError(
                "MOSS-TTS Prism multiple references require a structured prompt"
            )
        requested_task_type = tts_params.get("task_type") or params.get("task_type")
        if requested_task_type in (None, "Base"):
            if state.ref_audio is not None:
                task_type = (
                    "Instruction Voice Clone"
                    if state.instructions
                    else "Full VoiceClone"
                )
            elif state.instructions:
                task_type = "Instruction"
            else:
                task_type = "TTS"
        elif requested_task_type == "VoiceDesign":
            task_type = "Instruction"
        else:
            task_type = requested_task_type
        structured = PrismPrompt(
            script=state.text,
            global_instruction=state.instructions,
            task_type=task_type,
            show_language=state.language is not None,
            language=state.language,
            tokens_control=state.token_count is not None,
            global_tokens=state.token_count,
            references=(
                [PrismReference(id="audio1", uri=state.ref_audio)]
                if state.ref_audio is not None
                else []
            ),
        )
    else:
        structured = PrismPrompt.model_validate(inputs)
        if state.language is not None:
            raise ValueError("MOSS-TTS Prism language belongs in the structured prompt")
        elif state.token_count is not None:
            raise ValueError(
                "MOSS-TTS Prism global_tokens belongs in the structured prompt"
            )
    references = []
    for reference in structured.references:
        uri = reference.uri
        codes = (
            reference_encoder.encode_data_uri(uri)
            if _DATA_URI_RE.match(uri)
            else reference_encoder.encode(uri)
        )
        references.append(codes)
    message = processor.build_user_message(
        script=structured.script,
        reference=references or None,
        global_instruction=structured.global_instruction,
        tokens=structured.global_tokens if structured.tokens_control else None,
        task_type=structured.task_type,
        language=structured.language if structured.show_language else None,
    )
    prompt = build_prism_prompt(processor, message)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={**state.to_dict(), "prism_inputs": prompt},
    )


def build_prism_request(
    payload: StagePayload, *, model: MossTTSPrismSGLangModel
) -> PrismRequestData:
    state = MossTTSLocalState.from_dict(payload.data)
    prompt = payload.data["prism_inputs"]
    cfg = model.config
    admission_weight_version = None
    if model.enable_rl != state.return_omni_rollout:
        raise ValueError(
            "Prism RL request requires enable_rl=true and return_omni_rollout=true together"
        )
    if model.enable_rl:
        if not state.return_logprob:
            raise ValueError("Prism RL request requires return_logprob=true")
        for name, expected in (
            ("audio_top_p", 1.0),
            ("audio_top_k", -1),
            ("audio_repetition_penalty", 1.0),
        ):
            if state.generation_kwargs[name] != expected:
                raise ValueError(f"Prism RL request requires {name}={expected}")
        if (
            not math.isfinite(state.generation_kwargs["audio_temperature"])
            or state.generation_kwargs["audio_temperature"] <= 0
        ):
            raise ValueError(
                "Prism RL request requires a finite positive audio temperature"
            )
        version = get_serving().weight_version
        if version is None:
            raise RuntimeError("Prism rollout requires a weight version")
        admission_weight_version = str(version)
    max_new_tokens = state.generation_kwargs["max_new_tokens"]
    sampling = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[cfg.audio_end_token_id],
    )
    sampling.normalize(None)
    sampling.verify(cfg.language_config.vocab_size)
    token_ids = prompt["input_ids"][:, 0].tolist()
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=token_ids,
        sampling_params=sampling,
        eos_token_ids={cfg.audio_end_token_id},
        vocab_size=cfg.language_config.vocab_size,
    )
    seed = state.generation_kwargs.get("seed")
    return PrismRequestData(
        input_ids=torch.tensor(token_ids, dtype=torch.long),
        max_new_tokens=max_new_tokens,
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
        state=state,
        prompt=prompt,
        model_identity=model.model_identity,
        admission_weight_version=admission_weight_version,
        sampling_seed=(
            derive_moss_tts_sampling_seed(seed)
            if seed is not None
            else new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
        stream_metadata=build_moss_tts_local_stream_metadata(payload, n_vq=cfg.n_vq),
    )


def apply_prism_result(data: PrismRequestData) -> StagePayload:
    state = data.state
    if not data.output_codes and not state.return_omni_rollout:
        raise RuntimeError(
            "MOSS-TTS Prism generated no audio frames. Please retry the request."
        )
    payload = data.stage_payload
    assert payload is not None
    channels = data.prompt["input_ids"].shape[1] - 1
    codes = (
        torch.stack(data.output_codes).cpu()
        if data.output_codes
        else torch.empty((0, channels), dtype=torch.long)
    )
    if state.return_omni_rollout:
        if data.admission_weight_version is None or data.weight_version is None:
            raise RuntimeError("Prism rollout is missing its weight version")
        if data.admission_weight_version != str(data.weight_version):
            raise RuntimeError("Prism rollout crossed a weight update")
        assert data.model_identity is not None
        state.finish_reason = data.finish_reason
        state.weight_version = str(data.weight_version)
        state.omni_rollout = build_prism_rollout_trace(
            prompt=data.prompt,
            codes=codes,
            code_logprobs=(
                torch.stack(data.code_logprobs)
                if data.code_logprobs
                else torch.empty((0, channels), dtype=torch.float32)
            ),
            stop_probabilities=torch.stack(data.stop_probabilities),
            finish_reason=data.finish_reason,
            request_id=payload.request_id,
            admission_weight_version=data.admission_weight_version,
            model_identity=data.model_identity,
            sampling={**state.generation_kwargs, "sampling_seed": data.sampling_seed},
        )
    state.audio_codes = (
        codes
        if len(codes) and (payload.request.params or {}).get("return_audio", True)
        else None
    )
    state.prompt_tokens = len(data.prompt["input_ids"])
    state.completion_tokens = len(codes)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=state.to_dict()
    )
