# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local teacher scoring on the SGLang prefill scheduler."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.weight_checker import StrictWeightChecker
from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids
from sglang_omni.models.moss_tts_local.engine_builder import MossTtsLocalEngineBuilder
from sglang_omni.models.moss_tts_local.rollout_trace import (
    MOSS_TTS_LOCAL_LOGPROB_SEMANTICS,
    moss_tts_local_model_identity,
    selected_action_logprobs,
)
from sglang_omni.models.moss_tts_local.scoring_protocol import LocalScoreInput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData


@torch.no_grad()
def score_local_depth(
    model: Any,
    hidden: torch.Tensor,
    decisions: torch.Tensor,
    padded_codes: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    decision_parts, code_parts = [], []
    for start in range(0, len(decisions), chunk_size):
        current = model.local_transformer.step(
            hidden[start : start + chunk_size].to(model.dtype), 0
        )
        targets = padded_codes[start : start + chunk_size]
        temps = torch.full(
            (len(current),), temperature, dtype=torch.float32, device=hidden.device
        )
        decision_parts.append(
            selected_action_logprobs(
                F.linear(current, model.local_text_lm_head.weight).float(),
                decisions[start : start + chunk_size],
                temps,
            )
        )
        columns = []
        for depth in range(model.n_vq):
            columns.append(
                selected_action_logprobs(
                    F.linear(current, model.audio_head_weight(depth)).float(),
                    targets[:, depth],
                    temps,
                )
            )
            if depth + 1 < model.n_vq:
                current = model.local_transformer.step(
                    F.embedding(
                        targets[:, depth], model.audio_embedding_weight(depth)
                    ).to(model.dtype),
                    depth + 1,
                )
        code_parts.append(torch.stack(columns, dim=-1))
    return torch.cat(decision_parts), torch.cat(code_parts)


@dataclass
class ScoreRequestData(ARRequestData):
    score: LocalScoreInput | None = None
    rows: torch.Tensor | None = None
    score_result: dict[str, Any] | None = None
    stage_payload: StagePayload | None = None
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    enforce_request_limits: bool = True
    input_embeds_are_projected: bool = True


def build_score_request(payload: StagePayload, *, model: Any) -> ScoreRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    raw = (payload.request.params or {}).get("moss_local_score")
    if raw is None:
        raise ValueError("This pipeline only accepts /score_actions requests")
    score = LocalScoreInput.model_validate(raw)
    score.validate_model(model.config, model.score_context_length)
    rows = torch.tensor(
        score.prompt_rows
        + [
            [int(model.config.audio_assistant_slot_token_id), *row]
            for row in score.codes
        ],
        dtype=torch.long,
    )
    token_ids = build_row_cache_key_ids(rows)
    sampling = SamplingParams(max_new_tokens=0, temperature=1.0)
    sampling.normalize(None)
    sampling.verify(int(model.config.vocab_size_list[0]))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=token_ids,
        sampling_params=sampling,
        vocab_size=int(model.config.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    return ScoreRequestData(
        input_ids=torch.tensor(token_ids),
        max_new_tokens=0,
        temperature=1.0,
        req=req,
        score=score,
        rows=rows,
        stage_payload=payload,
        input_embeds_are_projected=True,
    )


def score_result(data: ScoreRequestData) -> StagePayload:
    if data.score_result is None:
        raise RuntimeError("Teacher returned without scoring the requested actions")
    return StagePayload(
        request_id=data.stage_payload.request_id,
        request=data.stage_payload.request,
        data={
            "modality": "text",
            "text": "",
            "omni_rollout": json.dumps(
                data.score_result, separators=(",", ":")
            ).encode(),
            "weight_version": data.score_result["weight_version"],
            "finish_reason": "stop",
        },
    )


class LocalScoreModelRunner(ModelRunner):
    def __init__(
        self, model_worker: Any, output_processor: Any, *, score_chunk_size: int
    ):
        super().__init__(model_worker, output_processor)
        self.score_chunk_size = score_chunk_size

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        if not schedule_batch.is_prefill_only:
            raise ValueError("Local scoring requires prefill-only requests")
        if any(
            int(size) != len(request.data.rows)
            for size, request in zip(
                forward_batch.extend_seq_lens_cpu, requests, strict=True
            )
        ):
            raise ValueError("Scoring requires unchunked prefill without prefix reuse")
        rows = torch.cat([request.data.rows for request in requests]).to(self.device)
        forward_batch.input_embeds = self.model.prepare_multi_modal_inputs(rows)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        hidden = result.logits_output.hidden_states
        expected = sum(len(request.data.rows) for request in requests)
        if hidden.ndim != 2 or len(hidden) != expected:
            raise RuntimeError("Scoring requires every prefill hidden state")
        batch_hash = hashlib.sha256(
            "".join(r.data.score.input_sha256() for r in requests).encode()
        ).hexdigest()
        cursor = 0
        for request in requests:
            data = request.data
            score = data.score
            decisions_count, frames = len(score.decisions), len(score.codes)
            begin = cursor + len(score.prompt_rows) - 1
            selected = hidden[begin : begin + decisions_count]
            cursor += len(data.rows)
            decisions = torch.tensor(
                score.decisions, dtype=torch.long, device=hidden.device
            )
            codes = torch.zeros(
                (decisions_count, self.model.n_vq),
                dtype=torch.long,
                device=hidden.device,
            )
            if frames:
                codes[:frames] = torch.tensor(
                    score.codes, dtype=torch.long, device=hidden.device
                )
            decision_lp, code_lp = score_local_depth(
                self.model,
                selected,
                decisions,
                codes,
                temperature=score.temperature,
                chunk_size=self.score_chunk_size,
            )
            decision_lp, code_lp = decision_lp.cpu(), code_lp[:frames].cpu()
            if not bool(torch.isfinite(decision_lp).all()) or not bool(
                torch.isfinite(code_lp).all()
            ):
                raise RuntimeError("Teacher scores must be finite")
            data.score_result = {
                "version": 1,
                "sample_id": score.sample_id,
                "input_sha256": score.input_sha256(),
                "scoring_batch_sha256": batch_hash,
                "score_chunk_size": self.score_chunk_size,
                "decision_logprobs": decision_lp.tolist(),
                "code_logprobs": code_lp.tolist(),
                "weight_version": self.model.teacher_weight_version,
                "temperature": score.temperature,
                "teacher_weight_sha256": self.model.teacher_weight_sha256,
                "logprob_semantics": MOSS_TTS_LOCAL_LOGPROB_SEMANTICS,
                "model_identity": moss_tts_local_model_identity(self.model.config),
            }


class TeacherWeightChecker(StrictWeightChecker):
    @staticmethod
    def iter_named_tensors(model: Any):
        # note (Zhang Yiyang): Decode staging is not part of the checkpoint.
        return (
            (name, value)
            for name, value in model.named_parameters()
            if not name.startswith("_decode_input_embedding.")
        )


class LocalScoreEngineBuilder(MossTtsLocalEngineBuilder):
    def __init__(self, *, score_chunk_size: int, **kwargs: Any):
        super().__init__(**kwargs)
        if score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive")
        self.score_chunk_size = score_chunk_size

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults = super().generation_defaults(dtype=dtype)
        defaults.update(
            disable_cuda_graph=True, disable_radix_cache=True, chunked_prefill_size=-1
        )
        return defaults

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        if (
            not server_args.disable_cuda_graph
            or not server_args.disable_radix_cache
            or server_args.chunked_prefill_size > 0
        ):
            raise ValueError(
                "Scoring requires disabled CUDA graphs, disabled radix cache and unchunked prefill"
            )
        if server_args.quantization is not None:
            raise ValueError("Local teacher scoring requires unquantized weights")
        self.model.score_only = True
        self.model.score_context_length = self.context_length
        model_worker._strict_weight_checker = TeacherWeightChecker(
            model_worker.model_runner
        )

    def make_model_runner(
        self, model_worker: Any, output_proc: Any
    ) -> LocalScoreModelRunner:
        return LocalScoreModelRunner(
            model_worker, output_proc, score_chunk_size=self.score_chunk_size
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return partial(build_score_request, model=model), score_result

    def make_abort_callback(self) -> None:
        return None

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        from sglang.srt.runtime_context import get_serving

        digest = TeacherWeightChecker(SimpleNamespace(model=self.model)).checksum()
        self.model.teacher_weight_sha256 = digest["per_gpu_checksum"]
        self.model.teacher_weight_version = str(
            get_serving().weight_version or self.model.teacher_weight_sha256
        )


def create_score_engine(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    score_chunk_size: int = 128,
) -> Any:
    return LocalScoreEngineBuilder(
        score_chunk_size=score_chunk_size,
        enable_async_decode=False,
        async_decode_min_batch_size=2,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        process_total_gpu_memory_fraction=process_total_gpu_memory_fraction,
        codec_mem_reserve=0.0,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
