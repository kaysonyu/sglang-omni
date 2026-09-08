# SPDX-License-Identifier: Apache-2.0
"""SGLang causal-prefill teacher, without sampling, tokenizer or codec work."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids
from sglang_omni.models.moss_tts_local.engine_builder import MossTtsLocalEngineBuilder
from sglang_omni.models.moss_tts_local.rollout_trace import (
    moss_tts_local_model_identity,
    selected_action_logprobs,
)
from sglang_omni.models.moss_tts_local.scoring_protocol import LocalScoreInput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData


@dataclass
class ScoreRequestData(ARRequestData):
    score: LocalScoreInput | None = None
    rows: torch.Tensor | None = None
    score_result: dict | None = None
    stage_payload: Any = None
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    enforce_request_limits: bool = True
    admission_version: str | None = None


def build_score_request(payload, *, model):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import get_serving
    from sglang.srt.sampling.sampling_params import SamplingParams

    raw = (payload.request.params or {}).get("moss_local_score")
    if raw is None:
        raise ValueError("This pipeline only accepts /score_actions requests")
    score = LocalScoreInput.model_validate(raw)
    rows = torch.tensor(
        score.prompt_rows + [[151656, *row] for row in score.codes], dtype=torch.long
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
    req._moss_score_group = payload.request.params.get("moss_score_group")
    req._moss_score_group_size = int(
        payload.request.params.get("moss_score_group_size", 1)
    )
    req._moss_score_group_index = int(
        payload.request.params.get("moss_score_group_index", 0)
    )
    if (
        not 1 <= req._moss_score_group_size <= 8
        or not 0 <= req._moss_score_group_index < req._moss_score_group_size
    ):
        raise ValueError("Invalid atomic scoring group metadata")
    req._moss_score_group_created = time.perf_counter()
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
        admission_version=str(get_serving().weight_version),
    )


def score_result(data):
    if data.score_result is None:
        raise RuntimeError("Teacher returned without scoring the requested actions")
    if str(data.weight_version) != data.admission_version:
        raise RuntimeError("Teacher weights changed during scoring")
    return StagePayload(
        request_id=data.stage_payload.request_id,
        request=data.stage_payload.request,
        data={
            "modality": "text",
            "text": "",
            "omni_rollout": data.score_result,
            "weight_version": str(data.weight_version),
            "finish_reason": "stop",
        },
    )


@torch.no_grad()
def score_local_depth(
    model, hidden, decisions, padded_codes, *, temperature, chunk_size
):
    """Frames are independent batch rows; only the 12 depth steps are sequential."""
    decision_parts, code_parts = [], []
    for start in range(0, len(decisions), chunk_size):
        h = hidden[start : start + chunk_size]
        targets = padded_codes[start : start + chunk_size]
        temps = torch.full((len(h),), temperature, dtype=torch.float32, device=h.device)
        current = model.local_transformer.step(h.to(model.dtype), 0)
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
                    F.linear(current, model._audio_head_weight(depth)).float(),
                    targets[:, depth],
                    temps,
                )
            )
            if depth + 1 < model.n_vq:
                current = model.local_transformer.step(
                    F.embedding(
                        targets[:, depth], model._audio_embedding_weight(depth)
                    ).to(model.dtype),
                    depth + 1,
                )
        code_parts.append(torch.stack(columns, dim=-1))
    return torch.cat(decision_parts), torch.cat(code_parts)


class LocalScoreModelRunner(ModelRunner):
    def __init__(self, *args, score_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_chunk_size = score_chunk_size

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        if not schedule_batch.is_prefill_only:
            raise ValueError("Local scoring must not enter autoregressive generation")
        if any(
            int(n) != len(r.data.rows)
            for n, r in zip(forward_batch.extend_seq_lens_cpu, requests, strict=True)
        ):
            raise ValueError(
                "Score-only pipeline requires unchunked prefill without prefix cache reuse"
            )
        rows = torch.cat([r.data.rows for r in requests]).to(self.device)
        forward_batch.input_embeds = self.model._prepare_multi_modal_inputs(rows)
        return None

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        hidden = result.logits_output.hidden_states
        expected_rows = sum(len(request.data.rows) for request in requests)
        if hidden.ndim != 2 or len(hidden) != expected_rows:
            raise RuntimeError(
                f"Scoring requires all prefill hidden states: got {tuple(hidden.shape)}, expected {expected_rows}; score_only={getattr(self.model, '_moss_local_score_only', None)}"
            )
        batch_sha256 = hashlib.sha256(
            "".join(r.data.score.input_sha256() for r in requests).encode()
        ).hexdigest()
        cursor = 0
        started = time.perf_counter()
        for request in requests:
            data = request.data
            score = data.score
            d, f = len(score.decisions), len(score.codes)
            positions = torch.arange(
                len(score.prompt_rows) - 1,
                len(score.prompt_rows) - 1 + d,
                device=hidden.device,
            )
            selected = hidden[cursor : cursor + len(data.rows)].index_select(
                0, positions
            )
            cursor += len(data.rows)
            decisions = torch.tensor(
                score.decisions, dtype=torch.long, device=hidden.device
            )
            codes = torch.zeros((d, 12), dtype=torch.long, device=hidden.device)
            if f:
                codes[:f] = torch.tensor(
                    score.codes, dtype=torch.long, device=hidden.device
                )
            dl, cl = score_local_depth(
                self.model,
                selected,
                decisions,
                codes,
                temperature=score.temperature,
                chunk_size=self.score_chunk_size,
            )
            dl, cl = dl.float().cpu(), cl[:f].float().cpu()
            if not torch.isfinite(dl).all() or not torch.isfinite(cl).all():
                raise RuntimeError("Teacher scores must be finite")
            data.score_result = {
                "version": 1,
                "scoring_batch_sha256": batch_sha256,
                "score_chunk_size": self.score_chunk_size,
                "sample_id": score.sample_id,
                "input_sha256": score.input_sha256(),
                "decision_logprobs": dl.tolist(),
                "code_logprobs": cl.tolist(),
                "weight_version": data.admission_version,
                "temperature": score.temperature,
                "teacher_weight_sha256": self.model._moss_teacher_weight_sha256,
                "logprob_semantics": "temperature_scaled_full_vocab_v1",
                "model_identity": moss_tts_local_model_identity(self.model.config),
            }
        result.moss_score_depth_seconds = time.perf_counter() - started


class LocalScoreEngineBuilder(MossTtsLocalEngineBuilder):
    def __init__(self, *, score_chunk_size=128, frozen=True, **kwargs):
        super().__init__(**kwargs)
        self.score_chunk_size = score_chunk_size
        self.frozen = frozen

    def generation_defaults(self, *, dtype):
        defaults = super().generation_defaults(dtype=dtype)
        defaults.update(
            disable_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_running_requests=32,
            max_prefill_tokens=4096,
            max_total_tokens=8192,
            mem_fraction_static=0.2,
        )
        return defaults

    def setup_model(self, *, model_worker, checkpoint_dir, device, gpu_id, server_args):
        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        self.post_cuda_graph_setup(self.model, server_args)

    def post_cuda_graph_setup(self, model, server_args):
        if not server_args.disable_radix_cache or server_args.chunked_prefill_size > 0:
            raise ValueError(
                "Scoring requires disabled Radix Cache and unchunked prefill"
            )
        if server_args.quantization is not None:
            raise ValueError(
                "Local MOPD scoring currently requires unquantized weights"
            )
        model._moss_local_score_only = True
        model._moss_local_score_frozen = self.frozen

    def make_model_runner(self, model_worker, output_proc):
        return LocalScoreModelRunner(
            model_worker, output_proc, score_chunk_size=self.score_chunk_size
        )

    def make_adapters(self, model):
        return partial(build_score_request, model=model), score_result

    def make_abort_callback(self):
        return None

    def post_scheduler_setup(self, scheduler, model_runner):
        if not self.frozen:
            self.model._moss_teacher_weight_sha256 = None
            return
        from sglang_omni.model_runner.weight_checker import StrictWeightChecker

        class ParameterChecker(StrictWeightChecker):
            @staticmethod
            def _iter_named_tensors(model):
                # The synthetic decode input table is runtime staging, not a
                # checkpoint weight; it depends on concurrency and RNG state.
                return (
                    (name, value)
                    for name, value in model.named_parameters()
                    if not name.startswith("_decode_input_embedding.")
                )

        digest = ParameterChecker(SimpleNamespace(model=self.model)).checksum()
        self.model._moss_teacher_weight_sha256 = digest["per_gpu_checksum"]


def create_score_engine(
    model_path,
    *,
    device="cuda:0",
    gpu_id=None,
    dtype="bfloat16",
    server_args_overrides=None,
    total_gpu_memory_fraction=None,
    process_total_gpu_memory_fraction=None,
    score_chunk_size=128,
    frozen=True,
):
    return LocalScoreEngineBuilder(
        score_chunk_size=score_chunk_size,
        frozen=frozen,
        enable_async_decode=False,
        async_decode_min_batch_size=2,
        prefill_coalesce_requests=4,
        prefill_coalesce_wait_ms=2,
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
