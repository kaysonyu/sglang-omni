# SPDX-License-Identifier: Apache-2.0
"""Frozen Prism scoring through prefill-only scheduler requests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.model_runner.weight_checker import StrictWeightChecker
from sglang_omni.models.moss_tts_prism.embedding_kernels import feedback_embeddings
from sglang_omni.models.moss_tts_prism.scoring_protocol import (
    PrismScoreInput,
    PrismScoreResult,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismInputs,
)
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.logprobs import selected_action_logprobs
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import ARRequestData, SchedulerRequest


@dataclass(kw_only=True)
class PrismScoreInputs(PrismInputs):
    prediction_indices: torch.Tensor
    actions: torch.Tensor
    temperatures: torch.Tensor


@torch.no_grad()
def score_prism_actions(
    model: MossTTSPrismSGLangModel,
    inputs: PrismScoreInputs,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    chunk_size: int,
) -> LogitsProcessorOutput:
    hidden = model.prepare_inputs(inputs)
    predictions = len(inputs.prediction_indices)
    code_logprobs = torch.empty(
        predictions, model.n_vq, dtype=torch.float32, device=hidden.device
    )
    stop_probabilities = torch.empty(
        predictions, dtype=torch.float32, device=hidden.device
    )

    def read_site(index: int, value: torch.Tensor) -> torch.Tensor | None:
        heads = model.topology.execution_units[index].rvq_heads
        if not heads:
            return None
        norm = model.prediction_norms[
            model.topology.prediction_norm_index_by_execution_unit[index]
        ]
        for start in range(0, predictions, chunk_size):
            stop = start + chunk_size
            normalized = norm(value[inputs.prediction_indices[start:stop]])
            if 1 in heads:
                with torch.autocast(device_type=value.device.type, enabled=False):
                    logits = F.linear(
                        normalized.float(), model.stop_head.weight.float()
                    )
                stop_probabilities[start:stop] = logits.softmax(dim=-1)[:, 1]
            for head in heads:
                logits = model.audio_lm_heads[head - 1](normalized)
                code_logprobs[start:stop, head - 1] = selected_action_logprobs(
                    logits,
                    inputs.actions[start:stop, head - 1],
                    inputs.temperatures[start:stop],
                )
        if index == len(model.topology.execution_units) - 1:
            return None
        if value.is_cuda:
            return feedback_embeddings(
                inputs.successor_codes,
                inputs.successor_mask,
                tuple(model.audio_weight(head - 1) for head in heads),
                tuple(head - 1 for head in heads),
            )
        else:
            delta = torch.zeros_like(value)
            for head in heads:
                delta += (
                    F.embedding(
                        inputs.successor_codes[:, head - 1],
                        model.audio_weight(head - 1),
                    )
                    * inputs.successor_mask[:, None]
                )
            return delta

    model.execute_schedule(hidden, positions, forward_batch, read_site)
    return LogitsProcessorOutput(
        next_token_logits=None,
        customized_info={
            "code_logprobs": code_logprobs,
            "stop_probabilities": stop_probabilities,
        },
    )


@dataclass
class PrismScoreRequestData(ARRequestData):
    req: Req | None = None
    score: PrismScoreInput | None = None
    inputs: PrismInputs | None = None
    score_result: PrismScoreResult | None = None
    stage_payload: StagePayload | None = None
    synced: bool = False
    generation_steps: int = 0
    enforce_request_limits: bool = True


class PrismScoreResultAdapter(Protocol):
    def __call__(self, request: PrismScoreRequestData) -> StagePayload: ...


def build_score_request(
    payload: StagePayload, *, model: MossTTSPrismSGLangModel, context_length: int
) -> PrismScoreRequestData:
    raw = (payload.request.params or {}).get("moss_prism_score")
    try:
        score = PrismScoreInput.model_validate(raw)
        score.validate_model(model.config, context_length)
    except ValueError as error:
        raise ValueError(f"Prism score input: {error}") from error
    prompt_rows = len(score.prompt_rows)
    predictions = score.prediction_count
    rows = torch.tensor(
        score.prompt_rows
        + [
            [model.config.text_pad_idx, *codes]
            for codes in score.codes[: predictions - 1]
        ],
        dtype=torch.long,
    )
    kind = torch.cat(
        (torch.tensor(score.item_kind), torch.ones(predictions - 1, dtype=torch.long))
    )
    role = torch.cat(
        (
            torch.tensor(score.audio_role),
            torch.full((predictions - 1,), 2, dtype=torch.long),
        )
    )
    retention = torch.zeros(predictions - 1, model.n_vq, dtype=torch.bool)
    retention[:, [channel - 1 for channel in model.config.prism_input_rvq_channels]] = (
        True
    )
    retention = torch.cat(
        (torch.tensor(score.target_history_retention_mask), retention)
    )
    successors = torch.cat(
        (
            torch.tensor(score.successor_audio_codes),
            torch.zeros(predictions - 1, model.n_vq, dtype=torch.long),
        )
    )
    mask = torch.cat(
        (
            torch.tensor(score.successor_audio_mask),
            torch.zeros(predictions - 1, dtype=torch.bool),
        )
    )
    if score.codes:
        successors[prompt_rows - 1 : prompt_rows - 1 + len(score.codes)] = torch.tensor(
            score.codes
        )
        mask[prompt_rows - 1 : prompt_rows - 1 + len(score.codes)] = True
    inputs = PrismInputs(
        rows=rows,
        item_kind=kind,
        audio_role=role,
        retention=retention,
        successor_mask=mask,
        successor_codes=successors,
    )
    sampling = SamplingParams(max_new_tokens=0, temperature=1.0)
    sampling.normalize(None)
    sampling.verify(model.config.language_config.vocab_size)
    request = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=rows[:, 0].tolist(),
        sampling_params=sampling,
        vocab_size=model.config.language_config.vocab_size,
    )
    return PrismScoreRequestData(
        input_ids=rows[:, 0],
        req=request,
        max_new_tokens=0,
        score=score,
        inputs=inputs,
        stage_payload=payload,
    )


def score_result(request: PrismScoreRequestData) -> StagePayload:
    assert request.stage_payload is not None
    if request.score_result is None:
        raise RuntimeError(
            "Prism teacher returned without scoring the supplied actions"
        )
    return StagePayload(
        request_id=request.stage_payload.request_id,
        request=request.stage_payload.request,
        data={
            "modality": "text",
            "text": "",
            "omni_rollout": json.dumps(
                request.score_result, separators=(",", ":"), allow_nan=False
            ).encode(),
            "weight_version": request.score_result["weight_version"],
            "finish_reason": "stop",
        },
    )


class PrismScoreModelRunner(ModelRunner):
    model: MossTTSPrismSGLangModel

    def __init__(
        self,
        model_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        score_chunk_size: int,
    ) -> None:
        super().__init__(model_worker, output_processor)
        self.score_chunk_size = score_chunk_size

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult:
        assert schedule_batch.is_prefill_only
        indices, actions, temperatures = [], [], []
        cursor = 0
        inputs = []
        for length, request in zip(
            forward_batch.extend_seq_lens_cpu, requests, strict=True
        ):
            sample = request.data.score
            carrier = request.data.inputs
            assert int(length) == len(carrier.rows)
            indices.extend(
                range(cursor + len(sample.prompt_rows) - 1, cursor + len(carrier.rows))
            )
            actions.extend(
                sample.codes
                + ([[0] * self.model.n_vq] if sample.finish_reason == "stop" else [])
            )
            temperatures.extend([sample.temperature] * sample.prediction_count)
            inputs.append(carrier)
            cursor += len(carrier.rows)
        score_inputs = PrismScoreInputs(
            **{
                name: torch.cat([getattr(carrier, name) for carrier in inputs]).to(
                    self.device
                )
                for name in (
                    "rows",
                    "item_kind",
                    "audio_role",
                    "retention",
                    "successor_mask",
                    "successor_codes",
                )
            },
            prediction_indices=torch.tensor(
                indices, dtype=torch.long, device=self.device
            ),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            temperatures=torch.tensor(
                temperatures, dtype=torch.float32, device=self.device
            ),
        )

        backend = self.tp_worker.model_runner.attn_backend
        backend.init_forward_metadata(forward_batch)
        with attn_forward_context(backend):
            output = score_prism_actions(
                self.model,
                score_inputs,
                forward_batch.positions,
                forward_batch,
                chunk_size=self.score_chunk_size,
            )
        return GenerationBatchResult(logits_output=output, can_run_cuda_graph=False)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        scores = result.logits_output.customized_info
        code_logprobs = scores["code_logprobs"].cpu()
        probabilities = scores["stop_probabilities"].cpu()
        if (
            not torch.isfinite(code_logprobs).all()
            or not torch.isfinite(probabilities).all()
        ):
            raise RuntimeError("Prism teacher scores must be finite")
        cursor = 0
        for request in requests:
            sample = request.data.score
            request.data.score_result = PrismScoreResult(
                version=1,
                sample_id=sample.sample_id,
                input_sha256=sample.input_sha256(),
                code_logprobs=code_logprobs[
                    cursor : cursor + len(sample.codes)
                ].tolist(),
                stop_probabilities=probabilities[
                    cursor : cursor + sample.prediction_count
                ].tolist(),
                finish_reason=sample.finish_reason,
                temperature=sample.temperature,
                weight_version=self.model.teacher_weight_version,
                teacher_weight_sha256=self.model.teacher_weight_sha256,
                logprob_semantics="temperature_scaled_full_vocab_v1",
                stop_semantics="pre_frame_threshold_v1",
                model_identity=self.model.model_identity,
            )
            cursor += sample.prediction_count


class PrismTeacherWeightChecker(StrictWeightChecker):
    @staticmethod
    def iter_named_tensors(
        model: MossTTSPrismSGLangModel,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # note (Zhang Yiyang): Runtime caches are not frozen checkpoint parameters.
        return model.named_parameters()
