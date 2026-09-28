# SPDX-License-Identifier: Apache-2.0
"""Packed prefill, frame sampling and streaming handoff for Prism."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import PretrainedConfig

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.sampling_kernels import sample_seeded_fused
from sglang_omni.models.moss_tts_prism.request_builders import PrismRequestData
from sglang_omni.models.moss_tts_prism.rollout_trace import PRISM_STOP_THRESHOLD
from sglang_omni.models.moss_tts_prism.sglang_model import PrismBatchInputs
from sglang_omni.sampling.logprobs import selected_action_logprobs
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


def prism_request_inputs(
    data: PrismRequestData, config: PretrainedConfig, *, prefill: bool
) -> dict[str, torch.Tensor]:
    """Rebuild chronological successor edges when a request is re-prefilled."""
    if not data.output_codes:
        return data.prompt
    codes = torch.stack(data.output_codes if prefill else data.output_codes[-1:])
    count = len(codes)
    generated = {
        "input_ids": torch.cat(
            (torch.full((count, 1), config.text_pad_idx), codes), dim=1
        ),
        "item_kind": torch.ones(count, dtype=torch.long),
        "audio_role": torch.full((count,), 2, dtype=torch.long),
        "target_history_retention_mask": torch.zeros(
            (count, config.n_vq), dtype=torch.bool
        ),
        "successor_audio_mask": torch.ones(count, dtype=torch.bool),
        "successor_audio_codes": torch.zeros_like(codes),
    }
    generated["target_history_retention_mask"][
        :, [c - 1 for c in config.prism_input_rvq_channels]
    ] = True
    generated["successor_audio_mask"][-1] = False
    generated["successor_audio_codes"][:-1] = codes[1:]
    if not prefill:
        return {name: value[-1:] for name, value in generated.items()}
    prompt = {name: data.prompt[name].clone() for name in generated}
    prompt["successor_audio_mask"][-1] = True
    prompt["successor_audio_codes"][-1] = codes[0]
    return {name: torch.cat((prompt[name], value)) for name, value in generated.items()}


class MossTTSPrismModelRunner(ModelRunner):
    def prepare(self, requests: list[SchedulerRequest], *, prefill: bool) -> None:
        request_states = [request.data for request in requests]
        device = self.model.device
        params = [data.state.generation_kwargs for data in request_states]
        decode = self.model.decode_inputs
        batch_size = len(request_states)
        if not prefill and decode is not None:
            decode.rows_cpu[:batch_size, 1:].copy_(
                torch.stack([data.output_codes[-1] for data in request_states])
            )
            penalties = [p["audio_repetition_penalty"] for p in params]
            decode.floats_cpu[:, :batch_size].copy_(
                torch.tensor(
                    [
                        [p["audio_temperature"] for p in params],
                        [p["audio_top_p"] for p in params],
                        penalties,
                    ],
                    dtype=torch.float32,
                )
            )
            decode.ints_cpu[:, :batch_size].copy_(
                torch.tensor(
                    [
                        [p["audio_top_k"] for p in params],
                        [data.sampling_seed for data in request_states],
                        [
                            len(data.output_codes) * self.model.n_vq
                            for data in request_states
                        ],
                    ],
                    dtype=torch.long,
                )
            )
            decode.rows.copy_(decode.rows_cpu, non_blocking=True)
            decode.sampling_floats.copy_(decode.floats_cpu, non_blocking=True)
            decode.sampling_ints.copy_(decode.ints_cpu, non_blocking=True)
            if any(p != 1.0 for p in penalties):
                history = torch.zeros(
                    (batch_size, self.model.n_vq, self.model.config.speech_vocab_size),
                    dtype=torch.bool,
                )
                for row, data in enumerate(request_states):
                    if data.output_codes:
                        history[row].scatter_(1, torch.stack(data.output_codes).T, True)
                decode.history[:batch_size].copy_(history)
            return

        pieces = [
            prism_request_inputs(data, self.model.config, prefill=prefill)
            for data in request_states
        ]
        indices = []
        offset = 0
        for piece, data in zip(pieces, request_states):
            length = len(piece["input_ids"])
            if prefill:
                assert not len(data.req.prefix_indices)
                assert data.req.extend_range.length == length
            offset += length
            indices.append(offset - 1)
        temperatures = torch.tensor(
            [p["audio_temperature"] for p in params], device=device
        )
        top_p = torch.tensor([p["audio_top_p"] for p in params], device=device)
        top_k = torch.tensor([p["audio_top_k"] for p in params], device=device)
        seeds = torch.tensor(
            [data.sampling_seed for data in request_states], device=device
        )
        positions = torch.tensor(
            [len(data.output_codes) * self.model.n_vq for data in request_states],
            device=device,
        )

        penalty_histories = [
            (
                row,
                parameters["audio_repetition_penalty"],
                torch.stack(data.output_codes).to(device),
            )
            for row, (data, parameters) in enumerate(zip(request_states, params))
            if parameters["audio_repetition_penalty"] != 1.0 and data.output_codes
        ]

        code_logprobs = (
            torch.empty(batch_size, self.model.n_vq, device=device, dtype=torch.float32)
            if self.model.enable_rl
            else None
        )

        def sample(logits: torch.Tensor, channel: int) -> torch.Tensor:
            for row, penalty, history in penalty_histories:
                tokens = history[:, channel]
                scores = logits[row, tokens]
                logits[row, tokens] = torch.where(
                    scores < 0, scores * penalty, scores / penalty
                )
            if logits.is_cuda:
                return sample_seeded_fused(
                    logits,
                    temperature=temperatures,
                    top_p=top_p,
                    top_k=top_k,
                    seeds=seeds,
                    positions=positions + channel,
                    full_vocab_logprobs=(
                        code_logprobs[:, channel] if code_logprobs is not None else None
                    ),
                )
            else:
                selected = MossTTSModelRunner.sample_tokens(
                    logits,
                    temperature=temperatures,
                    top_p=top_p,
                    top_k=top_k,
                    seeds=seeds,
                    positions=positions + channel,
                )
                if code_logprobs is not None:
                    code_logprobs[:, channel] = selected_action_logprobs(
                        logits, selected, temperatures
                    )
                return selected

        packed = {
            name: torch.cat([piece[name] for piece in pieces]).to(device)
            for name in (
                "input_ids",
                "item_kind",
                "audio_role",
                "target_history_retention_mask",
                "successor_audio_mask",
                "successor_audio_codes",
            )
        }
        self.model.batch_inputs = PrismBatchInputs(
            rows=packed["input_ids"],
            item_kind=packed["item_kind"],
            audio_role=packed["audio_role"],
            retention=packed["target_history_retention_mask"],
            successor_mask=packed["successor_audio_mask"],
            successor_codes=packed["successor_audio_codes"],
            sample_indices=torch.tensor(indices, device=device),
            sample=sample,
            code_logprobs=code_logprobs,
        )

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.prepare(requests, prefill=True)

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        self.prepare(requests, prefill=False)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        cfg = self.model.config
        stop_probabilities = result.logits_output.next_token_logits.float().softmax(
            dim=-1
        )[:, 1]
        if self.model.enable_rl:
            result.logits_output.customized_info["stop_probabilities"] = (
                stop_probabilities
            )
        result.next_token_ids = torch.where(
            stop_probabilities > PRISM_STOP_THRESHOLD,
            cfg.audio_end_token_id,
            cfg.audio_assistant_gen_slot_token_id,
        )
        self.stage_token_ids(result, result.next_token_ids)
        self.model.batch_inputs = None

    post_prefill = post_decode

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        codes = result.logits_output.customized_info["audio_codes"][
            : len(scheduler_output.requests)
        ].cpu()
        if self.model.enable_rl:
            scores = result.logits_output.customized_info
            stop_probabilities = scores["stop_probabilities"].detach().clone()
            code_logprobs = scores["code_logprobs"].detach().clone()
        for index, request in enumerate(scheduler_output.requests):
            data = request.data
            if self.model.enable_rl:
                data.stop_probabilities.append(stop_probabilities[index])
            if outputs[request.request_id].data != self.model.config.audio_end_token_id:
                data.output_codes.append(codes[index])
                if self.model.enable_rl:
                    data.code_logprobs.append(code_logprobs[index])
