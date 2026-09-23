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
from sglang_omni.models.moss_tts_prism.request_builders import PrismRequestData
from sglang_omni.models.moss_tts_prism.sglang_model import PrismBatchInputs
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
        device = self.model.device
        params = [data.state.generation_kwargs for data in request_states]
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

        def sample(logits: torch.Tensor, channel: int) -> torch.Tensor:
            for row, data in enumerate(request_states):
                penalty = params[row]["audio_repetition_penalty"]
                if penalty != 1.0 and data.output_codes:
                    history = torch.stack(data.output_codes)[:, channel].to(device)
                    scores = logits[row, history]
                    logits[row, history] = torch.where(
                        scores < 0, scores * penalty, scores / penalty
                    )
            return MossTTSModelRunner.sample_tokens(
                logits,
                temperature=temperatures,
                top_p=top_p,
                top_k=top_k,
                seeds=seeds,
                positions=positions + channel,
            )

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
        result.next_token_ids = torch.where(
            result.logits_output.stop_logits.argmax(dim=-1).bool(),
            cfg.audio_end_token_id,
            cfg.audio_assistant_gen_slot_token_id,
        )
        self.model.batch_inputs = None

    post_prefill = post_decode

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        codes = result.logits_output.audio_codes.cpu()
        for index, request in enumerate(scheduler_output.requests):
            data = request.data
            # note (Zhang Yiyang): Prism's stop decision includes this frame;
            # dropping it would truncate both full and streaming output.
            data.output_codes.append(codes[index])
