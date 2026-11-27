# SPDX-License-Identifier: Apache-2.0
"""Packed prefill, frame sampling and streaming handoff for Prism."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import PretrainedConfig

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.sampling_kernels import sample_seeded_fused
from sglang_omni.models.moss_tts_prism.request_builders import PrismRequestData
from sglang_omni.models.moss_tts_prism.rollout_trace import PRISM_STOP_THRESHOLD
from sglang_omni.models.moss_tts_prism.sglang_model import PrismBatchInputs
from sglang_omni.sampling.logprobs import selected_action_logprobs
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

STOP_SAMPLING_SEED_XOR = 0x45D9F3B


def sample_prism_tokens(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    logprob_output: torch.Tensor | None,
) -> torch.Tensor:
    """Sample RVQ or stop actions with optional full-vocabulary scores."""
    if logits.is_cuda:
        return sample_seeded_fused(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=seeds,
            positions=positions,
            full_vocab_logprobs=logprob_output,
        )
    else:
        actions = MossTTSModelRunner.sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=seeds,
            positions=positions,
        )
        if logprob_output is not None:
            logprob_output.copy_(selected_action_logprobs(logits, actions, temperature))
        return actions


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


class PinnedStaging:
    """Two pinned host copies of each forward output, alternating by forward."""

    def __init__(self) -> None:
        # note (Linsey): A lagged resolve reads one slot while the next launch copies into the other.
        self.slots: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] = ({}, {})
        self.slot = 0

    def copy(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Queue a copy of a device tensor; it is complete once the forward's token ids are."""
        if not value.is_cuda:
            return value.clone()
        buffers = self.slots[self.slot]
        buffer = buffers.get(name)
        if (
            buffer is None
            or buffer.numel() < value.numel()
            or buffer.dtype != value.dtype
        ):
            buffer = torch.empty(value.numel(), dtype=value.dtype, pin_memory=True)
            buffers[name] = buffer
        host = buffer[: value.numel()].view(value.shape)
        host.copy_(value, non_blocking=True)
        return host


@dataclass(kw_only=True)
class PrismStepRecord:
    """Host copies of what one forward adds to each request of its batch."""

    codes: torch.Tensor
    code_logprobs: torch.Tensor | None
    stop_probabilities: torch.Tensor | None
    sampled_rows: list[int]
    stop_actions: torch.Tensor | None
    stop_logprobs: torch.Tensor | None
    routes: torch.Tensor | None
    route_rows: list[int]
    is_prefill: bool
    overrun_rows: set[int] = field(default_factory=set)


class MossTTSPrismModelRunner(ModelRunner):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.staging = PinnedStaging()

    def prepare(
        self,
        requests: list[SchedulerRequest],
        *,
        prefill: bool,
        forward_batch: ForwardBatch,
    ) -> None:
        request_states = [request.data for request in requests]
        device = self.model.device
        params = [data.state.generation_kwargs for data in request_states]
        decode = self.model.decode_inputs
        batch_size = len(request_states)
        if not prefill and decode is not None:
            decode.rows[:batch_size, 1:] = decode.last_codes[
                forward_batch.req_pool_indices
            ]
            floats_cpu = decode.floats_cpu[decode.host_slot]
            ints_cpu = decode.ints_cpu[decode.host_slot]
            decode.host_slot ^= 1
            penalties = [p["audio_repetition_penalty"] for p in params]
            floats_cpu[:, :batch_size].copy_(
                torch.tensor(
                    [
                        [p["audio_temperature"] for p in params],
                        [p["audio_top_p"] for p in params],
                        penalties,
                    ],
                    dtype=torch.float32,
                )
            )
            # note (Linsey): Positions count launched frames: a lookahead launch precedes the
            # host results of the frame before it, and on the synchronous path both counts agree.
            ints_cpu[:, :batch_size].copy_(
                torch.tensor(
                    [
                        [p["audio_top_k"] for p in params],
                        [data.sampling_seed for data in request_states],
                        [
                            data.frames_launched * self.model.n_vq
                            for data in request_states
                        ],
                    ],
                    dtype=torch.long,
                )
            )
            decode.sampling_floats.copy_(floats_cpu, non_blocking=True)
            decode.sampling_ints.copy_(ints_cpu, non_blocking=True)
            if any(p != 1.0 for p in penalties):
                history = torch.zeros(
                    (batch_size, self.model.n_vq, self.model.config.speech_vocab_size),
                    dtype=torch.bool,
                )
                for row, data in enumerate(request_states):
                    if data.output_codes:
                        history[row].scatter_(1, torch.stack(data.output_codes).T, True)
                decode.history[:batch_size].copy_(history)
            for data in request_states:
                data.frames_launched += 1
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
                data.frames_launched = len(data.output_codes) + 1
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
            return sample_prism_tokens(
                logits,
                temperature=temperatures,
                top_p=top_p,
                top_k=top_k,
                seeds=seeds,
                positions=positions + channel,
                logprob_output=(
                    code_logprobs[:, channel] if code_logprobs is not None else None
                ),
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
            code_logprobs=code_logprobs,
        )

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.prepare(requests, prefill=True, forward_batch=forward_batch)

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        self.prepare(requests, prefill=False, forward_batch=forward_batch)

    def collect(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> PrismStepRecord:
        """Decide each stop and queue host copies of the step's outputs without waiting on them."""
        cfg = self.model.config
        decode = self.model.decode_inputs
        batch_size = len(requests)
        is_prefill = forward_batch.forward_mode.is_extend()
        stop_logits = result.logits_output.next_token_logits.float()
        stop_probabilities = stop_logits.softmax(dim=-1)[:, 1]
        stop_actions = (stop_probabilities > PRISM_STOP_THRESHOLD).long()
        sampled_rows = [
            index
            for index, request in enumerate(requests)
            if request.data.stop_sampling
        ]
        sampled_actions = None
        sampled_logprobs = None
        if sampled_rows:
            device = stop_logits.device
            count = len(sampled_rows)
            if decode is not None and not is_prefill:
                seeds = decode.seeds[:batch_size] ^ STOP_SAMPLING_SEED_XOR
                positions = decode.positions[:batch_size] // self.model.n_vq
            else:
                seeds = torch.tensor(
                    [
                        request.data.sampling_seed ^ STOP_SAMPLING_SEED_XOR
                        for request in requests
                    ],
                    dtype=torch.long,
                    device=device,
                )
                positions = torch.tensor(
                    [len(request.data.output_codes) for request in requests],
                    dtype=torch.long,
                    device=device,
                )
            if count == batch_size:
                selected_logits = stop_logits
            else:
                indices = torch.tensor(sampled_rows, device=device, dtype=torch.long)
                selected_logits = stop_logits.index_select(0, indices)
                seeds = seeds.index_select(0, indices)
                positions = positions.index_select(0, indices)
            ones = torch.ones(count, dtype=torch.float32, device=device)
            sampled_logprobs = (
                torch.empty(count, dtype=torch.float32, device=device)
                if self.model.enable_rl
                else None
            )
            sampled_actions = sample_prism_tokens(
                selected_logits,
                temperature=ones,
                top_p=ones,
                top_k=torch.full((count,), -1, dtype=torch.long, device=device),
                seeds=seeds,
                positions=positions,
                logprob_output=sampled_logprobs,
            )
            if count == batch_size:
                stop_actions = sampled_actions
            else:
                stop_actions.index_copy_(0, indices, sampled_actions)
        info = result.logits_output.customized_info
        codes = info["audio_codes"][:batch_size]
        if decode is not None:
            decode.last_codes[forward_batch.req_pool_indices] = codes
        route_rows = (
            list(forward_batch.extend_seq_lens_cpu) if is_prefill else [1] * batch_size
        )
        enable_rl = self.model.enable_rl
        record = PrismStepRecord(
            codes=self.staging.copy("codes", codes),
            code_logprobs=(
                self.staging.copy("code_logprobs", info["code_logprobs"][:batch_size])
                if enable_rl
                else None
            ),
            stop_probabilities=(
                self.staging.copy("stop_probabilities", stop_probabilities)
                if enable_rl
                else None
            ),
            sampled_rows=sampled_rows,
            stop_actions=(
                self.staging.copy("stop_actions", sampled_actions)
                if enable_rl and sampled_rows
                else None
            ),
            stop_logprobs=(
                self.staging.copy("stop_logprobs", sampled_logprobs)
                if enable_rl and sampled_rows
                else None
            ),
            # note (Linsey): The next forward overwrites the capture buffer, so the copy is queued before it.
            routes=(
                self.staging.copy("routes", self.model.route_ids[: sum(route_rows)])
                if self.model.route_ids is not None
                else None
            ),
            route_rows=route_rows,
            is_prefill=is_prefill,
        )
        self.staging.slot ^= 1
        result.next_token_ids = torch.where(
            stop_actions.bool(),
            cfg.audio_end_token_id,
            cfg.audio_assistant_gen_slot_token_id,
        )
        # note (Linsey): Staged last, so the token-id event that finalize waits on covers every copy above.
        self.stage_token_ids(result, result.next_token_ids)
        result.prism_step = record
        self.model.batch_inputs = None
        return record

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect(result, forward_batch, requests)

    post_prefill = post_decode

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> PrismStepRecord:
        return self.collect(result, forward_batch, requests)

    def post_decode_resolve(
        self,
        launch_buf: PrismStepRecord,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        # note (Linsey): A lagged step still holds requests that finished or were retracted
        # in the step before it; their extra frame never joins the trajectory.
        launch_buf.overrun_rows = {
            index
            for index, request in enumerate(requests)
            if request.data.req.finished() or self.req_is_retracted(request.data.req)
        }

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Lookahead reads last frames on the device; penalty history is built on the host."""
        return self.model.decode_inputs is not None and all(
            request._omni_data.state.generation_kwargs["audio_repetition_penalty"]
            == 1.0
            for request in batch.reqs
        )

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        record = result.prism_step
        enable_rl = self.model.enable_rl
        codes = record.codes.clone()
        code_logprobs = record.code_logprobs.clone() if enable_rl else None
        stop_probabilities = record.stop_probabilities.clone() if enable_rl else None
        stop_actions = (
            record.stop_actions.clone() if record.stop_actions is not None else None
        )
        stop_logprobs = (
            record.stop_logprobs.clone() if record.stop_logprobs is not None else None
        )
        sampled_positions = {
            row: position for position, row in enumerate(record.sampled_rows)
        }
        routes = (
            record.routes.clone().split(record.route_rows)
            if record.routes is not None
            else None
        )
        for index, request in enumerate(scheduler_output.requests):
            if index in record.overrun_rows:
                continue
            data = request.data
            if routes is not None:
                # note (Linsey): A prefill recomputes every row it holds, including re-prefilled frames.
                if record.is_prefill:
                    data.routed_experts = [routes[index]]
                else:
                    data.routed_experts.append(routes[index])
            if enable_rl:
                data.stop_probabilities.append(stop_probabilities[index])
                if stop_actions is not None and index in sampled_positions:
                    data.stop_actions.append(stop_actions[sampled_positions[index]])
                    data.stop_logprobs.append(stop_logprobs[sampled_positions[index]])
            if outputs[request.request_id].data != self.model.config.audio_end_token_id:
                data.output_codes.append(codes[index])
                if enable_rl:
                    data.code_logprobs.append(code_logprobs[index])
