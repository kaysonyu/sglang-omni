# SPDX-License-Identifier: Apache-2.0
"""SGLang execution for MOSS-TTS Prism."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Attention, Qwen3MLP
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.moss_tts.sampling_kernels import sample_seeded_branchless
from sglang_omni.models.moss_tts_prism.embedding_kernels import (
    feedback_embeddings,
    input_embeddings,
)


class FrameSampler(Protocol):
    def __call__(self, logits: torch.Tensor, channel: int) -> torch.Tensor: ...


@dataclass(kw_only=True)
class PrismBatchInputs:
    rows: torch.Tensor
    item_kind: torch.Tensor
    audio_role: torch.Tensor
    retention: torch.Tensor
    successor_mask: torch.Tensor
    successor_codes: torch.Tensor
    sample_indices: torch.Tensor
    sample: FrameSampler


def prism_cache_layout(config: PretrainedConfig) -> tuple[tuple[int, int], ...]:
    return tuple(
        (unit_index, layer_index)
        for unit_index, unit in enumerate(config.prism_topology.execution_units)
        for layer_index in range(unit.layer_start, unit.layer_stop)
        if layer_index not in config.prism_mlp_only_layers
    )


class PrismDecodeInputs:
    """Stable typed inputs and per-request sampling buffers for decode graphs."""

    def __init__(
        self, config: PretrainedConfig, capacity: int, device: torch.device
    ) -> None:
        self.rows = torch.zeros(
            capacity, config.n_vq + 1, dtype=torch.long, device=device
        )
        self.rows[:, 0] = config.text_pad_idx
        self.item_kind = torch.ones(capacity, dtype=torch.long, device=device)
        self.audio_role = torch.full((capacity,), 2, dtype=torch.long, device=device)
        self.retention = torch.zeros(
            capacity, config.n_vq, dtype=torch.bool, device=device
        )
        self.retention[:, [c - 1 for c in config.prism_input_rvq_channels]] = True
        self.successor_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.successor_codes = torch.zeros(
            capacity, config.n_vq, dtype=torch.long, device=device
        )
        self.sample_indices = torch.arange(capacity, device=device)
        self.temperature = torch.ones(capacity, device=device)
        self.top_p = torch.ones(capacity, device=device)
        self.top_k = torch.ones(capacity, dtype=torch.long, device=device)
        self.seeds = torch.zeros(capacity, dtype=torch.long, device=device)
        self.positions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.penalty = torch.ones(capacity, device=device)
        self.history = torch.zeros(
            capacity,
            config.n_vq,
            config.speech_vocab_size,
            dtype=torch.bool,
            device=device,
        )

    def for_batch(self, batch_size: int) -> PrismBatchInputs:
        def sample(logits: torch.Tensor, channel: int) -> torch.Tensor:
            penalty = self.penalty[:batch_size, None]
            logits = torch.where(
                self.history[:batch_size, channel],
                torch.where(logits < 0, logits * penalty, logits / penalty),
                logits,
            )
            return sample_seeded_branchless(
                logits,
                temperature=self.temperature[:batch_size],
                top_p=self.top_p[:batch_size],
                top_k=self.top_k[:batch_size],
                seeds=self.seeds[:batch_size],
                positions=self.positions[:batch_size] + channel,
            )

        return PrismBatchInputs(
            rows=self.rows[:batch_size],
            item_kind=self.item_kind[:batch_size],
            audio_role=self.audio_role[:batch_size],
            retention=self.retention[:batch_size],
            successor_mask=self.successor_mask[:batch_size],
            successor_codes=self.successor_codes[:batch_size],
            sample_indices=self.sample_indices[:batch_size],
            sample=sample,
        )


class PrismDecoderLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, *, layer_id: int, mlp_only: bool
    ) -> None:
        super().__init__()
        self.mlp_only = mlp_only
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(
            config.hidden_size, config.intermediate_size, config.hidden_act
        )
        if not mlp_only:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = Qwen3Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_id,
                rope_theta=config.rope_parameters["rope_theta"],
                rope_scaling=config.rope_parameters,
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                attention_bias=config.attention_bias,
            )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.mlp_only:
            hidden = hidden + self.self_attn(
                positions, self.input_layernorm(hidden), forward_batch
            )
        return hidden + self.mlp(self.post_attention_layernorm(hidden), forward_batch)


class PrismTransformer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        language = config.language_config
        self.embed_tokens = nn.Embedding(language.vocab_size, language.hidden_size)
        self.layers = nn.ModuleList(
            PrismDecoderLayer(
                language,
                layer_id=index,
                mlp_only=index in config.prism_mlp_only_layers,
            )
            for index in range(config.prism_topology.num_layers)
        )


class MossTTSPrismSGLangModel(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("MOSS-TTS Prism does not support quantized weights yet")
        self.config = config
        self.topology = config.prism_topology
        self.n_vq = config.n_vq
        self.hidden_size = config.language_config.hidden_size
        self.transformer = PrismTransformer(config)
        self.prediction_norms = nn.ModuleList(
            RMSNorm(self.hidden_size, eps=config.language_config.rms_norm_eps)
            for _ in range(self.topology.num_prediction_norms)
        )
        self.audio_embeddings = nn.ModuleList(
            nn.Embedding(config.speech_vocab_size, self.hidden_size)
            for _ in range(self.n_vq)
        )
        self.audio_lm_heads = nn.ModuleList(
            nn.Linear(self.hidden_size, config.speech_vocab_size, bias=False)
            for _ in range(self.n_vq)
        )
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=False)
        self.start_layer = 0
        self.end_layer = len(prism_cache_layout(config))
        self.batch_inputs: PrismBatchInputs | None = None
        self.decode_inputs: PrismDecodeInputs | None = None

    @property
    def device(self) -> torch.device:
        return self.transformer.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.embed_tokens.weight.dtype

    def audio_weight(self, channel: int) -> torch.Tensor:
        if self.config.embedding_head_usage == "tied_head_authoritative":
            return self.audio_lm_heads[channel].weight
        return self.audio_embeddings[channel].weight

    def prepare_inputs(self, inputs: PrismBatchInputs) -> torch.Tensor:
        if inputs.rows.is_cuda:
            return input_embeddings(
                inputs.rows,
                inputs.item_kind,
                inputs.audio_role,
                inputs.retention,
                self.transformer.embed_tokens.weight,
                tuple(self.audio_weight(channel) for channel in range(self.n_vq)),
                self.config.text_pad_idx,
                tuple(self.config.prism_input_rvq_channels),
            )
        text = inputs.item_kind.eq(0)
        active = text & inputs.rows[:, 0].ne(self.config.text_pad_idx)
        hidden = self.transformer.embed_tokens(
            inputs.rows[:, 0].masked_fill(~active, 0)
        ) * active.unsqueeze(-1)
        target = inputs.audio_role.eq(2)
        for channel in range(self.n_vq):
            active = ~text & (
                ~target
                | (
                    (channel + 1 in self.config.prism_input_rvq_channels)
                    & inputs.retention[:, channel]
                )
            )
            hidden = hidden + F.embedding(
                inputs.rows[:, channel + 1].masked_fill(~active, 0),
                self.audio_weight(channel),
            ) * active.unsqueeze(-1)
        return hidden

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        inputs = self.batch_inputs
        if self.decode_inputs is not None and forward_batch.forward_mode.is_decode():
            inputs = self.decode_inputs.for_batch(input_ids.numel())
        assert inputs is not None
        hidden = self.prepare_inputs(inputs)
        codes = inputs.successor_codes.clone()
        mask = inputs.successor_mask.clone()
        mask.index_fill_(0, inputs.sample_indices, True)
        frames = torch.empty(
            (inputs.sample_indices.numel(), self.n_vq),
            dtype=torch.long,
            device=hidden.device,
        )
        stop_logits = None
        cache_slot = 0

        incoming = None
        index = 0
        units = self.topology.execution_units
        while index < len(units):
            unit = units[index]
            is_loop = unit.kind == "loop"
            stop = index + 1
            if is_loop:
                while (
                    stop < len(units)
                    and units[stop].kind == "loop"
                    and units[stop].index == unit.index
                ):
                    stop += 1
            block_input = hidden
            accumulated = incoming
            for application in range(index, stop):
                if is_loop:
                    # note (Zhang Yiyang): Repeated blocks reuse their original
                    # input residual and accumulate RVQ feedback separately.
                    hidden = (
                        block_input if application == index else hidden + block_input
                    )
                    if accumulated is not None:
                        hidden = hidden + accumulated
                elif incoming is not None:
                    hidden = hidden + incoming

                for layer_index in range(unit.layer_start, unit.layer_stop):
                    layer = self.transformer.layers[layer_index]
                    if not layer.mlp_only:
                        # note (Zhang Yiyang): Each invocation of a reused
                        # attention layer needs its own KV history.
                        layer.self_attn.attn.layer_id = cache_slot
                        cache_slot += 1
                    hidden = layer(hidden, positions, forward_batch)

                incoming = None
                heads = units[application].rvq_heads
                if heads:
                    normalized = self.prediction_norms[
                        self.topology.prediction_norm_index_by_execution_unit[
                            application
                        ]
                    ](hidden[inputs.sample_indices])
                    if 1 in heads:
                        with torch.autocast(
                            device_type=normalized.device.type, enabled=False
                        ):
                            stop_logits = F.linear(
                                normalized.float(), self.stop_head.weight.float()
                            )
                    logits = [
                        self.audio_lm_heads[head - 1](normalized) for head in heads
                    ]
                    for head, scores in zip(heads, logits):
                        selected = inputs.sample(scores.float(), head - 1)
                        frames[:, head - 1] = selected
                        codes[inputs.sample_indices, head - 1] = selected
                    if application != len(units) - 1:
                        if codes.is_cuda:
                            incoming = feedback_embeddings(
                                codes,
                                mask,
                                tuple(self.audio_weight(head - 1) for head in heads),
                                tuple(head - 1 for head in heads),
                            )
                        else:
                            for head in heads:
                                embedding = F.embedding(
                                    codes[:, head - 1].masked_fill(~mask, 0),
                                    self.audio_weight(head - 1),
                                ) * mask.unsqueeze(-1)
                                incoming = (
                                    embedding
                                    if incoming is None
                                    else incoming + embedding
                                )
                if is_loop:
                    if self.topology.reapply_rvq_conditioning_in_loop:
                        if incoming is not None:
                            accumulated = (
                                incoming
                                if accumulated is None
                                else accumulated + incoming
                            )
                    else:
                        accumulated = incoming
            index = stop
        assert stop_logits is not None
        return LogitsProcessorOutput(
            next_token_logits=stop_logits,
            customized_info={"audio_codes": frames},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        for name, weight in weights:
            for target, source, shard in stacked:
                if f".{source}." in name:
                    param = params[name.replace(f".{source}.", f".{target}.")]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)


EntryClass = MossTTSPrismSGLangModel
