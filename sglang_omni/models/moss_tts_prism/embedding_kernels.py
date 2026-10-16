# SPDX-License-Identifier: Apache-2.0
"""Typed input and RVQ feedback embeddings for Prism."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EMBEDDING_BLOCK_SIZE = 256


@triton.jit
def input_embedding_kernel(
    rows,
    kinds,
    roles,
    retention,
    text_weight,
    audio_weights,
    output,
    ROW_STRIDE: tl.constexpr,
    RETENTION_STRIDE: tl.constexpr,
    HIDDEN: tl.constexpr,
    PAD: tl.constexpr,
    INPUT_CHANNEL_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    text = tl.load(kinds + row) == 0
    target = tl.load(roles + row) == 2
    token = tl.load(rows + row * ROW_STRIDE)
    active = text & (token != PAD)
    value = tl.load(
        text_weight + tl.where(active, token, 0) * HIDDEN + col, col < HIDDEN, other=0
    ).to(tl.float32)
    value = value * active.to(tl.float32)
    for channel in tl.static_range(len(audio_weights)):
        retained = tl.load(retention + row * RETENTION_STRIDE + channel)
        active = (~text) & ((~target) | (INPUT_CHANNEL_MASK[channel] & retained))
        code = tl.load(rows + row * ROW_STRIDE + channel + 1)
        embedding = tl.load(
            audio_weights[channel] + tl.where(active, code, 0) * HIDDEN + col,
            col < HIDDEN,
            other=0,
        ).to(tl.float32)
        # note (Zhang Yiyang): Preserve eager rounding after each channel addition.
        value = (
            (value + embedding * active.to(tl.float32))
            .to(output.dtype.element_ty)
            .to(tl.float32)
        )
    tl.store(output + row * HIDDEN + col, value, col < HIDDEN)


@triton.jit
def feedback_embedding_kernel(
    codes,
    mask,
    weights,
    output,
    CHANNELS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(mask + row)
    for index in tl.static_range(len(CHANNELS)):
        code = tl.load(codes + row * ROW_STRIDE + CHANNELS[index])
        embedding = tl.load(
            weights[index] + tl.where(active, code, 0) * HIDDEN + col,
            col < HIDDEN,
            other=0,
        ).to(tl.float32) * active.to(tl.float32)
        if index == 0:
            value = embedding
        else:
            value = (value + embedding).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + row * HIDDEN + col, value, col < HIDDEN)


def input_embeddings(
    rows: torch.Tensor,
    item_kind: torch.Tensor,
    audio_role: torch.Tensor,
    retention: torch.Tensor,
    text_weight: torch.Tensor,
    audio_weights: tuple[torch.Tensor, ...],
    text_pad_idx: int,
    input_channels: tuple[int, ...],
) -> torch.Tensor:
    hidden = text_weight.shape[1]
    output = text_weight.new_empty((rows.shape[0], hidden))
    input_embedding_kernel[
        (output.shape[0], triton.cdiv(hidden, EMBEDDING_BLOCK_SIZE))
    ](
        rows,
        item_kind,
        audio_role,
        retention,
        text_weight,
        audio_weights,
        output,
        rows.stride(0),
        retention.stride(0),
        hidden,
        text_pad_idx,
        tuple(channel + 1 in input_channels for channel in range(len(audio_weights))),
        EMBEDDING_BLOCK_SIZE,
        enable_fp_fusion=False,
    )
    return output


def feedback_embeddings(
    codes: torch.Tensor,
    mask: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    channels: tuple[int, ...],
) -> torch.Tensor:
    hidden = weights[0].shape[1]
    output = weights[0].new_empty((codes.shape[0], hidden))
    feedback_embedding_kernel[
        (output.shape[0], triton.cdiv(hidden, EMBEDDING_BLOCK_SIZE))
    ](
        codes,
        mask,
        weights,
        output,
        channels,
        codes.stride(0),
        hidden,
        EMBEDDING_BLOCK_SIZE,
        enable_fp_fusion=False,
    )
    return output
