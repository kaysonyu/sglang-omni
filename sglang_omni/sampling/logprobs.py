# SPDX-License-Identifier: Apache-2.0
"""Selected full-vocabulary action logprobs."""

import torch
import triton
import triton.language as tl


@triton.jit
def selected_action_logprobs_kernel(
    logits,
    actions,
    temperature,
    output,
    vocab_size: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    action_stride: tl.constexpr,
    temperature_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits + row * row_stride + cols * col_stride,
        mask=cols < vocab_size,
        other=-float("inf"),
    ).to(tl.float32)
    scale = tl.load(temperature + row * temperature_stride).to(tl.float32)
    scores = tl.div_rn(values, scale)
    shifted = scores - tl.max(scores, axis=0)
    normalizer = tl.log(tl.sum(tl.exp(shifted), axis=0))
    action = tl.load(actions + row * action_stride)
    selected = tl.sum(tl.where(cols == action, shifted, 0.0), axis=0)
    tl.store(output + row, selected - normalizer)


def selected_action_logprobs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Selected fp32 logprobs for the neutral temperature-only sampler."""

    if logits.ndim != 2 or actions.shape != logits.shape[:1]:
        raise ValueError("Selected logprob inputs are misaligned.")
    if temperature.shape != logits.shape[:1]:
        raise ValueError("Selected logprob temperatures must be row-aligned.")
    if logits.is_cuda:
        rows, vocab_size = logits.shape
        output = torch.empty(rows, device=logits.device, dtype=torch.float32)
        selected_action_logprobs_kernel[(rows,)](
            logits,
            actions,
            temperature,
            output,
            vocab_size,
            logits.stride(0),
            logits.stride(1),
            actions.stride(0),
            temperature.stride(0),
            BLOCK_SIZE=triton.next_power_of_2(vocab_size),
        )
        return output
    else:
        return (
            torch.log_softmax(
                logits.float() / temperature.float().unsqueeze(-1), dim=-1
            )
            .gather(-1, actions.long().unsqueeze(-1))
            .squeeze(-1)
        )
