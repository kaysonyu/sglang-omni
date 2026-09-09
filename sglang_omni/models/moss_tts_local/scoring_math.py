# SPDX-License-Identifier: Apache-2.0
"""Numerical Local text/audio action logprob computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.moss_tts_local.rollout_trace import selected_action_logprobs

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
