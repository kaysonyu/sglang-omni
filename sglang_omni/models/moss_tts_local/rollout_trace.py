# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local structured rollout schema v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

MOSS_TTS_LOCAL_ROLLOUT_VERSION = 2
MOSS_TTS_LOCAL_LOGPROB_SEMANTICS = "temperature_scaled_full_vocab_v1"


def selected_action_logprobs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Selected fp32 logprobs for the neutral temperature-only sampler."""

    if logits.ndim != 2 or actions.shape != logits.shape[:1]:
        raise ValueError("MOSS-TTS Local selected logprob inputs are misaligned.")
    # Positivity is validated once by the request builder. Avoid turning the
    # CUDA tensor into a Python bool here: that would introduce thirteen device
    # synchronizations per generated frame.
    if temperature.shape != logits.shape[:1]:
        raise ValueError(
            "MOSS-TTS Local selected logprob temperatures must be row-aligned."
        )
    return (
        torch.log_softmax(logits.float() / temperature.float().unsqueeze(-1), dim=-1)
        .gather(-1, actions.long().unsqueeze(-1))
        .squeeze(-1)
    )


def moss_tts_local_model_identity(config: Any) -> dict[str, Any]:
    language = (
        getattr(config, "language_config", None)
        or getattr(config, "qwen3_config", None)
        or config
    )
    identity = {
        "policy_family": "moss_tts_local_v1_5",
        "architecture": "MossTTSLocalModel",
        "n_vq": int(config.n_vq),
        "audio_vocab_size": int(config.audio_vocab_size),
        "audio_pad_code": int(config.audio_pad_code),
        "audio_assistant_slot_token_id": int(config.audio_assistant_slot_token_id),
        "audio_end_token_id": int(config.audio_end_token_id),
        "text_vocab_size": int(
            getattr(config, "vocab_size", getattr(language, "vocab_size", 0))
        ),
        "hidden_size": int(getattr(config, "hidden_size", language.hidden_size)),
        "global_layers": int(getattr(language, "num_hidden_layers", 0)),
        "local_layers": int(getattr(config, "local_transformer_layers", 1)),
        "tie_audio_embeddings": True,
        "sample_rate": 48000,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["config_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def build_moss_tts_local_rollout_trace(
    *,
    prompt_rows: torch.Tensor,
    decisions: torch.Tensor,
    decision_logprobs: torch.Tensor,
    codes: torch.Tensor,
    code_logprobs: torch.Tensor,
    finish_reason: str,
    admission_weight_version: str,
    request_id: str,
    sampling: dict[str, Any],
    model_config: Any,
) -> dict[str, Any]:
    if prompt_rows.ndim != 2 or prompt_rows.shape[1] != int(model_config.n_vq) + 1:
        raise ValueError(
            f"MOSS-TTS Local prompt rows must be [P, {int(model_config.n_vq) + 1}]."
        )
    decisions = decisions.to(dtype=torch.long, device="cpu").reshape(-1)
    decision_logprobs = decision_logprobs.to(dtype=torch.float32, device="cpu").reshape(
        -1
    )
    codes = codes.to(dtype=torch.long, device="cpu").reshape(-1, int(model_config.n_vq))
    code_logprobs = code_logprobs.to(dtype=torch.float32, device="cpu").reshape(
        -1, int(model_config.n_vq)
    )
    if decision_logprobs.shape != decisions.shape:
        raise ValueError("MOSS-TTS Local decision actions/logprobs are misaligned.")
    if code_logprobs.shape != codes.shape:
        raise ValueError("MOSS-TTS Local code actions/logprobs are misaligned.")
    if not bool(torch.isfinite(decision_logprobs).all()) or not bool(
        torch.isfinite(code_logprobs).all()
    ):
        raise ValueError(
            "MOSS-TTS Local rollout contains non-finite selected-action logprobs."
        )
    if decisions.numel() and not bool(((decisions == 0) | (decisions == 1)).all()):
        raise ValueError(
            "MOSS-TTS Local decision actions must be 0=continue or 1=stop."
        )
    if codes.numel() and not bool(
        ((codes >= 0) & (codes < int(model_config.audio_vocab_size))).all()
    ):
        raise ValueError(
            "MOSS-TTS Local code actions are outside the audio vocabulary."
        )
    num_frames = int(codes.shape[0])
    if finish_reason == "stop":
        if decisions.numel() != num_frames + 1 or int(decisions[-1]) != 1:
            raise ValueError(
                "MOSS-TTS Local natural stop requires T continue decisions plus one terminal stop."
            )
        if num_frames and not bool((decisions[:-1] == 0).all()):
            raise ValueError(
                "MOSS-TTS Local emitted frames require continue decisions."
            )
    elif finish_reason == "length":
        if decisions.numel() != num_frames or (
            decisions.numel() and not bool((decisions == 0).all())
        ):
            raise ValueError(
                "MOSS-TTS Local length finish requires exactly T continue decisions."
            )
    else:
        raise ValueError(f"Unsupported MOSS-TTS Local finish reason {finish_reason!r}.")

    decision_mask = torch.ones_like(decisions, dtype=torch.bool)
    code_mask = torch.ones_like(codes, dtype=torch.bool)
    return {
        "version": MOSS_TTS_LOCAL_ROLLOUT_VERSION,
        "model_family": "moss_tts_local_v1_5",
        "stages": ["tts_engine"],
        "request_id": str(request_id),
        "model_identity": moss_tts_local_model_identity(model_config),
        "sampling": {
            **sampling,
            "logprob_semantics": MOSS_TTS_LOCAL_LOGPROB_SEMANTICS,
        },
        "logprob_semantics": MOSS_TTS_LOCAL_LOGPROB_SEMANTICS,
        "replay_inputs": {
            "layout": "moss_local_rows",
            "prompt_rows": prompt_rows.to(dtype=torch.long, device="cpu").tolist(),
        },
        "action_streams": [
            {
                "name": "decision",
                "stage": "tts_engine",
                "modality": "audio",
                "action_type": "discrete",
                "layout": "time_1d",
                "vocab_size": 2,
                "actions": decisions.tolist(),
                "logprobs": decision_logprobs.tolist(),
                "action_mask": decision_mask.to(torch.int64).tolist(),
                "deterministic_mask": None,
                "channel_roles": ["continue_or_stop"],
            },
            {
                "name": "codes",
                "stage": "tts_engine",
                "modality": "audio",
                "action_type": "discrete",
                "layout": "time_depth_autoregressive",
                "flatten_order": "time_major",
                "shape": [num_frames, int(model_config.n_vq)],
                "vocab_size": int(model_config.audio_vocab_size),
                "actions": codes.tolist(),
                "logprobs": code_logprobs.tolist(),
                "action_mask": code_mask.to(torch.int64).tolist(),
                "deterministic_mask": None,
                "channel_ids": list(range(int(model_config.n_vq))),
                "channel_roles": [
                    f"rvq_{depth}" for depth in range(int(model_config.n_vq))
                ],
            },
        ],
        "finish_reason": finish_reason,
        "admission_weight_version": str(admission_weight_version),
        "total_action_count": int(decision_mask.sum().item() + code_mask.sum().item()),
        "non_action_outputs": [],
    }


__all__ = [
    "MOSS_TTS_LOCAL_LOGPROB_SEMANTICS",
    "MOSS_TTS_LOCAL_ROLLOUT_VERSION",
    "build_moss_tts_local_rollout_trace",
    "moss_tts_local_model_identity",
    "selected_action_logprobs",
]
