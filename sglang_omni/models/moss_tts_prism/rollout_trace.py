# SPDX-License-Identifier: Apache-2.0
"""Prism RVQ actions and deterministic stop observations for replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

PRISM_STOP_THRESHOLD = 0.1


def prism_model_identity(checkpoint_dir: str) -> dict[str, str | int | bool]:
    artifact = Path(checkpoint_dir)
    config = json.loads((artifact / "config.json").read_bytes())
    return {
        "policy_family": "moss_tts_prism",
        "architecture": "MOSS-TTS-Prism",
        **{
            name: hashlib.sha256((artifact / filename).read_bytes()).hexdigest()
            for name, filename in (
                ("config_sha256", "config.json"),
                ("modeling_sha256", "modeling_moss_tts.py"),
                ("processor_sha256", "processing_moss_tts.py"),
            )
        },
        "n_vq": config["n_vq"],
        "audio_vocab_size": config["speech_vocab_size"],
        "sample_rate": config["sampling_rate"],
        "stop_semantics": "pre_frame_threshold_v1",
    }


def build_prism_rollout_trace(
    *,
    prompt: dict[str, torch.Tensor],
    codes: torch.Tensor,
    code_logprobs: torch.Tensor,
    stop_probabilities: torch.Tensor,
    finish_reason: str,
    request_id: str,
    admission_weight_version: str,
    model_identity: dict[str, str | int | bool],
    sampling: dict[str, float | int],
) -> bytes:
    codes = codes.long().cpu()
    code_logprobs = code_logprobs.float().cpu()
    stop_probabilities = stop_probabilities.float().cpu()
    frames, channels = codes.shape
    if code_logprobs.shape != codes.shape:
        raise RuntimeError("Prism codes and selected logprobs are misaligned")
    if finish_reason not in ("stop", "length"):
        raise RuntimeError(f"Unsupported Prism finish reason {finish_reason}")
    if stop_probabilities.shape != (frames + int(finish_reason == "stop"),):
        raise RuntimeError("Prism stop observations are not frame-aligned")
    if (
        not torch.isfinite(code_logprobs).all()
        or not torch.isfinite(stop_probabilities).all()
    ):
        raise RuntimeError("Prism rollout contains non-finite scores")
    stopped = stop_probabilities > PRISM_STOP_THRESHOLD
    if stopped[:frames].any() or (finish_reason == "stop" and not stopped[-1]):
        raise RuntimeError("Prism stop observations disagree with emitted frames")
    trace = {
        "version": 2,
        "model_family": "moss_tts_prism",
        "stages": ["tts_engine"],
        "request_id": request_id,
        "model_identity": model_identity,
        "admission_weight_version": admission_weight_version,
        "logprob_semantics": "temperature_scaled_full_vocab_v1",
        "stop_semantics": "pre_frame_threshold_v1",
        "sampling": {**sampling, "stop_threshold": PRISM_STOP_THRESHOLD},
        "replay_inputs": {
            "layout": "moss_prism_typed_v1",
            "prompt_rows": prompt["input_ids"].tolist(),
            **{
                name: prompt[name].tolist()
                for name in (
                    "item_kind",
                    "audio_role",
                    "successor_audio_mask",
                    "successor_audio_codes",
                    "target_history_retention_mask",
                )
            },
        },
        "action_streams": [
            {
                "name": "codes",
                "stage": "tts_engine",
                "modality": "audio",
                "action_type": "discrete",
                "layout": "time_depth_autoregressive",
                "flatten_order": "time_major",
                "shape": [frames, channels],
                "vocab_size": model_identity["audio_vocab_size"],
                "actions": codes.tolist(),
                "logprobs": code_logprobs.tolist(),
                "action_mask": torch.ones_like(codes).tolist(),
                "deterministic_mask": None,
                "channel_ids": list(range(channels)),
                "channel_roles": [f"rvq_{channel}" for channel in range(channels)],
            }
        ],
        "non_action_outputs": [
            {
                "name": "stop_probabilities",
                "stage": "tts_engine",
                "values": stop_probabilities.tolist(),
            }
        ],
        "finish_reason": finish_reason,
        "total_action_count": frames * channels,
    }
    # note (Zhang Yiyang): Avoid routing every scalar of a long rollout separately.
    return json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
