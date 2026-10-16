# SPDX-License-Identifier: Apache-2.0
"""Prism RVQ and optional sampled-stop actions for replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

PRISM_STOP_THRESHOLD = 0.1
SAMPLED_STOP_SEMANTICS = "pre_frame_bernoulli_v1"


def prism_model_identity(checkpoint_dir: str) -> dict[str, str | int | bool]:
    artifact = Path(checkpoint_dir)
    config = json.loads((artifact / "config.json").read_bytes())
    revision = config.get("prompt_renderer_revision")
    if type(revision) is not int or revision < 1:
        raise ValueError("Prism checkpoint must record a positive renderer revision")
    return {
        "policy_family": "moss_tts_prism",
        "architecture": "MOSS-TTS-Prism",
        **{
            name: hashlib.sha256((artifact / filename).read_bytes()).hexdigest()
            for name, filename in (
                ("config_sha256", "config.json"),
                ("modeling_sha256", "modeling_moss_tts.py"),
                ("processor_sha256", "processing_moss_tts.py"),
                ("prompt_protocol_sha256", "prompt_protocol.py"),
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
    sampling: dict[str, float | int | bool],
    stop_actions: torch.Tensor | None = None,
    stop_logprobs: torch.Tensor | None = None,
) -> bytes:
    codes = codes.long().cpu()
    code_logprobs = code_logprobs.float().cpu()
    stop_probabilities = stop_probabilities.float().cpu()
    frames, channels = codes.shape
    if code_logprobs.shape != codes.shape:
        raise RuntimeError("Prism codes and selected logprobs are misaligned")
    if finish_reason not in ("stop", "length"):
        raise RuntimeError(f"Unsupported Prism finish reason {finish_reason}")
    if frames == 0 and finish_reason != "stop":
        raise RuntimeError("A zero-frame Prism trajectory must end with stop")
    if stop_probabilities.shape != (frames + int(finish_reason == "stop"),):
        raise RuntimeError("Prism stop observations are not frame-aligned")
    if (
        not torch.isfinite(code_logprobs).all()
        or not torch.isfinite(stop_probabilities).all()
    ):
        raise RuntimeError("Prism rollout contains non-finite scores")
    if (stop_actions is None) != (stop_logprobs is None):
        raise RuntimeError(
            "Prism sampled stop actions and scores must be supplied together"
        )
    sampled_stop = stop_actions is not None
    if sampled_stop:
        stop_actions = stop_actions.long().cpu()
        stop_logprobs = stop_logprobs.float().cpu()
        stop_count = frames + int(finish_reason == "stop")
        if stop_actions.shape != (stop_count,) or stop_logprobs.shape != (stop_count,):
            raise RuntimeError(
                "Prism sampled stop actions and scores are not frame-aligned"
            )
        if (
            torch.any(stop_actions[:frames] != 0)
            or (finish_reason == "stop" and stop_actions[-1] != 1)
            or not torch.isfinite(stop_logprobs).all()
            or torch.any(stop_logprobs > 1e-5)
        ):
            raise RuntimeError("Prism sampled stop decisions or logprobs are invalid")
        selected_probabilities = torch.where(
            stop_actions.bool(), stop_probabilities, 1 - stop_probabilities
        )
        if not torch.allclose(
            stop_logprobs.exp(), selected_probabilities, atol=2e-5, rtol=2e-4
        ):
            raise RuntimeError(
                "Prism sampled stop scores disagree with stop probabilities"
            )
    else:
        stopped = stop_probabilities > PRISM_STOP_THRESHOLD
        if stopped[:frames].any() or (finish_reason == "stop" and not stopped[-1]):
            raise RuntimeError("Prism stop observations disagree with emitted frames")
    action_streams = [
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
    ]
    if sampled_stop:
        action_streams.append(
            {
                "name": "stop_decisions",
                "stage": "tts_engine",
                "modality": "control",
                "action_type": "discrete",
                "layout": "pre_frame_autoregressive",
                "flatten_order": "time_major",
                "shape": [stop_count],
                "vocab_size": 2,
                "actions": stop_actions.tolist(),
                "logprobs": stop_logprobs.tolist(),
                "action_mask": torch.ones_like(stop_actions).tolist(),
                "deterministic_mask": None,
            }
        )
    trace = {
        "version": 2,
        "model_family": "moss_tts_prism",
        "stages": ["tts_engine"],
        "request_id": request_id,
        "model_identity": model_identity,
        "admission_weight_version": admission_weight_version,
        "logprob_semantics": "temperature_scaled_full_vocab_v1",
        "stop_semantics": (
            SAMPLED_STOP_SEMANTICS if sampled_stop else "pre_frame_threshold_v1"
        ),
        "sampling": (
            {**sampling, "stop_sampling": True}
            if sampled_stop
            else {**sampling, "stop_threshold": PRISM_STOP_THRESHOLD}
        ),
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
        "action_streams": action_streams,
        "non_action_outputs": [
            {
                "name": "stop_probabilities",
                "stage": "tts_engine",
                "values": stop_probabilities.tolist(),
            }
        ],
        "finish_reason": finish_reason,
        "total_action_count": frames * channels + (stop_count if sampled_stop else 0),
    }
    # note (Zhang Yiyang): Avoid routing every scalar of a long rollout separately.
    return json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
