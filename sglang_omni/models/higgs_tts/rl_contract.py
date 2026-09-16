# SPDX-License-Identifier: Apache-2.0
"""Versioned, exact-input Higgs RL trace without changing sampling or audio masks."""

import hashlib
import json
from pathlib import Path


def model_identity():
    from sglang.srt.runtime_context import get_model

    config = json.loads((Path(get_model().model_path) / "config.json").read_text())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"architecture": "HiggsMultimodalQwen3ForConditionalGeneration", "config_sha256": digest}


def complete_replay_trace(trace, data):
    admission = data.admission_weight_version
    if admission is None or data.weight_version is None or admission != str(data.weight_version):
        raise ValueError("Higgs RL request has missing or mixed admission/response weight versions")
    if data.finish_reason not in {"stop", "length"}:
        raise ValueError(f"Higgs RL cannot replay finish reason {data.finish_reason!r}")
    # Audio-validity masks intentionally omit EOC and delayed tail positions.
    # RL additionally needs every sampled choice that can affect later state.
    # Preserve the existing audio mask and publish a separate sampling mask.
    for stream in trace["action_streams"]:
        rows, channels = stream["shape"]
        stream["sampled_action_mask"] = [[int(channel <= row) for channel in range(channels)] for row in range(rows)]
        stream["sampled_action_mask_semantics"] = "higgs_delay_initialization_v1"
    trace["total_sampled_action_count"] = sum(sum(row) for stream in trace["action_streams"]
                                             for row in stream["sampled_action_mask"])
    trace.update(
        version=2,
        request_id=data.stage_payload.request_id,
        admission_weight_version=admission,
        finish_reason=data.finish_reason,
        model_identity=model_identity(),
        logprob_semantics="temperature_scaled_full_vocab_v1",
        sampling={"temperature": float(data.temperature), "top_p": 1.0, "top_k": -1},
        replay_inputs={
            "layout": "higgs_delayed_codes",
            "prompt_token_ids": data.input_ids.tolist(),
            "reference_codes_delayed": data.reference_codes_delayed,
        },
        action_mask_semantics="real_audio_codes_v1",
    )
