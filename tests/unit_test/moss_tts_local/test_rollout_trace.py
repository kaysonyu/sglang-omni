# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts_local.rollout_trace import (
    build_moss_tts_local_rollout_trace,
    moss_tts_local_model_identity,
    selected_action_logprobs,
)


def _config():
    language = SimpleNamespace(
        vocab_size=151936, hidden_size=2560, num_hidden_layers=36
    )
    return SimpleNamespace(
        n_vq=12,
        audio_vocab_size=1024,
        audio_pad_code=1024,
        audio_assistant_slot_token_id=151656,
        audio_end_token_id=151670,
        vocab_size=151936,
        hidden_size=2560,
        local_transformer_layers=1,
        language_config=language,
    )


def _prompt_rows():
    return torch.tensor([[151644, *([1024] * 12)]], dtype=torch.long)


@pytest.mark.parametrize(
    ("frames", "finish"), [(0, "stop"), (2, "stop"), (2, "length")]
)
def test_rollout_trace_geometry(frames: int, finish: str):
    decisions = torch.zeros(frames + int(finish == "stop"), dtype=torch.long)
    if finish == "stop":
        decisions[-1] = 1
    trace = build_moss_tts_local_rollout_trace(
        prompt_rows=_prompt_rows(),
        decisions=decisions,
        decision_logprobs=torch.full((len(decisions),), -0.2),
        codes=torch.zeros((frames, 12), dtype=torch.long),
        code_logprobs=torch.full((frames, 12), -1.0),
        finish_reason=finish,
        admission_weight_version="7",
        request_id="req-1",
        sampling={"text_temperature": 1.0, "audio_temperature": 1.0},
        model_config=_config(),
    )

    assert trace["version"] == 2
    assert trace["finish_reason"] == finish
    assert trace["total_action_count"] == len(decisions) + frames * 12
    assert trace["action_streams"][0]["actions"] == decisions.tolist()
    assert len(trace["action_streams"][1]["actions"]) == frames
    assert len(trace["model_identity"]["config_sha256"]) == 64


def test_rollout_trace_rejects_discarded_stop_codes():
    with pytest.raises(ValueError, match="T continue decisions"):
        build_moss_tts_local_rollout_trace(
            prompt_rows=_prompt_rows(),
            decisions=torch.tensor([1]),
            decision_logprobs=torch.tensor([-0.2]),
            codes=torch.zeros((1, 12), dtype=torch.long),
            code_logprobs=torch.zeros((1, 12)),
            finish_reason="stop",
            admission_weight_version="7",
            request_id="req-1",
            sampling={},
            model_config=_config(),
        )


def test_model_identity_contains_trainer_handshake_fields():
    identity = moss_tts_local_model_identity(_config())

    assert identity["policy_family"] == "moss_tts_local_v1_5"
    assert identity["n_vq"] == 12
    assert identity["audio_end_token_id"] == 151670
    assert identity["global_layers"] == 36
    assert identity["local_layers"] == 1
    assert identity["tie_audio_embeddings"] is True


def test_selected_action_logprobs_match_temperature_scaled_full_vocab():
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, -1.0, 0.0]])
    actions = torch.tensor([2, 0])
    temperatures = torch.tensor([2.0, 0.5])

    actual = selected_action_logprobs(logits, actions, temperatures)
    expected = (
        torch.log_softmax(logits / temperatures[:, None], dim=-1)
        .gather(-1, actions[:, None])
        .squeeze(-1)
    )

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
