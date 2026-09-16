"""Exact replay inputs and sampled-action ownership at the public RL boundary."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts import rl_contract
from sglang_omni.models.higgs_tts.scoring_protocol import HiggsScoreInput
from sglang_omni.serve.protocol import RolloutGenerateRequest


def test_structured_generation_prompt_preserves_reference_media():
    prompt = {
        "text": "Target.",
        "reference_audio": "/shared/voice.wav",
        "reference_text": "Reference.",
    }
    assert RolloutGenerateRequest(prompt=prompt).prompt == prompt


def test_score_digest_binds_original_reference_codes_and_actions():
    score = HiggsScoreInput(
        sample_id="a",
        prompt_token_ids=[1, -100, 2],
        reference_codes_delayed=[[1, 2]],
        codes=[[2, 4], [3, 1]],
    )
    digest = score.input_sha256()
    assert score.model_copy(update={"sample_id": "b"}).input_sha256() == digest
    assert score.model_copy(update={"codes": [[2, 4], [3, 0]]}).input_sha256() != digest
    assert (
        score.model_copy(update={"reference_codes_delayed": [[0, 2]]}).input_sha256()
        != digest
    )
    with pytest.raises(ValueError, match="prompt slots"):
        HiggsScoreInput(sample_id="a", prompt_token_ids=[1, -100], codes=[[1, 2]])
    with pytest.raises(ValueError, match="aligned codebook"):
        HiggsScoreInput(sample_id="a", prompt_token_ids=[1], codes=[[1, 2], [3]])


def test_sampled_termination_and_tail_survive_audio_mask(monkeypatch):
    monkeypatch.setattr(
        rl_contract, "model_identity", lambda: {"config_sha256": "same"}
    )
    data = SimpleNamespace(
        admission_weight_version="4",
        weight_version="4",
        finish_reason="stop",
        stage_payload=SimpleNamespace(request_id="r"),
        temperature=1.0,
        input_ids=torch.tensor([1, 2]),
        reference_codes_delayed=None,
    )
    mask = [[1, 0], [0, 1], [0, 0]]
    stream = {"shape": [3, 2], "action_mask": mask}
    trace = {"action_streams": [stream]}
    rl_contract.complete_replay_trace(trace, data)
    assert stream["action_mask"] == mask
    assert stream["sampled_action_mask"] == [[1, 0], [1, 1], [1, 1]]
    assert trace["total_sampled_action_count"] == 5


@pytest.mark.parametrize(
    "operation",
    [
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_distributed",
        "init_weights_update_group",
    ],
)
def test_frozen_scorer_rejects_mutation_before_transport(operation):
    from sglang_omni.model_runner.model_worker import ModelWorker

    worker = ModelWorker.__new__(ModelWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            rollout_model_info=lambda: {"supports_weight_update": False}
        )
    )
    success, message = getattr(worker, operation)({})
    assert not success and "frozen scorer" in message
    assert worker.weights_checker("reset_tensors")["success"] is False
