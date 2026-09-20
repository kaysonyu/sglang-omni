from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts import rl_contract
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState


def test_version_and_termination_survive_pipeline_state_serialization():
    state = HiggsTtsState(weight_version='run:3', finish_reason='length')
    result = HiggsTtsState.from_dict(state.to_dict())
    assert result.weight_version == 'run:3'
    assert result.finish_reason == 'length'


def test_exact_replay_trace_keeps_prompt_reference_and_admission(monkeypatch):
    monkeypatch.setattr(rl_contract, 'model_identity', lambda: {'config_sha256': 'a' * 64})
    data = SimpleNamespace(admission_weight_version='run:3', weight_version='run:3', finish_reason='length',
        stage_payload=SimpleNamespace(request_id='request'), temperature=.7,
        input_ids=torch.tensor([1,-100,2]), reference_codes_delayed=[[4,5]])
    trace = {'action_streams': []}
    rl_contract.complete_replay_trace(trace, data)
    assert trace['version'] == 2
    assert trace['replay_inputs']['prompt_token_ids'] == [1,-100,2]
    assert trace['replay_inputs']['reference_codes_delayed'] == [[4,5]]
    assert trace['admission_weight_version'] == 'run:3'
    data.weight_version = 'run:4'
    with pytest.raises(ValueError, match='mixed admission'):
        rl_contract.complete_replay_trace({}, data)
