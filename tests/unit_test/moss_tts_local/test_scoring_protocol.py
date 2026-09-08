import pytest
from sglang_omni.models.moss_tts_local.scoring_protocol import (
    LocalScoreInput,
    LocalScoreBatch,
)


def sample(**changes):
    data = dict(
        sample_id="a",
        prompt_rows=[[151656] + [1024] * 12],
        decisions=[0, 1],
        codes=[[0] * 12],
    )
    return LocalScoreInput(**(data | changes))


def test_score_identity_ignores_request_label_not_actions():
    assert sample().input_sha256() == sample(sample_id="b").input_sha256()
    assert sample().input_sha256() != sample(codes=[[1] * 12]).input_sha256()


@pytest.mark.parametrize(
    "changes",
    [
        dict(decisions=[1, 0]),
        dict(codes=[[1024] * 12]),
        dict(temperature=0),
        dict(prompt_rows=[[0] * 12]),
    ],
)
def test_invalid_actions_rejected(changes):
    with pytest.raises(ValueError):
        sample(**changes)


def test_stop_without_frames_and_truncation():
    assert len(sample(codes=[], decisions=[1]).codes) == 0
    assert len(sample(decisions=[0]).decisions) == 1


def test_duplicate_sample_ids_rejected():
    with pytest.raises(ValueError):
        LocalScoreBatch(samples=[sample(), sample()])
