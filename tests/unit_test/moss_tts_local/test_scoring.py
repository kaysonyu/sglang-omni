# SPDX-License-Identifier: Apache-2.0
"""Teacher protocol, action alignment and numerical replay."""

from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.models.moss_tts_local.config import MossTTSLocalScorePipelineConfig
from sglang_omni.models.moss_tts_local.local_transformer import MossTTSLocalTransformer
from sglang_omni.models.moss_tts_local.scoring import (
    build_score_request,
    score_local_depth,
)
from sglang_omni.models.moss_tts_local.scoring_protocol import (
    LocalScoreBatch,
    LocalScoreInput,
)
from sglang_omni.models.moss_tts_local.sglang_model import MossTTSLocalSGLangModel
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.local_scoring import register_local_scoring


def score_input(**kwargs):
    return LocalScoreInput(
        sample_id="sample",
        prompt_rows=[[1, 17, 17, 17]],
        decisions=[0, 1],
        codes=[[2, 3, 4]],
        **kwargs,
    )


def model_config():
    return SimpleNamespace(
        n_vq=3,
        audio_vocab_size=17,
        audio_pad_code=17,
        vocab_size_list=[32, 18, 18, 18],
        audio_assistant_slot_token_id=7,
    )


def test_score_config_has_only_teacher():
    config = MossTTSLocalScorePipelineConfig(model_path="test")
    assert config.resolved_entry_stage == "tts_engine"
    assert len(config.stages) == 1 and config.stages[0].terminal
    assert config.stages[0].engine.disable_radix_cache
    assert config.stages[0].engine.disable_cuda_graph


def test_score_input_uses_model_geometry_and_hash():
    sample = score_input()
    sample.validate_model(model_config(), 8)
    renamed = sample.model_copy(update={"sample_id": "other"})
    assert renamed.input_sha256() == sample.input_sha256()
    changed = sample.model_copy(update={"codes": [[3, 3, 4]]})
    assert changed.input_sha256() != sample.input_sha256()
    with pytest.raises(ValueError, match="channels"):
        sample.validate_model(SimpleNamespace(n_vq=12), 8)
    with pytest.raises(ValueError, match="context"):
        sample.validate_model(model_config(), 1)
    with pytest.raises(ValidationError, match="unique"):
        LocalScoreBatch(samples=[sample, sample])


@pytest.mark.parametrize(
    "decisions,codes", [([1], [[1, 2, 3]]), ([0], []), ([0, 0, 1], [[1, 2, 3]])]
)
def test_score_input_rejects_misaligned_actions(decisions, codes):
    with pytest.raises(ValidationError):
        LocalScoreInput(
            sample_id="bad",
            prompt_rows=[[1, 17, 17, 17]],
            decisions=decisions,
            codes=codes,
        )


@pytest.mark.parametrize(
    "decisions,codes", [([1], []), ([0], [[1, 2, 3]]), ([0, 1], [[1, 2, 3]])]
)
def test_teacher_request_replays_original_rows(decisions, codes):
    cfg = model_config()
    raw = dict(
        sample_id="s", prompt_rows=[[1, 17, 17, 17]], decisions=decisions, codes=codes
    )
    payload = StagePayload(
        request_id="r",
        request=OmniRequest(inputs={}, params={"moss_local_score": raw}, metadata={}),
        data={},
    )
    data = build_score_request(
        payload, model=SimpleNamespace(config=cfg, score_context_length=8)
    )
    assert data.rows.tolist() == raw["prompt_rows"] + [[7, *row] for row in codes]
    assert data.req.sampling_params.max_new_tokens == 0


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            ],
        ),
    ],
)
def test_teacher_scoring_matches_generated_depth_probabilities(device):
    torch.manual_seed(42)
    hidden_size, frames, channels, vocab = 16, 5, 3, 17
    embeddings = torch.nn.ModuleList(
        [torch.nn.Embedding(32, hidden_size)]
        + [torch.nn.Embedding(vocab, hidden_size) for _ in range(channels)]
    )
    heads = torch.nn.ModuleList(
        [torch.nn.Linear(hidden_size, vocab, bias=False) for _ in range(channels)]
    )
    model = SimpleNamespace(
        dtype=torch.float32,
        n_vq=channels,
        config=SimpleNamespace(audio_assistant_slot_token_id=7),
        local_transformer=MossTTSLocalTransformer(
            hidden_size=hidden_size,
            num_heads=2,
            inner_size=32,
            num_layers=1,
            max_positions=channels + 1,
            rope_base=10000,
            layer_norm_eps=1e-5,
        ),
        local_text_lm_head=torch.nn.Linear(hidden_size, 2, bias=False),
        embedding_list=embeddings,
        audio_head_weight=lambda channel: heads[channel].weight,
        audio_embedding_weight=lambda channel: embeddings[channel + 1].weight,
        _sample_seeded_branchless=lambda logits, **kwargs: logits.argmax(-1),
    )
    model.local_transformer.to(device)
    model.local_text_lm_head.to(device)
    embeddings.to(device)
    heads.to(device)
    hidden = torch.randn(frames, hidden_size, device=device)
    inputs = dict(
        hidden_states=hidden,
        text_temperature=torch.full((frames,), 1.3),
        text_top_p=torch.ones(frames),
        text_top_k=torch.full((frames,), -1),
        audio_temperature=torch.full((frames,), 1.3),
        audio_top_p=torch.ones(frames),
        audio_top_k=torch.full((frames,), -1),
        seeds=torch.arange(frames),
        base_positions=torch.arange(frames),
        return_logprobs=True,
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        decisions, codes, _, expected_decisions, expected_codes = (
            MossTTSLocalSGLangModel.decode_frame_graphable(model, **inputs)
        )
        for chunk_size in (1, 3, 8):
            actual_decisions, actual_codes = score_local_depth(
                model, hidden, decisions, codes, temperature=1.3, chunk_size=chunk_size
            )
            torch.testing.assert_close(
                actual_decisions, expected_decisions, atol=2e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                actual_codes, expected_codes, atol=2e-6, rtol=1e-6
            )


def test_teacher_batch_endpoint_returns_input_order():
    class ScoreClient:
        async def completion(self, request, *, request_id):
            sample = LocalScoreInput.model_validate(
                request.extra_params["moss_local_score"]
            )
            assert request.stream is False and request.sampling.max_new_tokens == 0
            return SimpleNamespace(
                omni_rollout={
                    "sample_id": sample.sample_id,
                    "input_sha256": sample.input_sha256(),
                }
            )

    app = FastAPI()
    app.state.client = ScoreClient()
    register_local_scoring(app)
    samples = [
        score_input().model_dump(),
        score_input().model_copy(update={"sample_id": "second"}).model_dump(),
    ]
    with TestClient(app) as client:
        response = client.post("/score_actions", json={"samples": samples})
    assert response.status_code == 200
    assert [row["sample_id"] for row in response.json()["results"]] == [
        "sample",
        "second",
    ]


def test_teacher_prefill_aligns_mixed_request_boundaries(monkeypatch):
    from sglang_omni.models.moss_tts_local import scoring

    def depth(model, hidden, decisions, codes, **kwargs):
        return hidden[:, 0], hidden[:, :1].expand(-1, model.n_vq)

    monkeypatch.setattr(scoring, "score_local_depth", depth)
    monkeypatch.setattr(scoring, "moss_tts_local_model_identity", lambda config: {})
    first = score_input().model_copy(update={"prompt_rows": [[1, 17, 17, 17]] * 2})
    second = score_input().model_copy(
        update={"prompt_rows": [[1, 17, 17, 17]] * 3, "codes": [], "decisions": [1]}
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                score=sample,
                rows=torch.zeros(len(sample.prompt_rows) + len(sample.codes), 4),
            )
        )
        for sample in [first, second]
    ]
    runner = object.__new__(scoring.LocalScoreModelRunner)
    runner.model = SimpleNamespace(
        n_vq=3,
        config=None,
        teacher_weight_sha256="checksum",
        teacher_weight_version="v1",
    )
    runner.score_chunk_size = 8
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=-torch.arange(6).float()[:, None])
    )
    runner.post_prefill(result, None, None, requests)
    assert requests[0].data.score_result["decision_logprobs"] == [-1.0, -2.0]
    assert requests[0].data.score_result["code_logprobs"] == [[-1.0, -1.0, -1.0]]
    assert requests[1].data.score_result["decision_logprobs"] == [-5.0]
    assert requests[1].data.score_result["code_logprobs"] == []
