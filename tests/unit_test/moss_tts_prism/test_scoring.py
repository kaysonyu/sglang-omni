# SPDX-License-Identifier: Apache-2.0
"""Prism fixed-action scoring and frozen teacher API contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import JsonValue, ValidationError
from torch import nn

from sglang_omni.models.moss_tts_prism.config import MossTTSPrismScorePipelineConfig
from sglang_omni.models.moss_tts_prism.scoring import (
    PrismScoreModelRunner,
    PrismTeacherWeightChecker,
    build_score_request,
    score_result,
)
from sglang_omni.models.moss_tts_prism.scoring_protocol import (
    PrismScoreBatch,
    PrismScoreInput,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismBatchInputs,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.serve.action_scoring import register_action_scoring


def score_input(
    *, sample_id: str = "sample", finish_reason: str = "stop", frames: int = 2
) -> PrismScoreInput:
    return PrismScoreInput(
        sample_id=sample_id,
        layout="moss_prism_typed_v1",
        prompt_rows=[[7, 0, 0, 0], [0, 0, 2, 3], [8, 0, 0, 0], [7, 0, 0, 0]],
        item_kind=[0, 1, 0, 0],
        audio_role=[0, 1, 0, 0],
        successor_audio_mask=[True, False, False, False],
        successor_audio_codes=[[0, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        target_history_retention_mask=[[True, True, True]] * 4,
        codes=[[index, index + 1, index + 2] for index in range(frames)],
        finish_reason=finish_reason,
        temperature=0.7,
    )


def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        n_vq=3,
        speech_vocab_size=17,
        language_config=SimpleNamespace(vocab_size=32),
        text_pad_idx=0,
        audio_start_token_id=7,
        prism_input_rvq_channels=(1, 3),
        embedding_head_usage="untied",
    )


def request_for(
    sample: PrismScoreInput, model: MossTTSPrismSGLangModel
) -> SchedulerRequest:
    payload = StagePayload(
        request_id=sample.sample_id,
        request=OmniRequest(
            inputs={}, params={"moss_prism_score": sample.model_dump()}
        ),
        data={},
    )
    return SchedulerRequest(
        request_id=sample.sample_id,
        data=build_score_request(payload, model=model, context_length=128),
    )


def test_teacher_config_has_only_scoring_stage() -> None:
    config = MossTTSPrismScorePipelineConfig(model_path="model")
    assert config.resolved_entry_stage == "tts_engine"
    assert len(config.stages) == 1 and config.stages[0].terminal
    assert config.architecture == "MossTTSPrismModel"
    assert config.stages[0].engine.disable_cuda_graph
    assert config.stages[0].engine.disable_radix_cache
    assert config.stages[0].engine.chunked_prefill_size == -1


def test_input_hash_and_model_geometry() -> None:
    sample = score_input()
    sample.validate_model(model_config(), 6)
    assert (
        sample.model_copy(update={"sample_id": "renamed"}).input_sha256()
        == sample.input_sha256()
    )
    assert (
        sample.model_copy(update={"temperature": 1.0}).input_sha256()
        != sample.input_sha256()
    )
    with pytest.raises(ValueError, match="context"):
        sample.validate_model(model_config(), 5)
    changed = sample.model_copy(deep=True)
    changed.codes[0][0] = 17
    with pytest.raises(ValueError, match="vocabulary"):
        changed.validate_model(model_config(), 6)
    changed = sample.model_copy(deep=True)
    changed.successor_audio_codes[0][0] = 1
    with pytest.raises(ValueError, match="successor"):
        changed.validate_model(model_config(), 6)
    changed = sample.model_copy(deep=True)
    changed.successor_audio_mask[-1] = True
    with pytest.raises(ValueError, match="target audio_start"):
        changed.validate_model(model_config(), 6)


@pytest.mark.parametrize(
    "updates",
    [
        {"codes": [[0, 1]]},
        {"item_kind": [0]},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"codes": [], "finish_reason": "length"},
    ],
)
def test_invalid_score_contract(updates: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        PrismScoreInput.model_validate({**score_input().model_dump(), **updates})


def test_score_batch_rejects_duplicate_ids_and_oversized_batch() -> None:
    with pytest.raises(ValidationError, match="unique"):
        PrismScoreBatch(samples=[score_input(), score_input()])
    with pytest.raises(ValidationError, match="64"):
        PrismScoreBatch(
            samples=[score_input(sample_id=str(index)) for index in range(65)]
        )


class ScoreAttentionBackend:
    def init_forward_metadata(self, forward_batch: SimpleNamespace) -> None:
        pass


class CausalLayer(nn.Module):
    def __init__(self, *, mlp_only: bool) -> None:
        super().__init__()
        self.mlp_only = mlp_only
        self.self_attn = SimpleNamespace(attn=SimpleNamespace(layer_id=0))

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimpleNamespace,
    ) -> torch.Tensor:
        if self.mlp_only:
            return hidden.tanh()
        else:
            outputs = []
            for values in hidden.split(forward_batch.extend_seq_lens_cpu):
                count = torch.arange(1, len(values) + 1, device=hidden.device)[:, None]
                outputs.append(values + values.cumsum(0) / count)
            return torch.cat(outputs)


def tiny_teacher() -> MossTTSPrismSGLangModel:
    torch.manual_seed(23)
    model = MossTTSPrismSGLangModel.__new__(MossTTSPrismSGLangModel)
    nn.Module.__init__(model)
    model.config = model_config()
    model.n_vq = 3
    model.model = None
    model.decode_inputs = None
    model.teacher_weight_sha256 = "frozen"
    model.teacher_weight_version = "teacher:0"
    model.model_identity = {"n_vq": 3}
    model.transformer = nn.Module()
    model.transformer.embed_tokens = nn.Embedding(32, 8)
    model.transformer.layers = nn.ModuleList(
        [CausalLayer(mlp_only=False), CausalLayer(mlp_only=True)]
    )
    model.audio_embeddings = nn.ModuleList([nn.Embedding(17, 8) for _ in range(3)])
    model.audio_lm_heads = nn.ModuleList(
        [nn.Linear(8, 17, bias=False) for _ in range(3)]
    )
    model.prediction_norms = nn.ModuleList([nn.LayerNorm(8) for _ in range(3)])
    model.stop_head = nn.Linear(8, 2, bias=False)
    model.topology = SimpleNamespace(
        execution_units=[
            SimpleNamespace(
                kind="loop", index=0, layer_start=0, layer_stop=2, rvq_heads=(1,)
            ),
            SimpleNamespace(
                kind="loop", index=0, layer_start=0, layer_stop=2, rvq_heads=(2,)
            ),
            SimpleNamespace(
                kind="ordinary", index=1, layer_start=0, layer_stop=1, rvq_heads=(3,)
            ),
        ],
        prediction_norm_index_by_execution_unit=[0, 1, 2],
        reapply_rvq_conditioning_in_loop=True,
    )
    return model


@pytest.mark.parametrize(
    "finish,frames", [("stop", 0), ("stop", 3), ("length", 1), ("length", 3)]
)
def test_teacher_matches_forced_generation_prefixes(finish: str, frames: int) -> None:
    model = tiny_teacher()
    sample = score_input(finish_reason=finish, frames=frames)
    request = request_for(sample, model)
    runner = PrismScoreModelRunner.__new__(PrismScoreModelRunner)
    runner.model, runner.device = model, torch.device("cpu")
    runner.score_chunk_size = 2
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=ScoreAttentionBackend())
    )
    carrier = request.data.inputs
    lengths = SimpleNamespace(
        extend_seq_lens_cpu=[len(carrier.rows)],
        positions=torch.arange(len(carrier.rows)),
    )
    actual = runner.custom_prefill_forward(
        lengths, SimpleNamespace(is_prefill_only=True), [request]
    ).logits_output
    predictions = range(len(sample.prompt_rows) - 1, len(carrier.rows))
    expected_codes, expected_stop = [], []
    for frame, position in enumerate(predictions):
        logprobs = torch.empty(1, model.n_vq)
        action = sample.codes[frame] if frame < frames else [0] * model.n_vq

        def force_action(logits: torch.Tensor, channel: int) -> torch.Tensor:
            logprobs[:, channel] = (logits / sample.temperature).log_softmax(-1)[
                :, action[channel]
            ]
            return torch.tensor([action[channel]])

        model.batch_inputs = PrismBatchInputs(
            rows=carrier.rows[: position + 1],
            item_kind=carrier.item_kind[: position + 1],
            audio_role=carrier.audio_role[: position + 1],
            retention=carrier.retention[: position + 1],
            successor_mask=carrier.successor_mask[: position + 1],
            successor_codes=carrier.successor_codes[: position + 1],
            sample_indices=torch.tensor([position]),
            sample=force_action,
        )
        output = MossTTSPrismSGLangModel.forward(
            model,
            carrier.rows[: position + 1, 0],
            torch.arange(position + 1),
            SimpleNamespace(extend_seq_lens_cpu=[position + 1]),
        )
        expected_codes.append(logprobs[0])
        expected_stop.append(output.next_token_logits.softmax(-1)[0, 1])
    torch.testing.assert_close(
        actual.customized_info["code_logprobs"][:frames],
        torch.stack(expected_codes)[:frames],
        atol=2e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual.customized_info["stop_probabilities"],
        torch.stack(expected_stop),
        atol=2e-6,
        rtol=1e-6,
    )


def test_mixed_scores_keep_boundaries_stop_count_and_input_order() -> None:
    model = tiny_teacher()
    runner = PrismScoreModelRunner.__new__(PrismScoreModelRunner)
    runner.model, runner.device = model, torch.device("cpu")
    runner.score_chunk_size = 2
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=ScoreAttentionBackend())
    )
    samples = [
        score_input(finish_reason="length", frames=3),
        score_input(sample_id="empty", frames=0),
    ]
    requests = [request_for(sample, model) for sample in samples]
    lengths = SimpleNamespace(
        extend_seq_lens_cpu=[len(request.data.inputs.rows) for request in requests],
        positions=torch.cat(
            [torch.arange(len(request.data.inputs.rows)) for request in requests]
        ),
    )
    batch = SimpleNamespace(is_prefill_only=True)
    output = runner.custom_prefill_forward(lengths, batch, requests)
    runner.post_prefill(output, lengths, batch, requests)
    for sample, request in zip(samples, requests, strict=True):
        result = json.loads(score_result(request.data).data["omni_rollout"])
        assert len(result["code_logprobs"]) == len(sample.codes)
        assert len(result["stop_probabilities"]) == sample.prediction_count
        assert result["input_sha256"] == sample.input_sha256()
        assert result["sample_id"] == sample.sample_id
        assert result["teacher_weight_sha256"] == "frozen"
        assert result["weight_version"] == "teacher:0"
    first = requests[0].data.score_result
    rows = len(requests[0].data.inputs.rows)
    alone = runner.custom_prefill_forward(
        SimpleNamespace(extend_seq_lens_cpu=[rows], positions=torch.arange(rows)),
        batch,
        requests[:1],
    ).logits_output
    torch.testing.assert_close(
        torch.tensor(first["code_logprobs"]), alone.customized_info["code_logprobs"]
    )


def test_teacher_checksum_ignores_runtime_buffers() -> None:
    model = tiny_teacher()
    model.register_buffer("cache", torch.zeros(2), persistent=False)
    checker = PrismTeacherWeightChecker(SimpleNamespace(model=model))
    checksum = checker.checksum()["per_gpu_checksum"]
    model.cache.add_(1)
    assert checker.checksum()["per_gpu_checksum"] == checksum
    with torch.no_grad():
        model.audio_lm_heads[0].weight.add_(1)
    assert checker.checksum()["per_gpu_checksum"] != checksum
    metadata = model.rollout_model_info()
    assert metadata["supports_action_scoring"]
    assert not metadata["supports_weight_update"]


def test_prism_endpoint_dispatch_order_and_validation() -> None:
    class ScoreClient:
        async def completion(
            self, request: SimpleNamespace, *, request_id: str
        ) -> SimpleNamespace:
            sample = PrismScoreInput.model_validate(
                request.extra_params["moss_prism_score"]
            )
            assert request.stream is False and request.sampling.max_new_tokens == 0
            if sample.sample_id == "bad":
                raise RuntimeError(
                    "Prism score input: Audio code is outside the model vocabulary"
                )
            await asyncio.sleep(0.01 if sample.sample_id == "sample" else 0)
            return SimpleNamespace(
                omni_rollout={
                    "sample_id": sample.sample_id,
                    "input_sha256": sample.input_sha256(),
                }
            )

    app = FastAPI()
    app.state.client = ScoreClient()
    register_action_scoring(app, ["MossTTSPrismModel"])
    with TestClient(app) as client:
        result = client.post(
            "/score_actions",
            json={
                "samples": [
                    score_input().model_dump(),
                    score_input(sample_id="second", frames=0).model_dump(),
                ]
            },
        )
        assert result.status_code == 200
        assert [sample["sample_id"] for sample in result.json()["results"]] == [
            "sample",
            "second",
        ]
        invalid = client.post(
            "/score_actions",
            json={"samples": [score_input(sample_id="bad").model_dump()]},
        )
        assert invalid.status_code == 400
        assert "PrismScoreBatch" in client.get("/openapi.json").text
