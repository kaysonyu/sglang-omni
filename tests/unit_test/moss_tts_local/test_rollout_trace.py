# SPDX-License-Identifier: Apache-2.0
"""MOSS rollout admission, action alignment and terminal payload contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.client import Client
from sglang_omni.models.moss_tts_local import request_builders as rb
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.moss_tts_local.rollout_trace import selected_action_logprobs
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, OmniRequest, StagePayload


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("batch", [1, 16, 64])
@pytest.mark.parametrize("vocab", [2, 1024])
def test_selected_logprobs_graph_replay(batch, vocab):
    torch.manual_seed(42)
    logits = torch.randn(batch, vocab * 2, device="cuda")[:, ::2]
    actions = torch.randint(vocab, (batch * 2,), device="cuda")[::2]
    temperature = torch.linspace(0.2, 2.0, batch * 2, device="cuda")[::2]
    selected_action_logprobs(logits, actions, temperature)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = selected_action_logprobs(logits, actions, temperature)
    for magnitude in (1.0, 30.0):
        logits.normal_().mul_(magnitude)
        actions.random_(vocab)
        temperature.uniform_(0.2, 2.0)
        expected = torch.log_softmax(logits.float() / temperature[:, None], -1)
        expected = expected.gather(1, actions[:, None]).squeeze(1)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)


@pytest.fixture
def config():
    return SimpleNamespace(
        n_vq=2,
        audio_vocab_size=8,
        audio_pad_code=8,
        audio_start_token_id=151669,
        audio_user_slot_token_id=151654,
        audio_assistant_slot_token_id=151656,
        audio_end_token_id=151670,
        vocab_size=151936,
        vocab_size_list=[151936, 8, 8],
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=8,
        rms_norm_eps=1e-6,
        gpt2_config=dict(
            n_head=2,
            n_inner=8,
            rope_base=10000.0,
            layer_norm_epsilon=1e-5,
            activation_function="gelu",
        ),
    )


def payload(params=None):
    return StagePayload(
        request_id="rollout",
        data={},
        request=OmniRequest(inputs={"text": "hello"}, params=params or {}, metadata={}),
    )


def state():
    return MossTTSLocalState(
        return_omni_rollout=True,
        return_logprob=True,
        generation_kwargs=dict(
            max_new_tokens=5,
            text_temperature=1.0,
            audio_temperature=1.0,
            text_top_p=1.0,
            audio_top_p=1.0,
            text_top_k=-1,
            audio_top_k=-1,
            audio_repetition_penalty=1.0,
        ),
    )


def test_startup_mode_is_opt_in():
    normal = MossTTSLocalPipelineConfig(model_path="test-model")
    rl = MossTTSLocalPipelineConfig(model_path="test-model", enable_rl=True)
    assert not normal.enable_rl
    assert "enable_rl" not in normal.stage_factory_kwargs("tts_engine")
    assert rl.stage_factory_kwargs("tts_engine")["enable_rl"] is True


def test_explicit_sampling_reaches_preprocessing():
    p = payload(
        dict(
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            return_omni_rollout=True,
            return_logprob=True,
        )
    )
    p.request.metadata[EXPLICIT_GENERATION_PARAMS_KEY] = [
        "temperature",
        "top_p",
        "top_k",
    ]
    s = rb.build_moss_tts_local_state(p)
    assert s.return_omni_rollout and s.return_logprob
    assert s.generation_kwargs["audio_temperature"] == 0.7
    assert s.generation_kwargs["audio_top_k"] == -1
    assert s.generation_kwargs["audio_top_p"] == 1.0


@pytest.mark.parametrize("enabled", [False, True])
def test_rollout_admission_requires_enabled_server(monkeypatch, config, enabled):
    s = state()
    prepared = rb.MossTTSLocalPreparedRequest(
        s, [1], torch.tensor([1]), torch.tensor([[1, 8, 8]]), s.generation_kwargs
    )
    monkeypatch.setattr(rb, "pop_prepared_moss_tts_local_request", lambda p: prepared)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_serving",
        lambda: SimpleNamespace(weight_version="v1"),
    )
    model = SimpleNamespace(config=config, enable_rl=enabled)
    if not enabled:
        with pytest.raises(ValueError, match="enable_rl"):
            rb.build_sglang_moss_tts_local_request(payload(), model=model)
    else:
        data = rb.build_sglang_moss_tts_local_request(payload(), model=model)
        assert data.return_omni_rollout
        assert data.admission_weight_version == "v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("audio_top_p", 0.8),
        ("audio_top_k", 25),
        ("audio_repetition_penalty", 1.1),
        ("text_temperature", 0.0),
    ],
)
def test_rollout_rejects_unsupported_policy(monkeypatch, config, field, value):
    s = state()
    s.generation_kwargs[field] = value
    prepared = rb.MossTTSLocalPreparedRequest(
        s, [1], torch.tensor([1]), torch.tensor([[1, 8, 8]]), s.generation_kwargs
    )
    monkeypatch.setattr(rb, "pop_prepared_moss_tts_local_request", lambda p: prepared)
    with pytest.raises(ValueError):
        rb.build_sglang_moss_tts_local_request(
            payload(), model=SimpleNamespace(config=config, enable_rl=True)
        )


@pytest.mark.parametrize("frames,finish", [(2, "stop"), (2, "length"), (0, "stop")])
def test_trace_preserves_actions_and_stop_boundary(config, frames, finish):
    decisions = [0] * frames + ([1] if finish == "stop" else [])
    data = rb.MossTTSLocalSGLangRequestData(
        state=state(),
        model_config=config,
        prompt_rows=torch.tensor([[1, 8, 8]]),
        input_ids=torch.tensor([1]),
        return_omni_rollout=True,
        admission_weight_version="v1",
        weight_version="v1",
        finish_reason=finish,
        output_rows=[torch.tensor([151656, 2, 3]) for _ in range(frames)],
        output_decisions=[torch.tensor(x) for x in decisions],
        output_decision_logprobs=[torch.tensor(-0.5) for _ in decisions],
        output_code_logprobs=[torch.tensor([-0.7, -0.9]) for _ in range(frames)],
    )
    result = rb.apply_sglang_moss_tts_local_result(payload(), data)
    trace = Client.default_result_builder("rollout", result.data).omni_rollout
    assert trace["version"] == 2
    assert trace["replay_inputs"]["prompt_rows"] == [[1, 8, 8]]
    assert trace["action_streams"][0]["actions"] == decisions
    assert trace["action_streams"][1]["actions"] == [[2, 3]] * frames
    assert trace["total_action_count"] == len(decisions) + frames * 2
    if frames == 0:
        assert result.data.get("audio_codes") is None
        scheduler = object.__new__(MossTTSLocalStreamingVocoderScheduler)
        result = scheduler.vocode_batch([result])[0]
        assert (
            Client.default_result_builder("rollout", result.data).omni_rollout == trace
        )
        assert result.data["modality"] == "text"


def test_rollout_rejects_cross_version_result(config):
    data = rb.MossTTSLocalSGLangRequestData(
        state=state(),
        model_config=config,
        return_omni_rollout=True,
        admission_weight_version="v1",
        weight_version="v2",
    )
    with pytest.raises(RuntimeError, match="crossed"):
        rb.apply_sglang_moss_tts_local_result(payload(), data)


def test_rl_instance_requires_rollout_request(monkeypatch, config):
    s = state()
    s.return_omni_rollout = False
    prepared = rb.MossTTSLocalPreparedRequest(
        s, [1], torch.tensor([1]), torch.tensor([[1, 8, 8]]), s.generation_kwargs
    )
    monkeypatch.setattr(rb, "pop_prepared_moss_tts_local_request", lambda p: prepared)
    with pytest.raises(ValueError, match="return_omni_rollout"):
        rb.build_sglang_moss_tts_local_request(
            payload(), model=SimpleNamespace(config=config, enable_rl=True)
        )


def test_generate_accepts_reference_conditioning():
    from sglang_omni.serve.openai_api import build_rollout_generate_request
    from sglang_omni.serve.protocol import RolloutGenerateRequest

    conditioning = {
        "text": "hello",
        "references": [{"audio": "https://example.com/ref.wav", "text": "reference"}],
    }
    request = RolloutGenerateRequest(
        prompt=conditioning, return_logprob=True, return_omni_rollout=True
    )
    result = build_rollout_generate_request(request)
    assert result.prompt == conditioning
    assert result.extra_params["return_omni_rollout"] is True
