# SPDX-License-Identifier: Apache-2.0
"""Prism rollout actions, stop observations, and graph sampling."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.client.client import Client
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_prism.config import MossTTSPrismPipelineConfig
from sglang_omni.models.moss_tts_prism.model_runner import (
    STOP_SAMPLING_SEED_XOR,
    MossTTSPrismModelRunner,
    sample_prism_tokens,
)
from sglang_omni.models.moss_tts_prism.request_builders import (
    PrismRequestData,
    apply_prism_result,
    build_prism_request,
)
from sglang_omni.models.moss_tts_prism.rollout_trace import (
    build_prism_rollout_trace,
    prism_model_identity,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismDecodeInputs,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput, SchedulerRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator


@pytest.fixture
def carrier():
    return {
        "input_ids": torch.tensor([[7, 0, 0], [8, 0, 0]]),
        "item_kind": torch.zeros(2, dtype=torch.long),
        "audio_role": torch.zeros(2, dtype=torch.long),
        "successor_audio_mask": torch.zeros(2, dtype=torch.bool),
        "successor_audio_codes": torch.zeros(2, 2, dtype=torch.long),
        "target_history_retention_mask": torch.ones(2, 2, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "frames,finish",
    [(0, "stop"), (1, "stop"), (3, "stop"), (1, "length"), (3, "length")],
)
def test_trace_preserves_carrier_and_only_counts_rvq_actions(carrier, frames, finish):
    probabilities = [0.02] * frames + ([0.6] if finish == "stop" else [])
    trace = json.loads(
        build_prism_rollout_trace(
            prompt=carrier,
            codes=torch.zeros(frames, 2, dtype=torch.long),
            code_logprobs=torch.full((frames, 2), -2.0),
            stop_probabilities=torch.tensor(probabilities),
            finish_reason=finish,
            request_id="server-uuid",
            admission_weight_version="step:0",
            model_identity={"audio_vocab_size": 16},
            sampling={"audio_temperature": 0.7},
        )
    )
    assert trace["total_action_count"] == frames * 2
    assert [stream["name"] for stream in trace["action_streams"]] == ["codes"]
    assert trace["action_streams"][0]["shape"] == [frames, 2]
    assert trace["non_action_outputs"][0]["values"] == pytest.approx(probabilities)
    assert trace["stop_semantics"] == "pre_frame_threshold_v1"
    assert trace["replay_inputs"]["prompt_rows"] == carrier["input_ids"].tolist()
    for name in carrier.keys() - {"input_ids"}:
        assert trace["replay_inputs"][name] == carrier[name].tolist()


@pytest.mark.parametrize("frames,finish", [(0, "stop"), (2, "stop"), (2, "length")])
def test_sampled_stop_trace_counts_real_binary_actions(carrier, frames, finish):
    probabilities = torch.tensor([0.8] * frames + ([0.05] if finish == "stop" else []))
    actions = torch.tensor([0] * frames + ([1] if finish == "stop" else []))
    selected = torch.where(actions.bool(), probabilities, 1 - probabilities)
    trace = json.loads(
        build_prism_rollout_trace(
            prompt=carrier,
            codes=torch.zeros(frames, 2, dtype=torch.long),
            code_logprobs=torch.full((frames, 2), -2.0),
            stop_probabilities=probabilities,
            stop_actions=actions,
            stop_logprobs=selected.log(),
            finish_reason=finish,
            request_id="server-uuid",
            admission_weight_version="step:0",
            model_identity={"audio_vocab_size": 16},
            sampling={"audio_temperature": 1.0, "stop_sampling": True},
        )
    )
    assert trace["stop_semantics"] == "pre_frame_bernoulli_v1"
    assert [stream["name"] for stream in trace["action_streams"]] == [
        "codes",
        "stop_decisions",
    ]
    assert trace["action_streams"][1]["actions"] == actions.tolist()
    assert trace["action_streams"][1]["logprobs"] == pytest.approx(
        selected.log().tolist()
    )
    assert trace["total_action_count"] == frames * 2 + len(actions)
    assert "stop_threshold" not in trace["sampling"]


def test_sampled_stop_trace_rejects_inconsistent_selected_logprob(carrier):
    with pytest.raises(RuntimeError, match="scores disagree"):
        build_prism_rollout_trace(
            prompt=carrier,
            codes=torch.zeros((1, 2), dtype=torch.long),
            code_logprobs=torch.full((1, 2), -2.0),
            stop_probabilities=torch.tensor([0.8, 0.05]),
            stop_actions=torch.tensor([0, 1]),
            stop_logprobs=torch.tensor([-2.0, -2.0]),
            finish_reason="stop",
            request_id="server-uuid",
            admission_weight_version="step:0",
            model_identity={"audio_vocab_size": 16},
            sampling={"audio_temperature": 1.0, "stop_sampling": True},
        )


@pytest.mark.parametrize("revision", [4, 6, 7, 42])
def test_identity_hashes_original_artifact_bytes(tmp_path, revision):
    config = {
        "n_vq": 3,
        "speech_vocab_size": 17,
        "sampling_rate": 24000,
        "prompt_renderer_revision": revision,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    for filename in (
        "modeling_moss_tts.py",
        "processing_moss_tts.py",
        "prompt_protocol.py",
    ):
        (tmp_path / filename).write_text("original\n")
    first = prism_model_identity(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(config, indent=2))
    second = prism_model_identity(str(tmp_path))
    assert first["config_sha256"] != second["config_sha256"]
    assert first["processor_sha256"] == second["processor_sha256"]
    assert first["n_vq"] == 3
    assert first["audio_vocab_size"] == 17
    assert first["sample_rate"] == 24000

    assert len(first["prompt_protocol_sha256"]) == 64
    (tmp_path / "prompt_protocol.py").write_text("updated renderer")
    changed = prism_model_identity(str(tmp_path))
    assert changed["prompt_protocol_sha256"] != second["prompt_protocol_sha256"]
    assert changed["config_sha256"] == second["config_sha256"]
    config["prompt_renderer_revision"] = 3
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "prompt_protocol.py").unlink()
    with pytest.raises(FileNotFoundError, match="prompt_protocol"):
        prism_model_identity(str(tmp_path))


def test_rl_switch_defaults_off():
    normal = MossTTSPrismPipelineConfig(model_path="model")
    enabled = MossTTSPrismPipelineConfig(model_path="model", enable_rl=True)
    assert normal.stage_factory_kwargs("tts_engine") == {}
    assert enabled.stage_factory_kwargs("tts_engine") == {"enable_rl": True}


@pytest.fixture
def request_payload(carrier):
    return StagePayload(
        request_id="rollout",
        request=OmniRequest(inputs="test", params={"return_audio": False}),
        data={
            "prism_inputs": carrier,
            "return_logprob": True,
            "return_omni_rollout": True,
            "generation_kwargs": {
                "audio_temperature": 0.7,
                "audio_top_p": 1.0,
                "audio_top_k": -1,
                "audio_repetition_penalty": 1.0,
                "max_new_tokens": 3,
                "seed": 42,
            },
        },
    )


def request_model():
    return SimpleNamespace(
        enable_rl=True,
        model_identity={"audio_vocab_size": 16, "n_vq": 2},
        config=SimpleNamespace(
            audio_end_token_id=9, n_vq=2, language_config=SimpleNamespace(vocab_size=16)
        ),
    )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("audio_temperature", 0),
        ("audio_temperature", float("nan")),
        ("audio_top_p", 0.9),
        ("audio_top_k", 4),
        ("audio_repetition_penalty", 1.2),
    ],
)
def test_rollout_rejects_non_neutral_sampling(request_payload, parameter, value):
    request_payload.data["generation_kwargs"][parameter] = value
    with pytest.raises(ValueError, match="Prism RL request requires"):
        build_prism_request(request_payload, model=request_model())


def test_audio_free_rollout_and_version_check(monkeypatch, request_payload):
    from sglang_omni.models.moss_tts_local.streaming_vocoder import (
        MossTTSLocalStreamingVocoderScheduler,
    )
    from sglang_omni.models.moss_tts_prism import request_builders

    monkeypatch.setattr(
        request_builders,
        "get_serving",
        lambda: SimpleNamespace(weight_version="step:0"),
    )
    request = build_prism_request(request_payload, model=request_model())
    request.output_codes = [torch.tensor([1, 2])]
    request.code_logprobs = [torch.tensor([-1.0, -2.0])]
    request.stop_probabilities = [torch.tensor(0.01), torch.tensor(0.5)]
    request.finish_reason = "stop"
    request.weight_version = "step:0"
    result = apply_prism_result(request)
    scheduler = MossTTSLocalStreamingVocoderScheduler.__new__(
        MossTTSLocalStreamingVocoderScheduler
    )

    def fail_decode(codes):
        raise AssertionError("audio-free rollout must not run the vocoder")

    monkeypatch.setattr(scheduler, "decode_codes_rows", fail_decode)
    result = scheduler.vocode(result)
    assert result.data.get("audio_codes") is None
    trace = json.loads(result.data["omni_rollout"])
    assert trace["action_streams"][0]["actions"] == [[1, 2]]
    assert result.data["completion_tokens"] == 1
    assert result.data["prompt_tokens"] == 2
    assert result.data["weight_version"] == "step:0"
    request.weight_version = "step:1"
    with pytest.raises(RuntimeError, match="crossed a weight update"):
        apply_prism_result(request)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_stop_observations_survive_reused_output_buffers(carrier, device):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=True,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    requests = [
        SchedulerRequest(request_id=str(i), data=PrismRequestData(prompt=carrier))
        for i in range(2)
    ]
    scores = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device=device)
    result = SimpleNamespace(
        logits_output=LogitsProcessorOutput(
            next_token_logits=torch.tensor([[4.0, 0.0], [0.0, 4.0]], device=device),
            customized_info={
                "audio_codes": torch.tensor([[1, 2], [3, 4]], device=device),
                "code_logprobs": scores,
            },
        )
    )
    runner.post_decode(result, None, None, requests)
    token_ids = runner.resolve_host_token_ids(result).tolist()
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            str(i): RequestOutput(request_id=str(i), data=token)
            for i, token in enumerate(token_ids)
        },
    )
    expected = scores[0].clone()
    scores.zero_()
    result.logits_output.customized_info["stop_probabilities"].zero_()
    assert token_ids == [8, 9]
    assert len(requests[0].data.output_codes) == 1
    assert not requests[1].data.output_codes
    assert not requests[1].data.code_logprobs
    torch.testing.assert_close(requests[0].data.code_logprobs[0], expected)
    assert requests[0].data.stop_probabilities[0].item() == pytest.approx(
        torch.tensor([4.0, 0.0]).softmax(0)[1].item()
    )


def test_sampled_stop_controls_emission_and_records_selected_scores(
    monkeypatch, carrier
):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=True,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    requests = [
        SchedulerRequest(request_id=str(i), data=PrismRequestData(prompt=carrier))
        for i in range(2)
    ]
    for request in requests:
        request.data.stop_sampling = True
    monkeypatch.setattr(
        MossTTSModelRunner,
        "sample_tokens",
        staticmethod(
            lambda logits, **kwargs: torch.tensor([0, 1], device=logits.device)
        ),
    )
    logits = torch.tensor([[0.0, 2.0], [4.0, 0.0]])
    result = SimpleNamespace(
        logits_output=LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={
                "audio_codes": torch.tensor([[1, 2], [3, 4]]),
                "code_logprobs": torch.full((2, 2), -0.5),
            },
        )
    )
    runner.post_decode(result, None, None, requests)
    token_ids = runner.resolve_host_token_ids(result).tolist()
    assert token_ids == [8, 9]
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            str(i): RequestOutput(request_id=str(i), data=token)
            for i, token in enumerate(token_ids)
        },
    )
    assert [int(request.data.stop_actions[0]) for request in requests] == [0, 1]
    assert [len(request.data.output_codes) for request in requests] == [1, 0]
    actual_scores = torch.stack([request.data.stop_logprobs[0] for request in requests])
    expected_scores = logits.log_softmax(-1)[torch.arange(2), torch.tensor([0, 1])]
    torch.testing.assert_close(actual_scores, expected_scores)


def test_sampled_stop_can_serve_without_rollout_trace(monkeypatch, carrier):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=False,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    requests = [
        SchedulerRequest(request_id=str(i), data=PrismRequestData(prompt=carrier))
        for i in range(2)
    ]
    for request in requests:
        request.data.stop_sampling = True
    monkeypatch.setattr(
        MossTTSModelRunner,
        "sample_tokens",
        staticmethod(lambda logits, **kwargs: torch.tensor([0, 1])),
    )
    result = SimpleNamespace(
        logits_output=LogitsProcessorOutput(
            next_token_logits=torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
            customized_info={"audio_codes": torch.tensor([[1, 2], [3, 4]])},
        )
    )
    runner.post_decode(result, None, None, requests)
    token_ids = runner.resolve_host_token_ids(result).tolist()
    assert token_ids == [8, 9]
    assert "sampled_stop_logprobs" not in result.logits_output.customized_info
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            str(i): RequestOutput(request_id=str(i), data=token)
            for i, token in enumerate(token_ids)
        },
    )
    assert [len(request.data.output_codes) for request in requests] == [1, 0]
    assert all(not request.data.stop_actions for request in requests)


@pytest.mark.parametrize("mode", [False, True])
def test_request_resolves_stop_mode_once(
    request_payload: StagePayload, monkeypatch: pytest.MonkeyPatch, mode: bool
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.moss_tts_prism.request_builders.get_serving",
        lambda: SimpleNamespace(weight_version="step:0"),
    )
    request_payload.data["generation_kwargs"]["stop_sampling"] = mode
    request = build_prism_request(request_payload, model=request_model())
    assert request.stop_sampling is mode
    request.state.generation_kwargs["stop_sampling"] = not mode
    assert request.stop_sampling is mode


def run_stop_batch(
    logits: torch.Tensor,
    modes: list[bool],
    seeds: list[int],
    positions: list[int],
    enable_rl: bool,
) -> tuple[torch.Tensor, list[PrismRequestData], dict[str, torch.Tensor]]:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=enable_rl,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    requests = [
        SchedulerRequest(
            request_id=str(index),
            data=PrismRequestData(
                stop_sampling=mode,
                sampling_seed=seed,
                output_codes=[
                    torch.zeros(2, dtype=torch.long) for _ in range(position)
                ],
            ),
        )
        for index, (mode, seed, position) in enumerate(
            zip(modes, seeds, positions, strict=True)
        )
    ]
    result = SimpleNamespace(
        logits_output=LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={
                "audio_codes": torch.zeros(
                    len(requests), 2, dtype=torch.long, device=logits.device
                ),
                "code_logprobs": torch.zeros(len(requests), 2, device=logits.device),
            },
        )
    )
    runner.post_decode(result, None, None, requests)
    tokens = runner.resolve_host_token_ids(result).clone()
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            str(index): RequestOutput(request_id=str(index), data=int(token))
            for index, token in enumerate(tokens)
        },
    )
    return (
        tokens.eq(9).long(),
        [request.data for request in requests],
        result.logits_output.customized_info,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_stop_sampling_is_independent_of_batch_order_and_rl_recording(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    logits = torch.tensor(
        [[0.1, 0.7], [4.0, 0.0], [-0.4, 0.3], [0.0, 2.0]], device=device
    )
    modes, seeds, positions = [True, False, True, True], [3, 17, 42, 913], [0, 2, 11, 1]
    actions, states, scores = run_stop_batch(logits, modes, seeds, positions, True)
    ordinary, _, ordinary_scores = run_stop_batch(
        logits, modes, seeds, positions, False
    )
    torch.testing.assert_close(actions, ordinary, rtol=0, atol=0)
    assert "sampled_stop_logprobs" not in ordinary_scores
    assert actions[1] == int(logits[1].softmax(-1)[1] > 0.1)
    assert not states[1].stop_actions and not states[1].stop_logprobs
    assert scores["sampled_stop_logprobs"].shape == (3,)
    expected = (
        logits[[0, 2, 3]]
        .log_softmax(-1)
        .gather(1, actions[[0, 2, 3]].to(device)[:, None])
        .squeeze(1)
    )
    actual = torch.stack([states[index].stop_logprobs[0] for index in (0, 2, 3)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for index in (0, 2, 3):
        singleton, _, _ = run_stop_batch(
            logits[index : index + 1], [True], [seeds[index]], [positions[index]], True
        )
        assert singleton[0] == actions[index]
    order = [3, 1, 0, 2]
    reordered, _, _ = run_stop_batch(
        logits[order],
        [modes[index] for index in order],
        [seeds[index] for index in order],
        [positions[index] for index in order],
        True,
    )
    torch.testing.assert_close(reordered, actions[order], rtol=0, atol=0)
    scores["sampled_stop_actions"].fill_(7)
    scores["sampled_stop_logprobs"].fill_(float("nan"))
    torch.testing.assert_close(
        torch.stack([states[index].stop_logprobs[0] for index in (0, 2, 3)]), expected
    )
    assert [int(states[index].stop_actions[0]) for index in (0, 2, 3)] == actions[
        [0, 2, 3]
    ].tolist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("vocab", [2, 16])
def test_shared_sampler_preserves_actions_and_selected_scores(
    device: str, vocab: int
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(8, vocab, generator=generator).to(device)
    ones = torch.ones(8, device=device)
    seeds = torch.arange(8, device=device) ^ STOP_SAMPLING_SEED_XOR
    positions = torch.arange(8, device=device) * 3
    scores = torch.empty(8, device=device)
    arguments = dict(
        temperature=ones,
        top_p=ones,
        top_k=torch.full((8,), -1, device=device),
        seeds=seeds,
        positions=positions,
    )
    recorded = sample_prism_tokens(logits, logprob_output=scores, **arguments)
    ordinary = sample_prism_tokens(logits, logprob_output=None, **arguments)
    assert torch.equal(recorded, ordinary)
    torch.testing.assert_close(
        scores,
        logits.log_softmax(-1).gather(1, recorded[:, None]).squeeze(1),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rl_decode_graph_reads_live_temperature_and_seed():
    config = SimpleNamespace(
        n_vq=2, text_pad_idx=0, prism_input_rvq_channels=(1,), speech_vocab_size=16
    )
    buffers = PrismDecodeInputs(config, 4, torch.device("cuda"), enable_rl=True)
    model = MossTTSPrismSGLangModel.__new__(MossTTSPrismSGLangModel)
    nn.Module.__init__(model)
    model.topology = SimpleNamespace(
        execution_units=[SimpleNamespace(rvq_heads=(1, 2))],
        prediction_norm_index_by_execution_unit=[0],
    )
    model.prediction_norms = nn.ModuleList([nn.Identity()])
    torch.manual_seed(23)
    model.stop_head = nn.Linear(4, 2, bias=False).to(
        device="cuda", dtype=torch.bfloat16
    )
    model.audio_lm_heads = nn.ModuleList(
        [
            nn.Linear(4, 16, bias=False).to(device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
    )
    hidden = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    inputs = buffers.for_batch(4)
    buffers.top_k.fill_(-1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.sample_audio_heads(0, hidden, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stops, sampled = model.sample_audio_heads(0, hidden, inputs)
    for seed, temperatures in [(42, [0.7, 1.0, 1.2, 2.0]), (17, [2.0, 1.2, 1.0, 0.7])]:
        buffers.seeds.copy_(torch.arange(4, device="cuda") + seed)
        buffers.temperature.copy_(torch.tensor(temperatures, device="cuda"))
        graph.replay()
        actual = inputs.code_logprobs.clone()
        actual_codes = [value.clone() for _, value in sampled]
        expected_stops, expected_sampled = model.sample_audio_heads(0, hidden, inputs)
        torch.testing.assert_close(stops, expected_stops, rtol=0, atol=0)
        for channel, (_, expected_codes) in enumerate(expected_sampled):
            torch.testing.assert_close(
                actual_codes[channel], expected_codes, rtol=0, atol=0
            )
            expected_scores = (
                (
                    model.audio_lm_heads[channel](hidden).float()
                    / buffers.temperature[:, None]
                )
                .log_softmax(-1)
                .gather(1, expected_codes[:, None])
                .squeeze(1)
            )
            torch.testing.assert_close(
                actual[:, channel], expected_scores, rtol=1e-5, atol=1e-6
            )


def test_prompt_matches_artifact_typed_carrier():
    import os

    from transformers import AutoConfig, AutoTokenizer

    from sglang_omni.models.moss_tts.hf_loading import (
        load_moss_processor_class,
        moss_transformers_processor_compat,
    )
    from sglang_omni.models.moss_tts_prism.request_builders import (
        preprocess_prism_payload,
    )

    model_path = os.environ.get("MOSS_TTS_PRISM_TEST_MODEL")
    if not model_path:
        pytest.skip("MOSS_TTS_PRISM_TEST_MODEL is required")
    with moss_transformers_processor_compat():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = load_moss_processor_class(model_path)(
            tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
            audio_tokenizer=None,
            model_config=config,
        )
    references = [
        torch.arange(length * config.n_vq).reshape(length, config.n_vq)
        % config.speech_vocab_size
        for length in (2, 5)
    ]

    class ReferenceEncoder:
        def encode(self, uri):
            return references[int(uri)]

        def encode_data_uri(self, uri):
            return references[0]

    for task, script, count, instruction in [
        ("TTS", "Hello.", 0, None),
        ("Instruction", "Hello.", 0, '{"Speed":"slow"}'),
        ("Full VoiceClone", "Hello.", 1, None),
        ("Attribute Clone Triplet", "Hello.", 2, None),
        ("sing clone", "Hello.", 1, None),
        (
            "Local Instruction",
            "Hello. Goodbye.",
            0,
            json.dumps(
                {
                    "global instruction": {"Speed": "slow"},
                    "local instruction": [
                        {"text": "Goodbye.", "local instruction": {"Emotion": "sad"}}
                    ],
                }
            ),
        ),
    ]:
        inputs = {
            "script": script,
            "task_type": task,
            "show_language": True,
            "language": "English",
            "tokens_control": True,
            "global_tokens": 120,
            "global_instruction": instruction,
            "references": [
                {"id": f"audio{i + 1}", "uri": str(i)} for i in range(count)
            ],
        }
        prepared = preprocess_prism_payload(
            StagePayload(
                request_id="prompt", request=OmniRequest(inputs=inputs), data={}
            ),
            processor=processor,
            reference_encoder=ReferenceEncoder(),
        )
        message = processor.build_user_message(
            script=script,
            task_type=task,
            language="English",
            tokens=120,
            global_instruction=instruction,
            reference=references[:count] or None,
        )
        expected = processor([[message]], mode="generation")
        for name, actual in prepared.data["prism_inputs"].items():
            torch.testing.assert_close(actual, expected[name][0], rtol=0, atol=0)

    service = SpeechRequestValidator(default_model="prism")
    reference_uri = "data:audio/wav;base64,UklGRg=="
    for fields, task in [
        ({}, "TTS"),
        ({"instructions": '{"Speed":"slow"}'}, "Instruction"),
        ({"ref_audio": reference_uri}, "Full VoiceClone"),
        (
            {"ref_audio": reference_uri, "instructions": '{"Emotion":"happy"}'},
            "Instruction Voice Clone",
        ),
    ]:
        speech = service.parse_generation_request(
            {"input": "Hello.", "language": "English", "token_count": 120, **fields}
        )
        request = Client.build_omni_request(
            service.build_generate_request(
                speech.request,
                validate=False,
                reference_descriptors=speech.reference_descriptors,
            )
        )
        prepared = preprocess_prism_payload(
            StagePayload(request_id="speech", request=request, data={}),
            processor=processor,
            reference_encoder=ReferenceEncoder(),
        )
        message = processor.build_user_message(
            script="Hello.",
            task_type=task,
            language="English",
            tokens=120,
            global_instruction=fields.get("instructions"),
            reference=references[:1] if "ref_audio" in fields else None,
        )
        expected = processor([[message]], mode="generation")
        for name, actual in prepared.data["prism_inputs"].items():
            torch.testing.assert_close(actual, expected[name][0], rtol=0, atol=0)
