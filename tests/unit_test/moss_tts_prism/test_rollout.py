# SPDX-License-Identifier: Apache-2.0
"""Prism rollout actions, stop observations, and graph sampling."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_prism.config import MossTTSPrismPipelineConfig
from sglang_omni.models.moss_tts_prism.model_runner import MossTTSPrismModelRunner
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


def test_identity_hashes_original_artifact_bytes(tmp_path):
    config = {"n_vq": 3, "speech_vocab_size": 17, "sampling_rate": 24000}
    (tmp_path / "config.json").write_text(json.dumps(config))
    for filename in ("modeling_moss_tts.py", "processing_moss_tts.py"):
        (tmp_path / filename).write_text("original\n")
    first = prism_model_identity(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(config, indent=2))
    second = prism_model_identity(str(tmp_path))
    assert first["config_sha256"] != second["config_sha256"]
    assert first["processor_sha256"] == second["processor_sha256"]
    assert first["n_vq"] == 3
    assert first["audio_vocab_size"] == 17
    assert first["sample_rate"] == 24000


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

    for script, count in [
        ("Hello.", 0),
        (
            [
                {"text": "Hello.", "local_instruction": "Speak softly."},
                {"text": "Goodbye."},
            ],
            2,
        ),
    ]:
        inputs = {
            "script": script,
            "global_instruction": "Speak clearly.",
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
            global_instruction="Speak clearly.",
            reference=references[:count] or None,
        )
        expected = processor([[message]], mode="generation")
        for name, actual in prepared.data["prism_inputs"].items():
            torch.testing.assert_close(actual, expected[name][0], rtol=0, atol=0)
