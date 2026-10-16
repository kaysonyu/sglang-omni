# SPDX-License-Identifier: Apache-2.0
"""Prism schedule, carrier, loader and streaming contracts."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import nn

from sglang_omni.models.moss_tts_prism.config import MossTTSPrismPipelineConfig
from sglang_omni.models.moss_tts_prism.engine_builder import MossTTSPrismEngineBuilder
from sglang_omni.models.moss_tts_prism.model_runner import (
    MossTTSPrismModelRunner,
    prism_request_inputs,
)
from sglang_omni.models.moss_tts_prism.request_builders import (
    PrismRequestData,
    PrismStreamOutputBuilder,
    build_prism_prompt,
    preprocess_prism_payload,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismBatchInputs,
    PrismDecodeInputs,
    prism_cache_layout,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput, SchedulerRequest


def unit(kind, index, start, stop, heads=()):
    return SimpleNamespace(
        kind=kind, index=index, layer_start=start, layer_stop=stop, rvq_heads=heads
    )


@pytest.mark.parametrize("reapply, expected", [(True, 94.0), (False, 82.0)])
def test_loop_residual_and_conditioning(reapply, expected):
    visited_slots = []

    class DoubleLayer(nn.Module):
        mlp_only = False

        def __init__(self):
            super().__init__()
            self.self_attn = SimpleNamespace(attn=SimpleNamespace(layer_id=None))

        def forward(self, hidden, positions, forward_batch):
            visited_slots.append(self.self_attn.attn.layer_id)
            return hidden * 2

    model = embedding_model()
    model.n_vq = 4
    model.transformer.embed_tokens = nn.Embedding.from_pretrained(torch.ones(2, 2))
    model.transformer.layers = nn.ModuleList([DoubleLayer() for _ in range(3)])
    model.audio_embeddings = nn.ModuleList(
        nn.Embedding.from_pretrained(torch.full((2, 2), value))
        for value in (3.0, 5.0, 7.0, 0.0)
    )
    model.audio_lm_heads = nn.ModuleList(
        [nn.Linear(2, 2, bias=False) for _ in range(4)]
    )
    model.prediction_norms = nn.ModuleList([nn.Identity() for _ in range(4)])
    model.stop_head = nn.Linear(2, 2, bias=False)
    model.topology = SimpleNamespace(
        execution_units=(
            unit("ordinary", 0, 0, 1, (1,)),
            unit("loop", 1, 1, 2, (2,)),
            unit("loop", 1, 1, 2, (3,)),
            unit("ordinary", 2, 2, 3, (4,)),
        ),
        prediction_norm_index_by_execution_unit=(0, 1, 2, 3),
        reapply_rvq_conditioning_in_loop=reapply,
    )
    model.batch_inputs = PrismBatchInputs(
        rows=torch.tensor([[1, 0, 0, 0, 0]]),
        item_kind=torch.zeros(1, dtype=torch.long),
        audio_role=torch.zeros(1, dtype=torch.long),
        retention=torch.ones(1, 4, dtype=torch.bool),
        successor_mask=torch.zeros(1, dtype=torch.bool),
        successor_codes=torch.zeros(1, 4, dtype=torch.long),
        sample_indices=torch.tensor([0]),
        sample=lambda logits, channel: torch.zeros(1, dtype=torch.long),
    )
    predictions = []
    model.prediction_norms[-1].register_forward_hook(
        lambda module, args, output: predictions.append(output.clone())
    )
    output = model(torch.tensor([1]), torch.tensor([0]), SimpleNamespace())
    torch.testing.assert_close(predictions[0], torch.full((1, 2), expected))
    assert visited_slots == [0, 1, 2, 3]
    assert output.customized_info["audio_codes"].tolist() == [[0, 0, 0, 0]]


def test_virtual_cache_slots_exclude_mlp_only_repetitions():
    config = SimpleNamespace(
        prism_mlp_only_layers=(2,),
        prism_topology=SimpleNamespace(
            execution_units=(
                unit("ordinary", 0, 0, 1),
                unit("loop", 1, 1, 3),
                unit("loop", 1, 1, 3),
            )
        ),
    )
    assert prism_cache_layout(config) == ((0, 0), (1, 1), (2, 1))


def prompt():
    return {
        "input_ids": torch.tensor([[3, 0, 0], [7, 0, 0]]),
        "item_kind": torch.zeros(2, dtype=torch.long),
        "audio_role": torch.zeros(2, dtype=torch.long),
        "target_history_retention_mask": torch.ones(2, 2, dtype=torch.bool),
        "successor_audio_mask": torch.zeros(2, dtype=torch.bool),
        "successor_audio_codes": torch.zeros(2, 2, dtype=torch.long),
    }


def test_reprefill_reconstructs_generated_successors_without_mutating_prompt():
    data = PrismRequestData(
        prompt=prompt(), output_codes=[torch.tensor([1, 2]), torch.tensor([3, 4])]
    )
    config = SimpleNamespace(text_pad_idx=0, n_vq=2, prism_input_rvq_channels=(1,))
    full = prism_request_inputs(data, config, prefill=True)
    assert full["input_ids"].tolist() == [[3, 0, 0], [7, 0, 0], [0, 1, 2], [0, 3, 4]]
    assert full["successor_audio_mask"].tolist() == [False, True, True, False]
    assert full["successor_audio_codes"].tolist() == [[0, 0], [1, 2], [3, 4], [0, 0]]
    assert full["target_history_retention_mask"][-2:].tolist() == [[True, False]] * 2
    assert not data.prompt["successor_audio_mask"].any()
    decode = prism_request_inputs(data, config, prefill=False)
    for name, value in decode.items():
        torch.testing.assert_close(value, full[name][-1:])


def embedding_model():
    model = MossTTSPrismSGLangModel.__new__(MossTTSPrismSGLangModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_pad_idx=0, prism_input_rvq_channels=(1,), embedding_head_usage="untied"
    )
    model.decode_inputs = None
    model.model = None
    model.n_vq = 2
    model.transformer = nn.Module()
    model.transformer.embed_tokens = nn.Embedding.from_pretrained(
        torch.tensor([[9.0, 9.0], [1.0, 2.0]])
    )
    model.audio_embeddings = nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.tensor([[3.0, 4.0], [5.0, 6.0]])),
            nn.Embedding.from_pretrained(torch.tensor([[7.0, 8.0], [9.0, 10.0]])),
        ]
    )
    model.audio_lm_heads = nn.ModuleList(
        [nn.Linear(2, 2, bias=False) for _ in range(2)]
    )
    return model


def test_typed_embedding_preserves_reference_channels_and_active_code_zero():
    model = embedding_model()
    inputs = PrismBatchInputs(
        rows=torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        item_kind=torch.tensor([0, 1, 1, 1]),
        audio_role=torch.tensor([0, 1, 2, 2]),
        retention=torch.tensor(
            [[True, True], [True, True], [True, True], [False, True]]
        ),
        successor_mask=torch.zeros(4, dtype=torch.bool),
        successor_codes=torch.zeros(4, 2, dtype=torch.long),
        sample_indices=torch.tensor([3]),
        sample=lambda logits, channel: logits.argmax(-1),
    )
    torch.testing.assert_close(
        model.prepare_inputs(inputs),
        torch.tensor([[1.0, 2.0], [12.0, 14.0], [3.0, 4.0], [0.0, 0.0]]),
    )


def test_loading_independent_heads_and_reloading_preserves_storage():
    model = embedding_model()
    embedding = model.audio_embeddings[0].weight
    head = model.audio_lm_heads[0].weight
    addresses = embedding.data_ptr(), head.data_ptr()
    model.load_weights(
        [
            ("audio_embeddings.0.weight", torch.ones(2, 2)),
            ("audio_lm_heads.0.weight", torch.full((2, 2), 3.0)),
        ]
    )
    model.load_weights([("audio_lm_heads.0.weight", torch.full((2, 2), 4.0))])
    assert (embedding.data_ptr(), head.data_ptr()) == addresses
    assert torch.all(embedding == 1)
    assert torch.all(head == 4)


def test_stop_projection_stays_fp32_under_autocast():
    class Layer(nn.Module):
        mlp_only = True

        def forward(self, hidden, positions, forward_batch):
            return hidden

    model = embedding_model().to(torch.bfloat16)
    model.transformer.layers = nn.ModuleList([Layer()])
    model.prediction_norms = nn.ModuleList([nn.Identity()])
    model.stop_head = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.stop_head.weight.copy_(torch.tensor([[0.1, 0.3], [0.7, 0.2]]))
    model.topology = SimpleNamespace(
        execution_units=(unit("ordinary", 0, 0, 1, (1, 2)),),
        prediction_norm_index_by_execution_unit=(0,),
    )
    model.batch_inputs = PrismBatchInputs(
        rows=torch.tensor([[1, 0, 0]]),
        item_kind=torch.tensor([0]),
        audio_role=torch.tensor([0]),
        retention=torch.ones(1, 2, dtype=torch.bool),
        successor_mask=torch.zeros(1, dtype=torch.bool),
        successor_codes=torch.zeros(1, 2, dtype=torch.long),
        sample_indices=torch.tensor([0]),
        sample=lambda logits, channel: logits.argmax(-1),
    )
    expected = torch.nn.functional.linear(
        model.transformer.embed_tokens.weight[1:2].float(),
        model.stop_head.weight.float(),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.tensor([1]), torch.tensor([0]), SimpleNamespace())
    assert output.next_token_logits.dtype == torch.float32
    torch.testing.assert_close(output.next_token_logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [1, 3, 7])
@pytest.mark.parametrize("stop", [True, False])
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
def test_stop_step_is_excluded_but_length_limited_frame_is_kept(frames, stop, device):
    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=False,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    data = PrismRequestData(
        prompt=prompt(),
        stream_metadata={"stream": True, "n_codebooks": 2},
        req=SimpleNamespace(sampling_params=SimpleNamespace(stop_token_ids=[9])),
    )
    request = SchedulerRequest(request_id="test", data=data)
    stream = PrismStreamOutputBuilder()
    chunks = []
    for index in range(frames):
        result = SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                customized_info={
                    "audio_codes": torch.tensor([[index, index + 1]], device=device)
                },
                next_token_logits=torch.tensor(
                    [[0.0, 1.0]] if stop and index == frames - 1 else [[4.0, 0.0]],
                    device=device,
                ),
            )
        )
        runner.post_decode(result, None, None, [request])
        host_ids = runner.resolve_host_token_ids(result)
        assert host_ids.device.type == "cpu"
        output = RequestOutput(request_id="test", data=host_ids.item())
        runner.post_process_outputs(
            result, SimpleNamespace(requests=[request]), {"test": output}
        )
        chunks.extend(stream("test", data, output))
    chunks.extend(stream.flush("test", data))
    assert result.next_token_ids.item() == (9 if stop else 8)
    expected_frames = frames - int(stop)
    assert len(data.output_codes) == expected_frames
    if expected_frames:
        torch.testing.assert_close(
            torch.cat([chunk.data[:, 1:] for chunk in chunks]),
            torch.stack(data.output_codes),
        )
    else:
        assert chunks == []
        from sglang_omni.models.moss_tts_prism.request_builders import (
            apply_prism_result,
        )

        with pytest.raises(RuntimeError, match="generated no audio frames"):
            apply_prism_result(data)
    assert stream.flush("test", data) == []


def test_preprocess_uses_prism_prompt_fields():
    observed = {}

    class Processor:
        model_config = SimpleNamespace(
            n_vq=2, audio_start_token_id=5, audio_user_slot_token_id=6, text_pad_idx=0
        )
        chat_template = "template"
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
            encode=lambda content, **kwargs: [3, 7],
        )

        def build_user_message(self, **kwargs):
            observed.update(kwargs)
            return {"content": kwargs["script"], "audio_codes_list": []}

        def _normalize_user(self, message):
            return message

        def _replace_placeholders(self, content, references, *, role):
            return content

    payload = StagePayload(
        request_id="test",
        request=OmniRequest(
            inputs="hello",
            params={"instructions": "Speak softly", "max_new_tokens": 12, "seed": 42},
        ),
        data={},
    )
    prepared = preprocess_prism_payload(
        payload, processor=Processor(), reference_encoder=None
    )
    assert observed == {
        "script": "hello",
        "reference": None,
        "global_instruction": "Speak softly",
        "tokens": None,
    }
    assert prepared.data["prism_inputs"]["input_ids"].tolist() == [
        [3, 0, 0],
        [7, 0, 0],
        [5, 0, 0],
    ]
    assert prepared.data["generation_kwargs"]["seed"] == 42


def test_prompt_reference_frames_and_successor_edges():
    codes = torch.tensor([[0, 1], [2, 3], [4, 0]])
    message = {"content": "reference", "audio_codes_list": [codes]}
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            n_vq=2, audio_start_token_id=5, audio_user_slot_token_id=6, text_pad_idx=9
        ),
        chat_template="template",
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
            encode=lambda content, **kwargs: [11, 5, 6, 6, 6, 7, 12],
        ),
        _normalize_user=lambda value: value,
        _replace_placeholders=lambda content, refs, **kwargs: content,
    )
    prompt = build_prism_prompt(processor, message)
    assert prompt["input_ids"].tolist() == [
        [11, 0, 0],
        [5, 0, 0],
        [9, 0, 1],
        [9, 2, 3],
        [9, 4, 0],
        [7, 0, 0],
        [12, 0, 0],
        [5, 0, 0],
    ]
    assert prompt["item_kind"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    torch.testing.assert_close(prompt["audio_role"], prompt["item_kind"])
    assert prompt["successor_audio_mask"].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    torch.testing.assert_close(prompt["successor_audio_codes"][1:4], codes)
    assert not prompt["successor_audio_codes"][[0, 4, 5, 6, 7]].any()
    assert prompt["target_history_retention_mask"].all()
    assert prompt["attention_mask"].all()
    assert prompt["position_ids"].tolist() == list(range(8))


@pytest.mark.parametrize("vocoder_cuda_graph", [None, False, True])
def test_pipeline_registers_separate_architecture_with_decode_graphs(
    vocoder_cuda_graph,
):
    config = MossTTSPrismPipelineConfig(
        model_path="model", vocoder_cuda_graph=vocoder_cuda_graph
    )
    assert config.architecture == "MossTTSPrismModel"
    assert not config.stage_named("tts_engine").engine.disable_cuda_graph
    assert (
        config.stage_named("tts_engine").engine.cuda_graph_backend_prefill
        == "breakable"
    )
    assert config.stage_named("tts_engine").stream_to == ["vocoder"]
    assert config.stage_factory_kwargs("vocoder") == {
        "codec_model_path": config.codec_model_path,
        "vocoder_cuda_graph": vocoder_cuda_graph,
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tp_size": 2}, "TP=1"),
        ({"disable_radix_cache": False}, "radix cache"),
        ({"chunked_prefill_size": 512}, "chunked prefill"),
        ({"enable_torch_compile": True}, "without torch.compile"),
        ({"disable_overlap_schedule": False}, "synchronous"),
    ],
)
def test_unsupported_engine_modes_fail_before_loading(monkeypatch, overrides, message):
    from sglang_omni.models.moss_tts_prism import engine_builder

    config = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        enable_torch_compile=False,
        disable_overlap_schedule=True,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="disabled"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    monkeypatch.setattr(engine_builder, "resolved_view", lambda args: config)
    with pytest.raises(ValueError, match=message):
        MossTTSPrismEngineBuilder().validate_before_infrastructure(config)


def test_sampling_tracks_request_seed_and_history_after_batch_reordering():
    config = SimpleNamespace(text_pad_idx=0, n_vq=2, prism_input_rvq_channels=(1,))
    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner.model = SimpleNamespace(
        enable_rl=False,
        config=config,
        n_vq=2,
        device=torch.device("cpu"),
        decode_inputs=None,
    )
    requests = []
    for index, seed in enumerate((17, 42)):
        data = PrismRequestData(prompt=prompt(), sampling_seed=seed)
        data.output_codes = [torch.tensor([1, 2])] * (index + 1)
        data.state.generation_kwargs = {
            "audio_temperature": 1.7,
            "audio_top_p": 0.8,
            "audio_top_k": 5,
            "audio_repetition_penalty": 1.2,
        }
        requests.append(SchedulerRequest(request_id=str(index), data=data))
    logits = torch.tensor([[1.0, -1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, -2.0, 0.0]])
    runner.prepare(requests, prefill=False)
    first = runner.model.batch_inputs.sample(logits.clone(), 0)
    runner.prepare(requests[::-1], prefill=False)
    second = runner.model.batch_inputs.sample(logits.flip(0).clone(), 0)
    torch.testing.assert_close(first, second.flip(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_graph_refreshes_seed_history_and_batch_rows():
    device = torch.device("cuda")
    config = SimpleNamespace(
        text_pad_idx=0, n_vq=2, prism_input_rvq_channels=(1,), speech_vocab_size=1024
    )
    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    buffers = PrismDecodeInputs(config, 4, device)
    runner.model = SimpleNamespace(
        enable_rl=False, config=config, n_vq=2, device=device, decode_inputs=buffers
    )
    requests = []
    for index in range(3):
        data = PrismRequestData(prompt=prompt(), sampling_seed=42 + index)
        data.output_codes = [torch.tensor([1 + index, 2 + index])]
        data.state.generation_kwargs = {
            "audio_temperature": (1.7, 0.0, 0.8)[index],
            "audio_top_p": (0.8, 1.0, 0.9)[index],
            "audio_top_k": (25, 0, 50)[index],
            "audio_repetition_penalty": (1.0, 1.2, 0.9)[index],
        }
        requests.append(SchedulerRequest(request_id=str(index), data=data))
    logits = torch.randn(4, 1024, device=device)
    # note (Zhang Yiyang): Capture padded rows before any real request is staged.
    inputs = buffers.for_batch(4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            inputs.sample(logits, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled = inputs.sample(logits, 0)
        captured_rows = inputs.rows.clone()

    for active in (requests, requests[::-1], requests[1:2]):
        runner.model.decode_inputs = None
        runner.prepare(active, prefill=False)
        expected = runner.model.batch_inputs.sample(logits[: len(active)].clone(), 0)
        runner.model.decode_inputs = buffers
        runner.prepare(active, prefill=False)
        graph.replay()
        torch.testing.assert_close(sampled[: len(active)], expected, rtol=0, atol=0)
        assert captured_rows[: len(active), 0].tolist() == [config.text_pad_idx] * len(
            active
        )
        torch.testing.assert_close(
            captured_rows[: len(active), 1:].cpu(),
            torch.stack([request.data.output_codes[-1] for request in active]),
        )
        for request in active:
            request.data.output_codes.append(torch.tensor([3, 4]))
            request.data.sampling_seed += 7
            request.data.state.generation_kwargs["audio_repetition_penalty"] = 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows, hidden", [(1, 2048), (16, 2048), (125, 257)])
def test_fused_embeddings_match_eager_and_graph_replay(dtype, rows, hidden):
    from sglang_omni.models.moss_tts_prism.embedding_kernels import (
        feedback_embeddings,
        input_embeddings,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    weights = tuple(
        torch.randn(32, hidden, device=device, dtype=dtype, generator=generator)
        for _ in range(24)
    )
    text_weight = torch.randn(
        32, hidden, device=device, dtype=dtype, generator=generator
    )
    inputs = PrismBatchInputs(
        rows=torch.randint(0, 32, (rows, 25), device=device, generator=generator),
        item_kind=torch.arange(rows, device=device) % 2,
        audio_role=torch.arange(rows, device=device) % 3,
        retention=torch.rand(rows, 24, device=device, generator=generator) > 0.5,
        successor_mask=torch.arange(rows, device=device) % 2 == 0,
        successor_codes=torch.randint(
            0, 32, (rows, 24), device=device, generator=generator
        ),
        sample_indices=torch.tensor([rows - 1], device=device),
        sample=lambda logits, channel: logits.argmax(-1),
    )
    channels = (1, 3, 7)
    feedback_channels = (0,) if rows == 1 else (0, 23)

    def eager():
        text = inputs.item_kind == 0
        active = text & (inputs.rows[:, 0] != 0)
        value = text_weight[inputs.rows[:, 0].masked_fill(~active, 0)] * active[:, None]
        for channel, weight in enumerate(weights):
            active = ~text & (
                (inputs.audio_role != 2)
                | ((channel + 1 in channels) & inputs.retention[:, channel])
            )
            value = (
                value
                + weight[inputs.rows[:, channel + 1].masked_fill(~active, 0)]
                * active[:, None]
            )
        delta = None
        for channel in feedback_channels:
            embedding = (
                weights[channel][
                    inputs.successor_codes[:, channel].masked_fill(
                        ~inputs.successor_mask, 0
                    )
                ]
                * inputs.successor_mask[:, None]
            )
            delta = embedding if delta is None else delta + embedding
        return value, delta

    def fused():
        return (
            input_embeddings(
                inputs.rows,
                inputs.item_kind,
                inputs.audio_role,
                inputs.retention,
                text_weight,
                weights,
                0,
                channels,
            ),
            feedback_embeddings(
                inputs.successor_codes,
                inputs.successor_mask,
                tuple(weights[channel] for channel in feedback_channels),
                feedback_channels,
            ),
        )

    for actual, expected in zip(fused(), eager()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fused()
    inputs.rows.add_(1).remainder_(32)
    inputs.successor_codes.add_(3).remainder_(32)
    inputs.successor_mask.logical_not_()
    inputs.retention.logical_not_()
    inputs.item_kind.copy_(1 - inputs.item_kind)
    inputs.audio_role.add_(1).remainder_(3)
    text_weight.mul_(0.5)
    weights[0].mul_(0.5)
    weights[-1].add_(0.25)
    graph.replay()
    for actual, expected in zip(outputs, eager()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.accelerator
def test_prefill_graph_replays_live_requests(monkeypatch):
    import os

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.common import release_kv_cache
    from transformers import AutoConfig, AutoTokenizer

    from sglang_omni.models.moss_tts.hf_loading import (
        load_moss_processor_class,
        moss_transformers_processor_compat,
    )
    from sglang_omni.models.moss_tts_local.request_builders import (
        build_moss_tts_local_state,
    )
    from sglang_omni.models.moss_tts_prism.request_builders import build_prism_request
    from sglang_omni.scheduling.types import SchedulerOutput

    model_path = os.environ.get("MOSS_TTS_PRISM_TEST_MODEL")
    if not model_path or not torch.cuda.is_available():
        pytest.skip("CUDA and MOSS_TTS_PRISM_TEST_MODEL are required")
    scheduler = MossTTSPrismEngineBuilder(enable_rl=True).build(
        model_path,
        device="cuda",
        gpu_id=0,
        server_args_overrides={
            "max_total_tokens": 8192,
            "context_length": 4096,
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [64, 128, 256],
        },
    )
    runner = scheduler._model_runner
    native = runner.tp_worker.model_runner
    model = runner.model
    graph = native.prefill_cuda_graph_runner
    assert graph is not None
    with moss_transformers_processor_compat():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = load_moss_processor_class(model_path)(
            tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
            audio_tokenizer=None,
            model_config=config,
        )
    prompts = []
    for length in (7, 11, 19):
        codes = torch.arange(length * config.n_vq).reshape(length, config.n_vq)
        prompts.append(
            build_prism_prompt(
                processor,
                processor.build_user_message(
                    script="The train crossed the river as the city came to life.",
                    reference=[codes % config.speech_vocab_size],
                ),
            )
        )
    capacity = (
        scheduler.req_to_token_pool.available_size(),
        scheduler.token_to_kv_pool_allocator.available_size(),
    )
    observed = {}
    post_prefill = runner.post_prefill

    def collect(result, forward_batch, schedule_batch, requests):
        observed["logprobs"] = result.logits_output.customized_info[
            "code_logprobs"
        ].clone()
        observed["codes"] = result.logits_output.customized_info["audio_codes"].clone()
        observed["logits"] = result.logits_output.next_token_logits.clone()
        observed["graph"] = result.can_run_cuda_graph
        post_prefill(result, forward_batch, schedule_batch, requests)

    monkeypatch.setattr(runner, "post_prefill", collect)

    def padded_eager(shape_key, batch, **kwargs):
        return type(model.model).forward(
            model.model,
            batch.input_ids,
            batch.positions,
            batch,
            input_embeds=batch.input_embeds,
        )

    def run(items, seed):
        requests = []
        for index, prompt in enumerate(items):
            payload = StagePayload(
                request_id=f"prefill-{index}",
                request=OmniRequest(
                    inputs="test",
                    params={
                        "seed": seed + index,
                        "audio_top_k": -1,
                        "return_logprob": True,
                        "return_omni_rollout": True,
                        "audio_top_p": 1.0,
                        "audio_temperature": 0.7 + index * 0.2,
                    },
                ),
                data={},
            )
            state = build_moss_tts_local_state(payload)
            payload.data = {**state.to_dict(), "prism_inputs": prompt}
            data = build_prism_request(payload, model=model)
            scheduler.normalize_req_token_arrays(data.req)
            data.req.init_next_round_input(scheduler.tree_cache)
            data.req.set_extend_range(0, len(prompt["input_ids"]))
            requests.append(SchedulerRequest(request_id=payload.request_id, data=data))
        batch = ScheduleBatch.init_new(
            [request.data.req for request in requests],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            enable_overlap=False,
            spec_algorithm=scheduler.spec_algorithm,
        )
        try:
            batch.prepare_for_extend()
            runner.execute(SchedulerOutput(requests=requests, batch_data=batch))
            result = dict(observed)
            result["kv"] = [
                buffer[batch.out_cache_loc].clone()
                for layer in range(model.end_layer)
                for buffer in (
                    native.token_to_kv_pool.get_key_buffer(layer),
                    native.token_to_kv_pool.get_value_buffer(layer),
                )
            ]
            assert model.batch_inputs is None
            assert model.model.inputs is model.model.capture_inputs
            return result
        finally:
            for request in requests:
                if request.data.req.kv.holds_kv:
                    release_kv_cache(request.data.req, scheduler.tree_cache)
            assert capacity == (
                scheduler.req_to_token_pool.available_size(),
                scheduler.token_to_kv_pool_allocator.available_size(),
            )

    try:
        for items, seed in (
            (prompts[:1], 42),
            (prompts, 17),
            (prompts[::-1][:2], 71),
            (prompts[:1], 99),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(graph.backend, "replay", padded_eager)
                expected = run(items, seed)
            actual = run(items, seed)
            assert actual["graph"]
            for key in ("codes", "logits", "logprobs"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for actual_kv, expected_kv in zip(actual["kv"], expected["kv"]):
                torch.testing.assert_close(actual_kv, expected_kv, rtol=0, atol=0)
        long_batch = prompts * 3
        with monkeypatch.context() as patch:
            patch.setattr(native, "prefill_cuda_graph_runner", None)
            expected = run(long_batch, 42)
        actual = run(long_batch, 42)
        assert not actual["graph"]
        for key in ("codes", "logits", "logprobs"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


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
def test_stop_probability_threshold_and_mixed_batch(monkeypatch, device):
    from sglang_omni.models.moss_tts_prism import model_runner

    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    runner.model = SimpleNamespace(
        enable_rl=False,
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        ),
    )
    logits = torch.tensor([[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)
    result = SimpleNamespace(
        logits_output=LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={
                "audio_codes": torch.arange(6, device=device).reshape(3, 2)
            },
        )
    )
    requests = [
        SchedulerRequest(request_id=str(i), data=PrismRequestData()) for i in range(3)
    ]
    runner.post_decode(result, None, None, requests)
    ids = runner.resolve_host_token_ids(result).tolist()
    assert ids == [8, 9, 9]
    outputs = {
        str(i): RequestOutput(request_id=str(i), data=token)
        for i, token in enumerate(ids)
    }
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
    assert [len(request.data.output_codes) for request in requests] == [1, 0, 0]
    monkeypatch.setattr(
        model_runner, "PRISM_STOP_THRESHOLD", logits.float().softmax(-1)[1, 1].item()
    )
    runner.post_decode(result, None, None, requests)
    assert runner.resolve_host_token_ids(result).tolist() == [8, 8, 9]
