# SPDX-License-Identifier: Apache-2.0
"""Prism schedule, carrier, loader and streaming contracts."""

from types import SimpleNamespace

import pytest
import torch
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
    preprocess_prism_payload,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismBatchInputs,
    PrismModelOutput,
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
    assert output.audio_codes.tolist() == [[0, 0, 0, 0]]


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
    assert output.stop_logits.dtype == torch.float32
    torch.testing.assert_close(output.stop_logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [1, 3, 7])
@pytest.mark.parametrize("stop", [True, False])
def test_stop_frame_and_length_limited_stream_tail_are_kept(frames, stop):
    runner = MossTTSPrismModelRunner.__new__(MossTTSPrismModelRunner)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(
            audio_end_token_id=9, audio_assistant_gen_slot_token_id=8
        )
    )
    data = PrismRequestData(
        prompt=prompt(), stream_metadata={"stream": True, "n_codebooks": 2}
    )
    request = SchedulerRequest(request_id="test", data=data)
    stream = PrismStreamOutputBuilder()
    chunks = []
    for index in range(frames):
        result = SimpleNamespace(
            logits_output=PrismModelOutput(
                next_token_logits=torch.zeros(1, 2),
                audio_codes=torch.tensor([[index, index + 1]]),
                stop_logits=(
                    torch.tensor([[0.0, 1.0]])
                    if stop and index == frames - 1
                    else torch.tensor([[1.0, 0.0]])
                ),
            )
        )
        runner.post_decode(result, None, None, [request])
        output = RequestOutput(request_id="test", data=result.next_token_ids.item())
        runner.post_process_outputs(
            result, SimpleNamespace(requests=[request]), {"test": output}
        )
        chunks.extend(stream("test", data, output))
    chunks.extend(stream.flush("test", data))
    assert result.next_token_ids.item() == (9 if stop else 8)
    torch.testing.assert_close(
        torch.cat([chunk.data[:, 1:] for chunk in chunks]),
        torch.stack(data.output_codes),
    )
    assert len(data.output_codes) == frames
    assert stream.flush("test", data) == []


def test_preprocess_uses_prism_prompt_fields():
    observed = {}

    class Processor:
        def build_user_message(self, **kwargs):
            observed.update(kwargs)
            return kwargs

        def __call__(self, conversations, mode):
            return {name: value[None] for name, value in prompt().items()}

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
    assert prepared.data["prism_inputs"]["input_ids"].shape == (2, 3)
    assert prepared.data["generation_kwargs"]["seed"] == 42


def test_pipeline_registers_separate_architecture_with_eager_ar():
    config = MossTTSPrismPipelineConfig(model_path="model")
    assert config.architecture == "MossTTSPrismModel"
    assert config.stage_named("tts_engine").engine.disable_cuda_graph
    assert config.stage_named("tts_engine").stream_to == ["vocoder"]
    assert config.stage_factory_kwargs("vocoder") == {
        "codec_model_path": config.codec_model_path
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tp_size": 2}, "TP=1"),
        ({"disable_radix_cache": False}, "radix cache"),
        ({"chunked_prefill_size": 512}, "chunked prefill"),
        ({"enable_torch_compile": True}, "eager AR"),
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
    runner.model = SimpleNamespace(config=config, n_vq=2, device=torch.device("cpu"))
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
