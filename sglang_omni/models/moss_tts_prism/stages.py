# SPDX-License-Identifier: Apache-2.0
"""Prism AR factories reusing MOSS-Audio-Tokenizer v2."""

from __future__ import annotations

from functools import partial

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_tokenizer import (
    load_moss_audio_encoder,
    load_moss_audio_vocoder,
)
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts_local.config import resolve_vocoder_cuda_graph
from sglang_omni.models.moss_tts_local.stages import (
    BatchedReferenceEncoder,
    MossLocalReferenceEncoder,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.models.moss_tts_prism.engine_builder import MossTTSPrismEngineBuilder
from sglang_omni.models.moss_tts_prism.request_builders import preprocess_prism_payload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def create_preprocessing_executor(
    model_path: str,
    *,
    codec_model_path: str,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    device = str(resolve_concrete_device(device, gpu_id))
    with moss_transformers_processor_compat():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = load_moss_processor_class(model_path)(
            tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
            audio_tokenizer=None,
            model_config=config,
        )
    encoder = load_moss_audio_encoder(
        codec_model_path,
        device=device,
        compute_dtype=torch.bfloat16,
        attention_backend="auto",
    )
    reference = MossLocalReferenceEncoder(
        BatchedReferenceEncoder(encoder, n_vq=config.n_vq), n_vq=config.n_vq
    )
    return SimpleScheduler(
        partial(
            preprocess_prism_payload, processor=processor, reference_encoder=reference
        ),
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: (
        dict[str, str | int | float | bool | list[int]] | None
    ) = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    return MossTTSPrismEngineBuilder(
        total_gpu_memory_fraction=total_gpu_memory_fraction
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    codec_model_path: str,
    device: str | None = None,
    gpu_id: int | None = None,
    stream_slots: int = 16,
    stream_chunk_frames: int = 25,
    initial_chunk_frames: int = 5,
    max_batch_size: int = 8,
    vocoder_cuda_graph: bool | None = None,
    vocoder_cuda_graph_frames: list[int] | None = None,
    vocoder_cuda_graph_min_free_gb: float = 3.0,
) -> MossTTSLocalStreamingVocoderScheduler:
    vocoder_cuda_graph = resolve_vocoder_cuda_graph(vocoder_cuda_graph)
    device = str(resolve_concrete_device(device, gpu_id))
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocoder = load_moss_audio_vocoder(
        codec_model_path,
        device=device,
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
        attention_backend="auto",
    )
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        vocoder.model,
        n_vq=config.n_vq,
        sample_rate=vocoder.sample_rate,
        attention_backend="auto",
        stream_slots=stream_slots,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        vocoder_cuda_graph=vocoder_cuda_graph,
        vocoder_cuda_graph_frames=vocoder_cuda_graph_frames,
        vocoder_cuda_graph_min_free_gb=vocoder_cuda_graph_min_free_gb,
    )
    scheduler.warmup_now()
    return scheduler
