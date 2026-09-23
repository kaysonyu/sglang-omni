# SPDX-License-Identifier: Apache-2.0
"""Build the SGLang eager AR engine for MOSS-TTS Prism."""

from __future__ import annotations

from functools import partial
from typing import Protocol

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts_prism.model_runner import MossTTSPrismModelRunner
from sglang_omni.models.moss_tts_prism.request_builders import (
    PrismRequestData,
    PrismStreamOutputBuilder,
    apply_prism_result,
    build_prism_request,
)
from sglang_omni.models.moss_tts_prism.sglang_model import MossTTSPrismSGLangModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class PrismResultAdapter(Protocol):
    def __call__(self, data: PrismRequestData) -> StagePayload: ...


class MossTTSPrismEngineBuilder(MossTtsEngineBuilder):
    model_name = "MOSS-TTS Prism"
    model_arch_override = "MossTTSPrismSGLangModel"
    supports_breakable_prefill_cuda_graph = False

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        return {
            **super().generation_defaults(dtype=dtype),
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "chunked_prefill_size": -1,
        }

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        TtsEngineBuilder.validate_before_infrastructure(self, server_args)
        cfg = resolved_view(server_args)
        if cfg.tp_size != 1 or cfg.pp_size != 1:
            raise ValueError("MOSS-TTS Prism currently requires TP=1 and PP=1")
        if (
            cfg.cuda_graph_config.decode.backend != "disabled"
            or cfg.cuda_graph_config.prefill.backend != "disabled"
            or cfg.enable_torch_compile
        ):
            raise ValueError("MOSS-TTS Prism currently requires eager AR execution")
        if not cfg.disable_radix_cache or cfg.chunked_prefill_size > 0:
            raise ValueError(
                "MOSS-TTS Prism currently requires radix cache and chunked prefill disabled"
            )
        if not cfg.disable_overlap_schedule:
            raise ValueError("MOSS-TTS Prism currently requires synchronous scheduling")

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> MossTTSPrismModelRunner:
        return MossTTSPrismModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: MossTTSPrismSGLangModel
    ) -> tuple[partial[PrismRequestData], PrismResultAdapter]:
        return partial(build_prism_request, model=model), apply_prism_result

    def extra_scheduler_kwargs(self) -> dict[str, bool | PrismStreamOutputBuilder]:
        return {
            "enable_async_decode": False,
            "stream_output_builder": PrismStreamOutputBuilder(),
        }

    def make_abort_callback(self) -> None:
        return None
