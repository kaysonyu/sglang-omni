# SPDX-License-Identifier: Apache-2.0
"""Build the SGLang AR engine for MOSS-TTS Prism."""

from __future__ import annotations

from functools import partial
from typing import Protocol

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_serving
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
from sglang_omni.models.moss_tts_prism.rollout_trace import prism_model_identity
from sglang_omni.models.moss_tts_prism.scoring import (
    PrismScoreModelRunner,
    PrismScoreRequestData,
    PrismScoreResultAdapter,
    PrismTeacherWeightChecker,
    build_score_request,
    score_result,
)
from sglang_omni.models.moss_tts_prism.sglang_model import (
    MossTTSPrismSGLangModel,
    PrismDecodeInputs,
    PrismPrefillBody,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class PrismResultAdapter(Protocol):
    def __call__(self, data: PrismRequestData) -> StagePayload: ...


class MossTTSPrismEngineBuilder(MossTtsEngineBuilder):
    model_name = "MOSS-TTS Prism"
    model_arch_override = "MossTTSPrismSGLangModel"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self, *, total_gpu_memory_fraction: float | None = None, enable_rl: bool = False
    ) -> None:
        super().__init__(total_gpu_memory_fraction=total_gpu_memory_fraction)
        self.enable_rl = enable_rl

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        return {
            **super().generation_defaults(dtype=dtype),
            "disable_radix_cache": True,
            "chunked_prefill_size": -1,
        }

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        TtsEngineBuilder.validate_before_infrastructure(self, server_args)
        cfg = resolved_view(server_args)
        if cfg.tp_size != 1 or cfg.pp_size != 1:
            raise ValueError("MOSS-TTS Prism currently requires TP=1 and PP=1")
        if (
            cfg.cuda_graph_config.decode.backend not in ("disabled", "full")
            or cfg.cuda_graph_config.prefill.backend not in ("disabled", "breakable")
            or cfg.enable_torch_compile
        ):
            raise ValueError(
                "MOSS-TTS Prism supports full decode graphs and eager or breakable prefill, without torch.compile"
            )
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

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        cfg = resolved_view(server_args)
        model = model_worker.model_runner.model
        model.enable_rl = self.enable_rl
        if self.enable_rl:
            model.model_identity = prism_model_identity(checkpoint_dir)
        if cfg.cuda_graph_config.prefill.backend == "breakable":
            model.model = PrismPrefillBody(model, max(cfg.cuda_graph_config.prefill.bs))
        if cfg.cuda_graph_config.decode.backend != "disabled":
            model.decode_inputs = PrismDecodeInputs(
                model.config,
                max(get_decode_cuda_graph_bs(server_args)),
                model.device,
                enable_rl=self.enable_rl,
            )

    def post_cuda_graph_setup(
        self, model: MossTTSPrismSGLangModel, server_args: ServerArgs
    ) -> None:
        # note (Zhang Yiyang): Prism samples inside the AR graph itself.
        pass

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


class PrismScoreEngineBuilder(MossTTSPrismEngineBuilder):
    supports_breakable_prefill_cuda_graph = False

    def __init__(
        self, *, score_chunk_size: int, total_gpu_memory_fraction: float | None = None
    ) -> None:
        super().__init__(total_gpu_memory_fraction=total_gpu_memory_fraction)
        if score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive")
        self.score_chunk_size = score_chunk_size

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        return {**super().generation_defaults(dtype=dtype), "disable_cuda_graph": True}

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        super().validate_before_infrastructure(server_args)
        config = resolved_view(server_args)
        if (
            config.cuda_graph_config.decode.backend != "disabled"
            or config.cuda_graph_config.prefill.backend != "disabled"
        ):
            raise ValueError("Prism scoring requires disabled CUDA graphs")

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        model = model_worker.model_runner.model
        model.model_identity = prism_model_identity(checkpoint_dir)
        checker = PrismTeacherWeightChecker(model_worker.model_runner)
        model_worker._strict_weight_checker = checker
        model.teacher_weight_sha256 = checker.checksum()["per_gpu_checksum"]
        model.teacher_weight_version = str(get_serving().weight_version)

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> PrismScoreModelRunner:
        return PrismScoreModelRunner(
            model_worker, output_proc, score_chunk_size=self.score_chunk_size
        )

    def make_adapters(
        self, model: MossTTSPrismSGLangModel
    ) -> tuple[partial[PrismScoreRequestData], PrismScoreResultAdapter]:
        return (
            partial(
                build_score_request, model=model, context_length=self.context_length
            ),
            score_result,
        )

    def extra_scheduler_kwargs(self) -> dict[str, bool]:
        return {"enable_async_decode": False}


def create_score_engine(
    model_path: str,
    *,
    score_chunk_size: int,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: (
        dict[str, str | int | float | bool | list[int]] | None
    ) = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    return PrismScoreEngineBuilder(
        score_chunk_size=score_chunk_size,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
