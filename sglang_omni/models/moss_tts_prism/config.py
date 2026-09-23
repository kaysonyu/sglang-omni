# SPDX-License-Identifier: Apache-2.0
"""Single-GPU MOSS-TTS Prism pipeline configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

PKG = "sglang_omni.models.moss_tts_prism"


def stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            gpu=0,
            factory_path=f"{PKG}.stages.create_preprocessing_executor",
            gpu_memory_fraction=0.15,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            gpu=0,
            factory_path=f"{PKG}.stages.create_sglang_tts_engine_executor",
            engine=EngineArgs(disable_cuda_graph=False),
            gpu_memory_fraction=0.67,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="vocoder",
            gpu=0,
            factory_path=f"{PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(),
            gpu_memory_fraction=0.18,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSPrismPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MossTTSPrismModel"
    requires_model_capabilities: ClassVar[bool] = True
    max_speech_input_chars: ClassVar[int | None] = None
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig
    }
    stages: list[StageConfig] = Field(default_factory=stages)
    codec_model_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: {"OMP_NUM_THREADS": "4"}
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str]:
        if stage_name in ("preprocessing", "vocoder"):
            return {"codec_model_path": self.codec_model_path}
        return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPrismPipelineConfig
