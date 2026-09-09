# SPDX-License-Identifier: Apache-2.0
"""mossLite MOSS-TTS Local pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


def moss_tts_local_special_token_defaults(
    audio_vocab_size: int = 1024,
) -> tuple[tuple[str, int], ...]:
    """Fixed MossFlux v2 token contract used by mossLite Local training."""
    return (
        ("audio_start_token_id", 151652),
        ("audio_end_token_id", 151653),
        ("audio_user_slot_token_id", 151654),
        ("audio_assistant_slot_token_id", 151656),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_pad_token_id", int(audio_vocab_size)),
        ("audio_pad_code", int(audio_vocab_size)),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    )


@dataclass
class MossTTSLocalState(DeclarativeStateBase):
    """Per-request state for MOSS-TTS Local generation."""

    sample_rate: int = wire(48000, codec="int_or")
    text: str = wire("", codec="str")
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    token_count: int | None = wire(None, codec="opt_int")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    return_logprob: bool = wire(False, emit="truthy", codec="bool")
    return_omni_rollout: bool = wire(False, emit="truthy", codec="bool")
    omni_rollout: dict[str, Any] | None = None
    finish_reason: str | None = None
    weight_version: str | None = None
    audio_codes: Any | None = wire(None, codec="tensor_cpu")
