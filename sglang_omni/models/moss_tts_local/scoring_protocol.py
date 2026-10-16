# SPDX-License-Identifier: Apache-2.0
"""Supplied-action scoring inputs for MOSS-TTS Local."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class LocalScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sample_id: str
    prompt_rows: list[list[StrictInt]] = Field(min_length=1)
    decisions: list[Literal[0, 1]] = Field(min_length=1)
    codes: list[list[StrictInt]]
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_actions(self) -> LocalScoreInput:
        frames = len(self.codes)
        if len(self.decisions) not in (frames, frames + 1):
            raise ValueError(
                "decisions must cover frames and an optional terminal stop"
            )
        if any(self.decisions[:frames]) or (
            len(self.decisions) > frames and self.decisions[-1] != 1
        ):
            raise ValueError("Only the final decision may stop, without an audio frame")
        return self

    def validate_model(self, config: Any, context_length: int) -> None:
        channels = int(config.n_vq)
        if any(len(row) != channels + 1 for row in self.prompt_rows):
            raise ValueError(f"prompt_rows must have {channels + 1} channels")
        if any(
            not 0 <= row[0] < int(config.vocab_size_list[0])
            or any(not 0 <= code <= int(config.audio_pad_code) for code in row[1:])
            for row in self.prompt_rows
        ):
            raise ValueError("Prompt tokens are outside the model vocabulary")
        if any(
            len(row) != channels
            or any(not 0 <= code < int(config.audio_vocab_size) for code in row)
            for row in self.codes
        ):
            raise ValueError(f"codes must have {channels} valid audio tokens per frame")
        if len(self.prompt_rows) + len(self.codes) > context_length:
            raise ValueError(
                f"Score input exceeds the model context of {context_length} rows"
            )

    def input_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(exclude={"sample_id"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class LocalScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[LocalScoreInput] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_ids(self) -> LocalScoreBatch:
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id must be unique within a score request")
        return self
