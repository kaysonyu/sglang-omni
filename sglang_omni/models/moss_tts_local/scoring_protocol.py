# SPDX-License-Identifier: Apache-2.0
"""Exact supplied-action scoring contract, separate from speech generation."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LocalScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sample_id: str
    prompt_rows: list[list[int]] = Field(min_length=1)
    decisions: list[Literal[0, 1]] = Field(min_length=1)
    codes: list[list[int]]
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_actions(self):
        if any(len(row) != 13 for row in self.prompt_rows):
            raise ValueError("prompt_rows must have 13 channels")
        if any(
            not 0 <= row[0] < 151936 or any(not 0 <= x <= 1024 for x in row[1:])
            for row in self.prompt_rows
        ):
            raise ValueError("prompt token/code outside the mossLite vocabulary")
        if any(
            len(row) != 12 or any(not 0 <= x < 1024 for x in row) for row in self.codes
        ):
            raise ValueError("codes must have 12 channels in [0, 1024)")
        frames = len(self.codes)
        if len(self.decisions) not in (frames, frames + 1):
            raise ValueError(
                "decisions must cover frames and an optional terminal stop"
            )
        if any(self.decisions[:frames]) or (
            len(self.decisions) > frames and self.decisions[-1] != 1
        ):
            raise ValueError(
                "frames require continue decisions; only the final event may stop"
            )
        if len(self.prompt_rows) + frames > 4096:
            raise ValueError("score request exceeds 4096 rows")
        return self

    def input_sha256(self) -> str:
        payload = self.model_dump(exclude={"sample_id"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class LocalScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[LocalScoreInput] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id must be unique within a score batch")
        return self
