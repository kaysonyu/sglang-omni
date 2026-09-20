"""Exact-input scoring protocol for discrete Higgs student trajectories."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HiggsScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sample_id: str
    prompt_token_ids: list[int] = Field(min_length=1)
    reference_codes_delayed: list[list[int]] = Field(default_factory=list)
    codes: list[list[int]] = Field(min_length=1)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_rows(self):
        channels = len(self.codes[0])
        if channels < 1 or any(
            len(row) != channels for row in self.codes + self.reference_codes_delayed
        ):
            raise ValueError("Higgs scoring requires aligned codebook rows")
        if self.prompt_token_ids.count(-100) != len(self.reference_codes_delayed):
            raise ValueError(
                "Higgs reference codes must match the original prompt slots"
            )
        if any(token < 0 and token != -100 for token in self.prompt_token_ids):
            raise ValueError("Invalid prompt token")
        if len(self.prompt_token_ids) + len(self.codes) - 1 > 4096:
            raise ValueError(
                "Higgs scoring exceeds the configured 4096-position context"
            )
        return self

    def input_sha256(self):
        return hashlib.sha256(
            json.dumps(
                self.model_dump(exclude={"sample_id"}),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
