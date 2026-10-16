# SPDX-License-Identifier: Apache-2.0
"""Typed replay contract for frozen Prism teacher scores."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from transformers import PretrainedConfig


class PrismScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    layout: Literal["moss_prism_typed_v1"]
    prompt_rows: list[list[StrictInt]] = Field(min_length=1)
    item_kind: list[Literal[0, 1]]
    audio_role: list[Literal[0, 1]]
    successor_audio_mask: list[StrictBool]
    successor_audio_codes: list[list[StrictInt]]
    target_history_retention_mask: list[list[StrictBool]]
    codes: list[list[StrictInt]]
    finish_reason: Literal["stop", "length"]
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_carrier(self) -> PrismScoreInput:
        rows = len(self.prompt_rows)
        channels = len(self.prompt_rows[0]) - 1
        if channels < 1 or any(len(row) != channels + 1 for row in self.prompt_rows):
            raise ValueError("prompt_rows must have a consistent text and audio width")
        for values in (
            self.item_kind,
            self.audio_role,
            self.successor_audio_mask,
            self.successor_audio_codes,
            self.target_history_retention_mask,
        ):
            if len(values) != rows:
                raise ValueError("Typed fields must align with prompt_rows")
        for values in (
            self.successor_audio_codes,
            self.target_history_retention_mask,
            self.codes,
        ):
            if any(len(row) != channels for row in values):
                raise ValueError("Audio fields must contain complete RVQ frames")
        if not self.codes and self.finish_reason != "stop":
            raise ValueError("A zero-frame trajectory must end with stop")
        if self.item_kind != self.audio_role:
            raise ValueError("The prompt must contain only text and reference audio")
        return self

    def validate_model(self, config: PretrainedConfig, context_length: int) -> None:
        if len(self.prompt_rows[0]) != config.n_vq + 1:
            raise ValueError(f"prompt_rows must have {config.n_vq + 1} channels")
        for row in self.prompt_rows:
            if not 0 <= row[0] < config.language_config.vocab_size:
                raise ValueError("Prompt text token is outside the model vocabulary")
        for row in [
            *[row[1:] for row in self.prompt_rows],
            *self.successor_audio_codes,
            *self.codes,
        ]:
            if any(not 0 <= code < config.speech_vocab_size for code in row):
                raise ValueError("Audio code is outside the model vocabulary")
        last = self.prompt_rows[-1]
        if (
            last[0] != config.audio_start_token_id
            or any(last[1:])
            or self.item_kind[-1]
            or self.successor_audio_mask[-1]
            or any(self.successor_audio_codes[-1])
        ):
            raise ValueError("Prompt must end in an unconditioned target audio_start")
        for index, row in enumerate(self.prompt_rows):
            audio = bool(self.item_kind[index])
            if (audio and row[0] != config.text_pad_idx) or (
                not audio and any(row[1:])
            ):
                raise ValueError("Prompt rows disagree with their item_kind")
            has_successor = (
                index + 1 < len(self.prompt_rows) and self.item_kind[index + 1] == 1
            )
            expected = (
                self.prompt_rows[index + 1][1:] if has_successor else [0] * config.n_vq
            )
            if (
                self.successor_audio_mask[index] != has_successor
                or self.successor_audio_codes[index] != expected
            ):
                raise ValueError("Reference successor edges disagree with prompt_rows")
            if has_successor and not audio and row[0] != config.audio_start_token_id:
                raise ValueError("Reference audio must follow an audio_start anchor")
        if len(self.prompt_rows) + self.prediction_count - 1 > context_length:
            raise ValueError(
                f"Score input exceeds the model context of {context_length} rows"
            )

    @property
    def prediction_count(self) -> int:
        return len(self.codes) + int(self.finish_reason == "stop")

    def input_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(exclude={"sample_id"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class PrismScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[PrismScoreInput] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_ids(self) -> PrismScoreBatch:
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id must be unique within a score request")
        return self


class PrismScoreResult(TypedDict):
    version: int
    sample_id: str
    input_sha256: str
    code_logprobs: list[list[float]]
    stop_probabilities: list[float]
    finish_reason: str
    temperature: float
    weight_version: str
    teacher_weight_sha256: str
    logprob_semantics: str
    stop_semantics: str
    model_identity: dict[str, str | int | bool]
