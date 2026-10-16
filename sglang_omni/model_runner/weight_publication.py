# SPDX-License-Identifier: Apache-2.0
"""Completeness tracking for a paused tensor weight publication."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WeightPublicationPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bucket_count: int = Field(ge=1, strict=True)
    bucket_index: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_index(self) -> "WeightPublicationPart":
        if self.bucket_index >= self.bucket_count:
            raise ValueError("Publication bucket index exceeds its declared count")
        else:
            return self


@dataclass(kw_only=True)
class WeightPublication:
    version: str
    manifest_sha256: str
    bucket_count: int
    received: int = 0
    failed: bool = False

    def begin_part(self, part: WeightPublicationPart, version: str) -> None:
        if (
            self.failed
            or version != self.version
            or part.manifest_sha256 != self.manifest_sha256
            or part.bucket_count != self.bucket_count
            or part.bucket_index != self.received
        ):
            self.failed = True
            raise ValueError(
                "Publication changed, skipped a bucket, or requires a full retry"
            )
        else:
            self.failed = True

    def finish_part(self) -> None:
        self.received += 1
        self.failed = False

    @property
    def complete(self) -> bool:
        return not self.failed and self.received == self.bucket_count

    def snapshot(self) -> dict[str, str | int | bool]:
        return {
            "version": self.version,
            "manifest_sha256": self.manifest_sha256,
            "bucket_count": self.bucket_count,
            "received": self.received,
            "failed": self.failed,
            "complete": self.complete,
        }
