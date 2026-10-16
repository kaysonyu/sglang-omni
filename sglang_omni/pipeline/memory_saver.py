# SPDX-License-Identifier: Apache-2.0
"""Preserve non-AR stage allocations while releasing their physical GPU memory."""

from dataclasses import dataclass, field

import torch
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


@dataclass(kw_only=True)
class StageMemorySaver:
    tag: str
    gpu_id: int
    adapter: TorchMemorySaverAdapter = field(
        default_factory=lambda: TorchMemorySaverAdapter.create(enable=True)
    )
    offloaded: bool = False

    def change_occupation(
        self, action: str, tags: list[str] | None
    ) -> dict[str, bool | str | dict[str, list[str]]]:
        with torch.cuda.device(self.gpu_id):
            torch.cuda.synchronize()
            if not tags or "weights" in tags:
                if action == "release_memory_occupation" and not self.offloaded:
                    self.adapter.pause(self.tag)
                    self.offloaded = True
                elif action == "resume_memory_occupation" and self.offloaded:
                    self.adapter.resume(self.tag)
                    self.offloaded = False
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {
            "success": True,
            "message": "memory occupation updated",
            "data": {"offload_tags": ["weights"] if self.offloaded else []},
        }
