# SPDX-License-Identifier: Apache-2.0
"""SGLang request and response adapters for Local action scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids
from sglang_omni.models.moss_tts_local.scoring_protocol import LocalScoreInput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData


@dataclass
class ScoreRequestData(ARRequestData):
    score: LocalScoreInput | None = None
    rows: torch.Tensor | None = None
    score_result: dict | None = None
    stage_payload: Any = None
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    enforce_request_limits: bool = True
    admission_version: str | None = None


def build_score_request(payload, *, model):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import get_serving
    from sglang.srt.sampling.sampling_params import SamplingParams

    raw = (payload.request.params or {}).get("moss_local_score")
    if raw is None:
        raise ValueError("This pipeline only accepts /score_actions requests")
    score = LocalScoreInput.model_validate(raw)
    rows = torch.tensor(
        score.prompt_rows + [[151656, *row] for row in score.codes], dtype=torch.long
    )
    token_ids = build_row_cache_key_ids(rows)
    sampling = SamplingParams(max_new_tokens=0, temperature=1.0)
    sampling.normalize(None)
    sampling.verify(int(model.config.vocab_size_list[0]))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=token_ids,
        sampling_params=sampling,
        vocab_size=int(model.config.vocab_size_list[0]),
    )
    req._moss_score_group = payload.request.params.get("moss_score_group")
    req._moss_score_group_size = int(
        payload.request.params.get("moss_score_group_size", 1)
    )
    req._moss_score_group_index = int(
        payload.request.params.get("moss_score_group_index", 0)
    )
    if (
        not 1 <= req._moss_score_group_size <= 8
        or not 0 <= req._moss_score_group_index < req._moss_score_group_size
    ):
        raise ValueError("Invalid atomic scoring group metadata")
    req._moss_score_group_created = time.perf_counter()
    req.tokenizer = None
    req._input_embeds_are_projected = True
    return ScoreRequestData(
        input_ids=torch.tensor(token_ids),
        max_new_tokens=0,
        temperature=1.0,
        req=req,
        score=score,
        rows=rows,
        stage_payload=payload,
        admission_version=str(get_serving().weight_version),
    )


def score_result(data):
    if data.score_result is None:
        raise RuntimeError("Teacher returned without scoring the requested actions")
    if str(data.weight_version) != data.admission_version:
        raise RuntimeError("Teacher weights changed during scoring")
    return StagePayload(
        request_id=data.stage_payload.request_id,
        request=data.stage_payload.request,
        data={
            "modality": "text",
            "text": "",
            "omni_rollout": data.score_result,
            "weight_version": str(data.weight_version),
            "finish_reason": "stop",
        },
    )
