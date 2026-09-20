# SPDX-License-Identifier: Apache-2.0
"""Batch exact-action scoring endpoint for discrete TTS teacher pipelines."""

import asyncio
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sglang_omni.client import ClientError, GenerateRequest, SamplingParams
from sglang_omni.models.moss_tts_local.scoring_protocol import LocalScoreInput
from sglang_omni.models.higgs_tts.scoring_protocol import HiggsScoreInput


class ScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[LocalScoreInput | HiggsScoreInput] = Field(
        min_length=1, max_length=64
    )

    @model_validator(mode="after")
    def unique_ids(self):
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("Sample IDs must be unique within a score request")
        return self


def register_local_scoring(app):
    @app.post("/score_actions")
    async def score_actions(body: ScoreBatch):
        client = app.state.client

        async def score_one(sample, group_id, group_size, group_index):
            payload_key = (
                "moss_local_score"
                if isinstance(sample, LocalScoreInput)
                else "higgs_score"
            )
            request = GenerateRequest(
                prompt="",
                stream=False,
                output_modalities=["text"],
                sampling=SamplingParams(max_new_tokens=0),
                extra_params={
                    payload_key: sample.model_dump(),
                    "moss_score_group": group_id,
                    "moss_score_group_size": group_size,
                    "moss_score_group_index": group_index,
                },
            )
            result = await client.completion(request, request_id=str(uuid.uuid4()))
            score = result.omni_rollout
            if (
                not isinstance(score, dict)
                or score.get("input_sha256") != sample.input_sha256()
            ):
                raise ValueError(
                    "Backend did not return scores for the requested original actions"
                )
            return score

        try:
            groups, current, rows = [], [], 0
            for sample in body.samples:
                size = (
                    (len(sample.prompt_rows) + len(sample.codes))
                    if isinstance(sample, LocalScoreInput)
                    else (len(sample.prompt_token_ids) + len(sample.codes) - 1)
                )
                if current and (rows + size > 2048 or len(current) == 8):
                    groups.append(current)
                    current, rows = [], 0
                current.append(sample)
                rows += size
            if current:
                groups.append(current)
            results = []
            for group in groups:
                group_id = str(uuid.uuid4())
                tasks = [
                    asyncio.create_task(score_one(sample, group_id, len(group), index))
                    for index, sample in enumerate(group)
                ]
                try:
                    results.extend(
                        await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)
                    )
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        except (ClientError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"version": 1, "results": results}
