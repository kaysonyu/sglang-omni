# SPDX-License-Identifier: Apache-2.0
"""HTTP batch scoring for supplied MOSS-TTS Local actions."""

import asyncio
import uuid

from fastapi import FastAPI, HTTPException

from sglang_omni.client import ClientError, GenerateRequest, SamplingParams
from sglang_omni.models.moss_tts_local.scoring_protocol import (
    LocalScoreBatch,
    LocalScoreInput,
)


def register_local_scoring(app: FastAPI) -> None:
    @app.post("/score_actions")
    async def score_actions(body: LocalScoreBatch) -> dict:
        client = app.state.client

        async def score_one(sample: LocalScoreInput) -> dict:
            request_id = str(uuid.uuid4())
            request = GenerateRequest(
                prompt="",
                stream=False,
                output_modalities=["text"],
                sampling=SamplingParams(max_new_tokens=0),
                extra_params={"moss_local_score": sample.model_dump()},
            )
            try:
                result = await client.completion(request, request_id=request_id)
            except asyncio.CancelledError:
                await client.abort(request_id)
                raise
            score = result.omni_rollout
            if (
                not isinstance(score, dict)
                or score.get("input_sha256") != sample.input_sha256()
            ):
                raise ValueError("Backend did not score the supplied actions")
            return score

        tasks = [asyncio.create_task(score_one(sample)) for sample in body.samples]
        try:
            results = await asyncio.gather(*tasks)
        except (ClientError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {"version": 1, "results": results}
