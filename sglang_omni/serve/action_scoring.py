# SPDX-License-Identifier: Apache-2.0
"""HTTP batch scoring for supplied MOSS-TTS actions."""

import asyncio
import uuid
from typing import Literal, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import JsonValue

from sglang_omni.client import ClientError, GenerateRequest, SamplingParams
from sglang_omni.models.moss_tts_local.scoring_protocol import (
    LocalScoreBatch,
    LocalScoreInput,
)
from sglang_omni.models.moss_tts_prism.scoring_protocol import (
    PrismScoreBatch,
    PrismScoreInput,
)
from sglang_omni.serve.openai_errors import is_bad_request_error


class ScoreResponse(TypedDict):
    version: int
    results: list[dict[str, JsonValue]]


def register_action_scoring(
    app: FastAPI, architectures: list[str] | None = None
) -> None:
    async def score_samples(
        samples: list[LocalScoreInput] | list[PrismScoreInput],
        parameter: Literal["moss_local_score", "moss_prism_score"],
    ) -> ScoreResponse:
        client = app.state.client

        async def score_one(
            sample: LocalScoreInput | PrismScoreInput,
        ) -> dict[str, JsonValue]:
            request_id = str(uuid.uuid4())
            request = GenerateRequest(
                prompt="",
                stream=False,
                output_modalities=["text"],
                sampling=SamplingParams(max_new_tokens=0),
                extra_params={parameter: sample.model_dump()},
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

        tasks = [asyncio.create_task(score_one(sample)) for sample in samples]
        try:
            results = await asyncio.gather(*tasks)
        except (ClientError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            else:
                raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {"version": 1, "results": results}

    if "MossTTSPrismModel" in (architectures or []):

        @app.post("/score_actions")
        async def score_prism(body: PrismScoreBatch) -> ScoreResponse:
            return await score_samples(body.samples, "moss_prism_score")

    else:

        @app.post("/score_actions")
        async def score_local(body: LocalScoreBatch) -> ScoreResponse:
            return await score_samples(body.samples, "moss_local_score")
