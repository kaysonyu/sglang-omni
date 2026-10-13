# SPDX-License-Identifier: Apache-2.0
"""HTTP boundaries for request profiles, including body transfer waits."""

import time
import uuid
from contextvars import ContextVar
from typing import Literal

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang_omni.profiler.event_recorder import get_recorder

HTTP_PROFILE_ID: ContextVar[str | None] = ContextVar(
    "omni_http_profile_id", default=None
)
PROFILE_HEADER = b"x-omni-profile-id"
PROFILE_PATHS = frozenset({"/generate", "/score_actions", "/score"})


def current_http_profile_id() -> str | None:
    return HTTP_PROFILE_ID.get()


class RequestProfileMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        stage: Literal["frontend_http", "router_http", "asr_http"],
        header_only: bool = False,
    ) -> None:
        self.app: ASGIApp = app
        self.stage: Literal["frontend_http", "router_http", "asr_http"] = stage
        self.header_only: bool = header_only

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        recorder = get_recorder()
        if (
            not recorder.is_active()
            or scope["type"] != "http"
            or scope["path"] not in PROFILE_PATHS
        ):
            await self.app(scope, receive, send)
        else:
            profile_id = next(
                (
                    value.decode("latin-1")
                    for name, value in scope["headers"]
                    if name == PROFILE_HEADER
                ),
                None,
            )
            if self.header_only and profile_id is None:
                await self.app(scope, receive, send)
            else:
                profile_id = profile_id or str(uuid.uuid4())
                token = HTTP_PROFILE_ID.set(profile_id)
                request_bytes = 0
                response_bytes = 0
                receive_wait_ns = 0
                send_wait_ns = 0
                first_body = True
                status_code = 0
                recorder.emit(
                    request_id=profile_id,
                    stage=self.stage,
                    event_name="http_received",
                    metadata={
                        "path": scope["path"],
                        "monotonic_ns": time.monotonic_ns(),
                    },
                )

                async def receive_profiled() -> Message:
                    nonlocal request_bytes, receive_wait_ns
                    started = time.monotonic_ns()
                    message = await receive()
                    receive_wait_ns += time.monotonic_ns() - started
                    if message["type"] == "http.request":
                        request_bytes += len(message.get("body", b""))
                        if not message.get("more_body", False):
                            recorder.emit(
                                request_id=profile_id,
                                stage=self.stage,
                                event_name="http_body_received",
                                metadata={
                                    "bytes": request_bytes,
                                    "receive_wait_ns": receive_wait_ns,
                                },
                            )
                    return message

                async def send_profiled(message: Message) -> None:
                    nonlocal response_bytes, send_wait_ns, first_body, status_code
                    if message["type"] == "http.response.start":
                        status_code = message["status"]
                        recorder.emit(
                            request_id=profile_id,
                            stage=self.stage,
                            event_name="http_response_start",
                            metadata={"status": status_code},
                        )
                    elif message["type"] == "http.response.body":
                        response_bytes += len(message.get("body", b""))
                        if first_body:
                            first_body = False
                            recorder.emit(
                                request_id=profile_id,
                                stage=self.stage,
                                event_name="http_first_body",
                            )
                    started = time.monotonic_ns()
                    await send(message)
                    send_wait_ns += time.monotonic_ns() - started
                    if message["type"] == "http.response.body" and not message.get(
                        "more_body", False
                    ):
                        recorder.emit(
                            request_id=profile_id,
                            stage=self.stage,
                            event_name="http_response_sent",
                            metadata={
                                "bytes": response_bytes,
                                "send_wait_ns": send_wait_ns,
                            },
                        )

                try:
                    await self.app(scope, receive_profiled, send_profiled)
                finally:
                    recorder.emit(
                        request_id=profile_id,
                        stage=self.stage,
                        event_name="http_complete",
                        metadata={
                            "status": status_code,
                            "monotonic_ns": time.monotonic_ns(),
                        },
                    )
                    HTTP_PROFILE_ID.reset(token)
