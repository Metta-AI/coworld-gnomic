"""Small Gnomic player SDK and WebSocket process loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Protocol

import websockets

from gnomic.lifecycle import bounded, main_owned, player_loop
from gnomic.llm_transport import AttemptPublisher, LearnerWindow, current_window


class Policy(Protocol):
    seat: int

    async def respond(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


async def run_policy(policy: Policy, url: str | None = None) -> None:
    ws_url = url or os.environ.get("COWORLD_PLAYER_WS_URL")
    if not ws_url:
        raise RuntimeError("COWORLD_PLAYER_WS_URL is required")
    websocket = await websockets.connect(ws_url, max_size=16 * 1024 * 1024, ping_interval=20)

    async def handle(message: dict) -> None:
        if message["type"] == "lobby":
            policy.seat = message["seat"]
        token = None
        if "rid" in message:
            publisher = AttemptPublisher(message["rid"], send)
            token = current_window.set(
                LearnerWindow(
                    policy.seat,
                    type(policy).__name__,
                    time.monotonic() + message.get("timeout_s", 15),
                    publisher,
                )
            )
        try:
            reply = await policy.respond(message)
        finally:
            if token is not None:
                current_window.reset(token)
        if reply is not None:
            await send(reply)

    async def receive() -> dict | None:
        raw = await anext(websocket.__aiter__(), None)
        return None if raw is None else json.loads(raw)

    async def send(message: dict) -> None:
        await websocket.send(json.dumps(message, ensure_ascii=False))

    cleanup_deadline = float("inf")
    try:
        exit_state = await player_loop(receive, send, handle)
        cleanup_deadline = exit_state.cleanup_deadline
    finally:
        await bounded(
            websocket.close(),
            min(cleanup_deadline, asyncio.get_running_loop().time() + 1),
        )


def main(policy: Policy) -> None:
    main_owned(run_policy(policy))
