"""Small Gnomic player SDK and WebSocket process loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Protocol

import websockets

from gnomic.llm_transport import AttemptPublisher, LearnerWindow, current_window


class Policy(Protocol):
    seat: int

    async def respond(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


async def run_policy(policy: Policy, url: str | None = None) -> None:
    ws_url = url or os.environ.get("COWORLD_PLAYER_WS_URL")
    if not ws_url:
        raise RuntimeError("COWORLD_PLAYER_WS_URL is required")
    async with websockets.connect(
        ws_url, max_size=256 * 1024, ping_interval=20
    ) as websocket:
        async for raw in websocket:
            message = json.loads(raw)
            if message["type"] == "lobby":
                policy.seat = message["seat"]
            if message.get("type") == "final":
                # Give policies one terminal callback for usage/artifact logging.
                await policy.respond(message)
                return
            token = None
            publisher = None
            if "rid" in message:
                publisher = AttemptPublisher(
                    message["rid"], lambda packet: websocket.send(json.dumps(packet))
                )
                window = LearnerWindow(
                    policy.seat,
                    type(policy).__name__,
                    time.monotonic() + message.get("timeout_s", 15),
                    publisher,
                )
                token = current_window.set(window)
            try:
                reply = await policy.respond(message)
            finally:
                if token is not None:
                    current_window.reset(token)
                if publisher is not None:
                    await publisher.drain()
            if reply is not None:
                await websocket.send(json.dumps(reply, ensure_ascii=False))


def main(policy: Policy) -> None:
    asyncio.run(run_policy(policy))
