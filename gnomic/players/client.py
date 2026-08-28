"""Small Gnomic player SDK and WebSocket process loop."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol

import websockets


class Policy(Protocol):
    async def respond(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


async def run_policy(policy: Policy, url: str | None = None) -> None:
    ws_url = url or os.environ.get("COWORLD_PLAYER_WS_URL")
    if not ws_url:
        raise RuntimeError("COWORLD_PLAYER_WS_URL is required")
    async with websockets.connect(ws_url, max_size=256 * 1024, ping_interval=20) as websocket:
        async for raw in websocket:
            message = json.loads(raw)
            if message.get("type") == "final":
                # Give policies one terminal callback for usage/artifact logging.
                await policy.respond(message)
                return
            reply = await policy.respond(message)
            if reply is not None:
                await websocket.send(json.dumps(reply, ensure_ascii=False))


def main(policy: Policy) -> None:
    asyncio.run(run_policy(policy))
