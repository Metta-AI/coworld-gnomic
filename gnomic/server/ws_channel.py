"""WebSocket-backed SeatChannel over a FastAPI/starlette WebSocket.

A reader task drains the socket into a queue so the orchestrator can apply
per-window timeouts without racing the socket. On disconnect every subsequent
receive times out (defaults) until the app layer swaps in a reconnected channel.
"""

from __future__ import annotations

import asyncio
import json

from starlette.websockets import WebSocket, WebSocketDisconnect

from .channel import QueueChannelMixin, SeatChannel


class WebSocketSeatChannel(QueueChannelMixin, SeatChannel):
    def __init__(self, seat: int, websocket: WebSocket) -> None:
        self.seat = seat
        self.ws = websocket
        self.connected = True
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # unparseable frames are a no-op
                if isinstance(msg, dict):
                    await self._queue.put(msg)
        except (WebSocketDisconnect, RuntimeError):
            self.connected = False

    async def send(self, message: dict) -> None:
        if not self.connected:
            return
        try:
            await self.ws.send_text(json.dumps(message))
        except (WebSocketDisconnect, RuntimeError):
            self.connected = False

    def close_reader(self) -> None:
        if not self._reader.done():
            self._reader.cancel()
