"""WebSocket-backed SeatChannel over a FastAPI/starlette WebSocket.

A reader task drains the socket into a queue so the orchestrator can apply
per-window timeouts without racing the socket. On disconnect every subsequent
receive times out (defaults) until the app layer swaps in a reconnected channel.
"""

from __future__ import annotations

import asyncio
import json

from starlette.websockets import WebSocket, WebSocketDisconnect

from gnomic.lifecycle import settle
from gnomic.llm_transport import ReceivedHeaderPairs

from .channel import QueueChannelMixin


class WebSocketSeatChannel(QueueChannelMixin):
    def __init__(self, seat: int, websocket: WebSocket) -> None:
        self.seat = seat
        self.ws = websocket
        self.connected = True
        self.progress = {}
        self.active_rid = -1
        self.stop_id: str | None = None
        self.stop_deadline = 0.0
        self.stopped = asyncio.Event()
        self.sealed = False
        self.received_headers: dict[int, dict[str, ReceivedHeaderPairs]] = {}
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
                if isinstance(msg, dict) and not self.consume_progress(msg):
                    clean = self.private_receive(msg)
                    if clean is not None:
                        await self._queue.put(clean)
        except (WebSocketDisconnect, RuntimeError):
            self.connected = False

    async def send(self, message: dict) -> None:
        if not self.connected:
            return
        try:
            await self.ws.send_text(json.dumps(message))
        except (WebSocketDisconnect, RuntimeError):
            self.connected = False

    async def close_reader(self) -> None:
        deadline = asyncio.get_running_loop().time() + 2
        if not await settle({self._reader}, deadline, cancel=True):
            raise RuntimeError("channel reader ownership remains unresolved")
        if not self._reader.cancelled():
            self._reader.result()
