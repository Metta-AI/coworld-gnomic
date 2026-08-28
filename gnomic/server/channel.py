"""Seat channels: the orchestrator's transport-agnostic view of one player.

`NullSeatChannel` stands in for never-connected seats so an unattended episode
completes on defaults. `InProcessChannel` pairs with the SDK for tests and the
conformance gate. `WebSocketSeatChannel` (ws_channel.py) is the live transport.
"""

from __future__ import annotations

import asyncio


class SeatChannel:
    seat: int
    connected: bool = False

    async def send(self, message: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def recv_reply(self, rid: int, timeout: float) -> dict | None:  # pragma: no cover
        raise NotImplementedError

    def close_reader(self) -> None:
        pass


class NullSeatChannel(SeatChannel):
    def __init__(self, seat: int) -> None:
        self.seat = seat
        self.connected = False

    async def send(self, message: dict) -> None:
        return

    async def recv_reply(self, rid: int, timeout: float) -> dict | None:
        # A vacant seat never answers, but never stalls the episode either: the
        # orchestrator applies the phase default immediately.
        return None


class QueueChannelMixin:
    """Shared rid-matched receive over an inbound asyncio.Queue."""

    _queue: asyncio.Queue

    async def recv_reply(self, rid: int, timeout: float) -> dict | None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if not isinstance(msg, dict):
                continue
            if msg.get("rid") == rid:
                return msg
            # Stale reply from an earlier window (or noise): drop and keep waiting.


class InProcessChannel(QueueChannelMixin, SeatChannel):
    """In-memory pair for tests/conformance: server side + player side queues."""

    def __init__(self, seat: int) -> None:
        self.seat = seat
        self.connected = True
        self._queue: asyncio.Queue[dict] = asyncio.Queue()  # player -> server
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()  # server -> player

    async def send(self, message: dict) -> None:
        await self.outbox.put(message)

    async def player_send(self, message: dict) -> None:
        await self._queue.put(message)

    async def player_recv(self) -> dict:
        return await self.outbox.get()
