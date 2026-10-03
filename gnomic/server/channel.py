"""Seat channels: the orchestrator's transport-agnostic view of one player.

`NullSeatChannel` stands in for never-connected seats so an unattended episode
completes on defaults. `InProcessChannel` pairs with the SDK for tests and the
conformance gate. `WebSocketSeatChannel` (ws_channel.py) is the live transport.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Callable, Sequence

from gnomic.lifecycle import settle
from gnomic.llm_transport import Attempt, ReceivedHeaderPairs
from gnomic.protocol import Stop, Stopped


class SeatChannel:
    seat: int
    connected: bool = False
    private_receive: Callable[[dict], dict | None] = staticmethod(lambda packet: packet)

    async def send(self, message: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def recv_reply(
        self, rid: int, timeout: float
    ) -> dict | None:  # pragma: no cover
        raise NotImplementedError

    def evidence_for(self, rid: int) -> list[Attempt]:
        return []

    def received_header_records(self) -> list[dict]:
        return []

    def register_window(self, rid: int) -> None:
        pass

    async def close_reader(self) -> None:
        pass

    async def stop_player(self, deadline: float) -> bool:
        return False

    def seal(self) -> None:
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


class QueueChannelMixin(SeatChannel):
    """Shared rid-matched receive over an inbound asyncio.Queue."""

    _queue: asyncio.Queue
    stop_id: str | None
    stop_deadline: float
    stopped: asyncio.Event
    sealed: bool
    progress: dict[int, dict[str, Attempt]]
    active_rid: int
    received_headers: dict[int, dict[str, ReceivedHeaderPairs]]

    def register_window(self, rid: int) -> None:
        if self.sealed or self.stop_id is not None:
            raise RuntimeError("decision issued after ownership stop")
        self.active_rid = rid

    async def stop_player(self, deadline: float) -> bool:
        self.stop_id = str(uuid.uuid4())
        self.stop_deadline = deadline
        if not self.connected:
            return False
        sender = asyncio.create_task(self.send(Stop(stop_id=self.stop_id).model_dump()))
        if not await settle({sender}, deadline, cancel=False):
            await settle({sender}, deadline, cancel=True)
            return False
        sender.result()
        acknowledgement = asyncio.create_task(self.stopped.wait())
        if not await settle({acknowledgement}, deadline, cancel=False):
            await settle({acknowledgement}, deadline, cancel=True)
            return False
        return self.stopped.is_set()

    def seal(self) -> None:
        self.sealed = True

    def consume_progress(self, message: dict) -> bool:
        if self.sealed:
            return True
        if message.get("type") == "stopped":
            control = Stopped.model_validate(message)
            if (
                self.stop_id is None
                or control.stop_id != self.stop_id
                or asyncio.get_running_loop().time() > self.stop_deadline
                or self.stopped.is_set()
            ):
                raise ValueError("invalid or stale stopped acknowledgement")
            if any(
                attempt.response_reader_joined is False
                for attempts in self.progress.values()
                for attempt in attempts.values()
            ):
                raise ValueError(
                    "stopped acknowledgement contradicts unsettled native reader"
                )
            self.stopped.set()
            return True
        if self.stopped.is_set():
            raise ValueError("player evidence or action after stopped acknowledgement")
        if message.get("type") != "private_attempt":
            return self.stop_id is not None
        rid = message["rid"]
        attempt = Attempt.model_validate(message["attempt"])
        if rid != self.active_rid and (
            rid not in self.progress or attempt.attempt_id not in self.progress[rid]
        ):
            raise ValueError("private progress outside registered decision window")
        if "received_headers" in message and message["received_headers"] is not None:
            headers = ReceivedHeaderPairs.model_validate(message["received_headers"])
            if headers.attempt_id != attempt.attempt_id:
                raise ValueError(
                    "received header pairs differ from actual attempt identity"
                )
            recorded = self.received_headers.setdefault(rid, {})
            if (
                attempt.attempt_id in recorded
                and recorded[attempt.attempt_id] != headers
            ):
                raise ValueError("received header pairs cannot be rewritten")
            recorded[attempt.attempt_id] = headers
        attempts = self.progress.setdefault(rid, {})
        if attempt.attempt_id in attempts:
            before = attempts[attempt.attempt_id]
            if before.latency_ms is not None and before != attempt:
                raise ValueError("terminal native evidence cannot be rewritten")
            for name in ("policy", "origin", "prompt", "request", "decoder"):
                if getattr(before, name) != getattr(attempt, name):
                    raise ValueError("started native evidence cannot be rewritten")
            for field in ("response_body_b64", "response_headers_b64"):
                observed = getattr(before, field)
                received = getattr(attempt, field)
                if observed is not None:
                    if received is None or not base64.b64decode(
                        received, validate=True
                    ).startswith(base64.b64decode(observed, validate=True)):
                        raise ValueError("received native bytes cannot be rewritten")
                    if before.response_complete is True and received != observed:
                        raise ValueError("completed response cannot be rewritten")
            if (
                before.response_complete is True
                and attempt.response_complete is not True
            ):
                raise ValueError("completed response cannot be rewritten")
            for name in (
                "http_status",
                "response_headers",
                "platform_call_id",
                "provider_request_id",
                "model_identity",
                "tokenizer_identity",
                "chat_template_sha256",
            ):
                observed = getattr(before, name)
                if observed is not None and observed != getattr(attempt, name):
                    raise ValueError("received native identity cannot be rewritten")
        self.private_receive(message)
        attempts[attempt.attempt_id] = attempt
        return True

    def received_header_records(self) -> list[dict]:
        return [
            {
                "rid": rid,
                "attempts": [
                    record.model_dump(mode="json") for record in records.values()
                ],
            }
            for rid, records in sorted(self.received_headers.items())
        ]

    def evidence_for(self, rid: int) -> list[Attempt]:
        return [
            attempt.model_copy(deep=True)
            for attempt in self.progress.get(rid, {}).values()
        ]

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
                if self.consume_progress(msg):
                    continue
                return msg
            # Stale reply from an earlier window (or noise): drop and keep waiting.


class InProcessChannel(QueueChannelMixin):
    """In-memory pair for tests/conformance: server side + player side queues."""

    def __init__(self, seat: int) -> None:
        self.seat = seat
        self.connected = True
        self.progress = {}
        self.active_rid = -1
        self.stop_id: str | None = None
        self.stop_deadline = 0.0
        self.stopped = asyncio.Event()
        self.sealed = False
        self.received_headers: dict[int, dict[str, ReceivedHeaderPairs]] = {}
        self._queue: asyncio.Queue[dict] = asyncio.Queue()  # player -> server
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()  # server -> player

    async def send(self, message: dict) -> None:
        await self.outbox.put(message)

    async def player_send(self, message: dict) -> None:
        if not self.consume_progress(message):
            clean = self.private_receive(message)
            if clean is not None:
                await self._queue.put(clean)

    async def player_recv(self) -> dict:
        return await self.outbox.get()


async def settle_channels(channels: Sequence[SeatChannel], deadline: float) -> bool:
    """Keep receiving final evidence until nonce-bound ACKs, then seal and join readers."""
    stops = {asyncio.create_task(c.stop_player(deadline)) for c in channels}
    joined = await settle(stops, deadline, cancel=False)
    if not joined:
        await settle(stops, deadline, cancel=True)
    acknowledged = joined and all(
        not task.cancelled() and task.exception() is None and task.result()
        for task in stops
    )
    for channel in channels:
        channel.seal()
    readers = {asyncio.create_task(c.close_reader()) for c in channels}
    readers_joined = await settle(readers, deadline, cancel=False)
    if not readers_joined:
        await settle(readers, deadline, cancel=True)
    return (
        acknowledged
        and readers_joined
        and all(not task.cancelled() and task.exception() is None for task in readers)
        and asyncio.get_running_loop().time() <= deadline
    )
