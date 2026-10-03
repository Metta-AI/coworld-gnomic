"""Absolute deadlines for work owned by a game or player process."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from gnomic.protocol import Stop, Stopped

T = TypeVar("T")
CLEANUP_SECONDS = 2.0
shutdown_deadline: ContextVar[list[float | None] | None] = ContextVar(
    "shutdown_deadline", default=None
)
owned_children: ContextVar[set[asyncio.Task] | None] = ContextVar(
    "owned_children", default=None
)


def owned_task(work: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    task = asyncio.create_task(work)
    children = owned_children.get()
    if children is not None:
        children.add(task)
        task.add_done_callback(children.discard)
    return task


class OwnershipUnsettled(RuntimeError):
    """Process supervision must terminate work which did not release ownership."""


async def settle(tasks: set[asyncio.Task], deadline: float, *, cancel: bool) -> bool:
    """Wait once for actual task settlement; cancellation is never a join proof."""
    pending = {task for task in tasks if not task.done()}
    if cancel:
        for task in pending:
            if not task.cancelling():
                task.cancel()
    if pending:
        _, pending = await asyncio.wait(
            pending, timeout=max(0, deadline - asyncio.get_running_loop().time())
        )
    for task in tasks - pending:
        if not task.cancelled():
            task.exception()  # Retrieve failures without replacing the original failure.
    return not pending


async def bounded(work: Coroutine[Any, Any, T], deadline: float) -> T:
    task = owned_task(work)
    try:
        done, _ = await asyncio.wait(
            {task}, timeout=max(0, deadline - asyncio.get_running_loop().time())
        )
        if not done:
            raise TimeoutError("owned operation deadline exceeded")
        return task.result()
    finally:
        if not task.done():
            cleanup_deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
            owner_deadline = shutdown_deadline.get()
            if owner_deadline is not None and owner_deadline[0] is not None:
                cleanup_deadline = min(cleanup_deadline, owner_deadline[0])
            if not await settle({task}, cleanup_deadline, cancel=True):
                raise OwnershipUnsettled(
                    "owned operation exceeded its cleanup deadline"
                )


@dataclass(frozen=True)
class PlayerExit:
    kind: Literal["final", "stopped"]
    cleanup_deadline: float


async def player_loop(
    recv: Callable[[], Coroutine[Any, Any, dict | None]],
    send: Callable[[dict], Coroutine[Any, Any, None]],
    handle: Callable[[dict], Coroutine[Any, Any, None]],
) -> PlayerExit:
    """Receive engine stop controls while the owned policy callback is active."""
    children: set[asyncio.Task] = set()
    token = owned_children.set(children)
    inherited_deadline = shutdown_deadline.get()
    owner_deadline: list[float | None] = [None]
    deadline_token = shutdown_deadline.set(owner_deadline)
    receiver = asyncio.create_task(recv())
    callback: asyncio.Task | None = None
    backlog: list[dict] = []
    stopping = False
    try:
        while True:
            active = {receiver}
            if callback is not None:
                active.add(callback)
            timeout = None
            if stopping:
                assert owner_deadline[0] is not None
                # Reserve half the remaining absolute budget to settle the final reader.
                timeout = (
                    max(0, owner_deadline[0] - asyncio.get_running_loop().time()) / 2
                )
            done, _ = await asyncio.wait(
                active, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                assert owner_deadline[0] is not None
                return PlayerExit("stopped", owner_deadline[0])
            if receiver in done:
                message = receiver.result()
                if message is None:
                    if stopping:
                        assert owner_deadline[0] is not None
                        return PlayerExit("stopped", owner_deadline[0])
                    raise ConnectionError(
                        "player transport closed before final or ownership stop"
                    )
                if message.get("type") == "stop":
                    control = Stop.model_validate(message)
                    stopping = True
                    backlog.clear()
                    deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
                    if owner_deadline[0] is not None:
                        deadline = min(deadline, owner_deadline[0])
                    if (
                        inherited_deadline is not None
                        and inherited_deadline[0] is not None
                    ):
                        deadline = min(deadline, inherited_deadline[0])
                    owner_deadline[0] = deadline
                    owned = children | ({callback} if callback is not None else set())
                    joined = await settle(owned, deadline, cancel=True)
                    if joined:
                        await bounded(
                            send(Stopped(stop_id=control.stop_id).model_dump()),
                            deadline,
                        )
                    else:
                        raise OwnershipUnsettled("stopped acknowledgement withheld")
                    callback = None
                    receiver = asyncio.create_task(recv())
                    continue
                if stopping and message.get("type") != "final":
                    raise ValueError(
                        "decision or repeated stop after ownership acknowledgement"
                    )
                backlog.append(message)
                if message.get("type") != "final":
                    receiver = asyncio.create_task(recv())
            if callback is not None and callback in done:
                callback.result()
                callback = None
            if callback is None and backlog:
                message = backlog.pop(0)
                callback = asyncio.create_task(handle(message))
                if message.get("type") == "final":
                    if owner_deadline[0] is None:
                        owner_deadline[0] = (
                            asyncio.get_running_loop().time() + CLEANUP_SECONDS
                        )
                    finish_deadline = (
                        asyncio.get_running_loop().time() + owner_deadline[0]
                    ) / 2
                    if not await settle({callback}, finish_deadline, cancel=False):
                        raise OwnershipUnsettled("final callback ownership unresolved")
                    callback.result()
                    return PlayerExit("final", owner_deadline[0])
    finally:
        # A stop handshake already spent its one absolute budget.
        deadline = (
            owner_deadline[0]
            if owner_deadline[0] is not None
            else asyncio.get_running_loop().time() + CLEANUP_SECONDS
        )
        if inherited_deadline is not None and inherited_deadline[0] is not None:
            deadline = min(deadline, inherited_deadline[0])
        owner_deadline[0] = deadline
        tasks = {receiver}
        if callback is not None:
            tasks.add(callback)
        tasks |= children
        owned_children.reset(token)
        shutdown_deadline.reset(deadline_token)
        if not await settle(tasks, deadline, cancel=True):
            raise OwnershipUnsettled("player receive/callback ownership unresolved")


async def run_owned(
    work: Coroutine[Any, Any, T], on_stop: Callable[[], None] | None = None
) -> T | None:
    deadline_state: list[float | None] = [None]
    deadline_token = shutdown_deadline.set(deadline_state)
    task = asyncio.create_task(work)
    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()

    def stop() -> None:
        if not stopping.is_set():
            deadline_state[0] = loop.time() + CLEANUP_SECONDS
            if on_stop is not None:
                on_stop()
        stopping.set()

    def receive_signal(signum: int, frame: Any) -> None:
        if not stopping.is_set():
            loop.call_soon_threadsafe(stop)

    previous = {
        sig: signal.signal(sig, receive_signal)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    signal_task = asyncio.create_task(stopping.wait())
    try:
        done, _ = await asyncio.wait(
            {task, signal_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if task not in done:
            assert deadline_state[0] is not None
            if not await settle({task}, deadline_state[0] + 0.1, cancel=True):
                raise OwnershipUnsettled("signal shutdown left owned work unresolved")
        if task.cancelled() and stopping.is_set():
            return None
        return task.result()
    finally:
        if not stopping.is_set():
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        else:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
        # Keep repeated signals inert through interpreter teardown after ownership settles.
        # SIG_DFL here lets a second signal interrupt the private seal/process exit.
        await settle({signal_task}, loop.time(), cancel=True)
        shutdown_deadline.reset(deadline_token)


def main_owned(
    work: Coroutine[Any, Any, T], on_stop: Callable[[], None] | None = None
) -> T | None:
    """Do not invoke asyncio.run's unbounded cancellation sweep on failed ownership."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_owned(work, on_stop))
    finally:
        loop.close()
        asyncio.set_event_loop(None)
