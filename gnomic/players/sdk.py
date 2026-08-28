"""Player SDK: implement three callbacks and the harness ensures legal play.

    class MyPolicy(Policy):
        def propose(self, view) -> dict: ...
        def debate(self, view) -> dict: ...
        def vote(self, view) -> str:  # "aye" | "nay"

The harness owns transport, reconnects, deadlines, and clean exit. A callback
that raises or overruns its window sends NO reply — the server applies the phase
default and marks it ``default: true`` (never fabricate an action client-side).
Run ``python -m gnomic.players.conformance your_module:YourPolicy`` before
submitting: zero unintended defaults is the admission gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ..protocol import make_reply

REPLY_MARGIN_S = 0.35


def stable_rng_int(*parts: object, mod: int) -> int:
    """Deterministic pseudo-random int (builtin hash() is per-process salted)."""
    basis = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(basis.encode()).hexdigest(), 16) % mod


# --- view -------------------------------------------------------------------


@dataclass
class GameView:
    """Folded game state: every prompt is answerable from the view alone."""

    seat: int = -1
    session: dict = field(default_factory=dict)
    rules: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    turn: int = 0
    proposer: int = -1
    proposal: dict = field(default_factory=dict)
    debates: list[dict] = field(default_factory=list)
    votes: list[dict] = field(default_factory=list)
    last_ruling: dict = field(default_factory=dict)
    game_over: dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.session.get("seats", [])) or 3

    @property
    def limits(self) -> dict:
        return self.session.get("limits", {})

    def my_points(self) -> int:
        players = self.state.get("players", [])
        if 0 <= self.seat < len(players):
            try:
                return int(players[self.seat].get("points", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def fold(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "game_start":
            self.session = msg.get("session", {})
            self.seat = msg.get("you", {}).get("seat", self.seat)
            self.rules = msg.get("rules", [])
            self.state = msg.get("state", {})
            self.history = msg.get("history", [])
        elif mtype == "turn_start":
            self.turn = msg.get("turn", self.turn)
            self.proposer = msg.get("proposer", -1)
            self.rules = msg.get("rules", self.rules)
            self.state = msg.get("state", self.state)
            self.proposal = {}
            self.debates = []
            self.votes = []
        elif mtype == "proposal_made":
            self.proposal = msg.get("proposal", {})
        elif mtype == "debate_made":
            self.debates = list(msg.get("statements", []))
        elif mtype == "debate_request":
            self.proposal = msg.get("proposal", self.proposal)
        elif mtype == "vote_request":
            self.proposal = msg.get("proposal", self.proposal)
            self.debates = msg.get("debates", self.debates)
        elif mtype == "vote_reveal":
            self.votes = msg.get("votes", [])
        elif mtype == "judge_ruling":
            self.rules = msg.get("rules", self.rules)
            self.state = msg.get("state", self.state)
            self.last_ruling = msg
            self.history.append(
                {
                    "turn": msg.get("turn"),
                    "proposer": self.proposer,
                    "proposal": self.proposal,
                    "votes": self.votes,
                    "passed_vote": msg.get("passed_vote"),
                    "ruling": {"summary": msg.get("summary", ""), "source": msg.get("source", "")},
                }
            )
        elif mtype == "game_over":
            self.game_over = msg


# --- policy -----------------------------------------------------------------


class Policy:
    """Override the three decision callbacks. Raising or overrunning a window is
    safe (the server defaults) but fails the conformance gate."""

    def propose(self, view: GameView) -> dict:
        raise NotImplementedError

    def debate(self, view: GameView) -> dict:
        raise NotImplementedError

    def vote(self, view: GameView) -> str:
        raise NotImplementedError

    def on_message(self, view: GameView, raw: dict) -> None:
        """Optional observer for every inbound message."""


@dataclass
class DefaultEvent:
    turn: int
    phase: str
    reason: str


# --- transports ---------------------------------------------------------------


class Transport:
    async def recv(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    async def send(self, message: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        pass


class InProcessTransport(Transport):
    """Player side of a server InProcessChannel (tests + conformance gate)."""

    def __init__(self, channel: Any) -> None:
        self.channel = channel

    async def recv(self) -> dict:
        return await self.channel.player_recv()

    async def send(self, message: dict) -> None:
        await self.channel.player_send(message)


class WebSocketTransport(Transport):
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None

    async def connect(self) -> None:
        import websockets

        self.ws = await websockets.connect(self.url, max_size=8 * 1024 * 1024)

    async def recv(self) -> dict:
        assert self.ws is not None
        raw = await self.ws.recv()
        return json.loads(raw)

    async def send(self, message: dict) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(message))

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()


# --- session (the harness) -----------------------------------------------------


class PlayerSession:
    def __init__(self, policy: Policy, transport: Transport) -> None:
        self.policy = policy
        self.transport = transport
        self.view = GameView()
        self.defaults: list[DefaultEvent] = []
        self.final: dict | None = None

    async def _callback(self, phase: str, timeout_s: float) -> Any:
        """Run a sync policy callback in an executor under the window's deadline."""
        loop = asyncio.get_event_loop()
        budget = max(0.1, timeout_s - REPLY_MARGIN_S)
        fn = getattr(self.policy, phase)
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, fn, self.view), timeout=budget)
        except asyncio.TimeoutError:
            self.defaults.append(DefaultEvent(self.view.turn, phase, "timeout"))
        except Exception as e:  # noqa: BLE001 - a broken callback must not kill the episode
            self.defaults.append(DefaultEvent(self.view.turn, phase, f"exception: {type(e).__name__}: {e}"))
        return None

    async def handle(self, msg: dict) -> None:
        self.view.fold(msg)
        try:
            self.policy.on_message(self.view, msg)
        except Exception:  # noqa: BLE001 - observer hooks never crash the loop
            pass
        mtype = msg.get("type")
        if mtype == "proposal_request":
            proposal = await self._callback("propose", msg.get("timeout_s", 10))
            if isinstance(proposal, dict):
                await self.transport.send(make_reply(msg["rid"], {"proposal": proposal}))
            elif isinstance(proposal, str) and proposal.strip():
                await self.transport.send(make_reply(msg["rid"], {"proposal": {
                    "kind": "enact", "text": proposal.strip(), "rationale": "SDK shorthand proposal"
                }}))
            elif proposal is not None:
                self.defaults.append(DefaultEvent(self.view.turn, "propose", "empty return"))
        elif mtype == "debate_request":
            debate = await self._callback("debate", msg.get("timeout_s", 10))
            if isinstance(debate, dict) and isinstance(debate.get("text"), str):
                await self.transport.send(make_reply(msg["rid"], debate))
            elif isinstance(debate, str) and debate.strip():
                await self.transport.send(make_reply(msg["rid"], {
                    "text": debate.strip(), "vote_intent": "nay"
                }))
            elif debate is not None:
                self.defaults.append(DefaultEvent(self.view.turn, "debate", "empty return"))
        elif mtype == "vote_request":
            vote = await self._callback("vote", msg.get("timeout_s", 6))
            if vote in ("aye", "nay"):
                await self.transport.send(make_reply(msg["rid"], {"vote": vote}))
            elif vote is not None:
                self.defaults.append(DefaultEvent(self.view.turn, "vote", f"invalid return {vote!r}"))
        elif mtype == "final":
            self.final = msg

    async def run(self) -> None:
        """Consume messages until `final`. Treat any close after `final` as clean."""
        while self.final is None:
            msg = await self.transport.recv()
            if isinstance(msg, dict):
                await self.handle(msg)


async def run_ws_player(policy: Policy, url: str | None = None, *, max_attempts: int = 8) -> None:
    """Connect to the game (reconnecting with backoff) and play until `final`."""
    url = url or os.environ["COWORLD_PLAYER_WS_URL"]
    session = PlayerSession(policy, WebSocketTransport(url))
    attempt = 0
    while session.final is None:
        transport = WebSocketTransport(url)
        try:
            await transport.connect()
            attempt = 0
            session.transport = transport
            await session.run()
        except Exception as e:  # noqa: BLE001 - reconnect on any transport error
            attempt += 1
            if attempt >= max_attempts:
                print(f"[sdk] giving up after {attempt} connection failures: {e}", file=sys.stderr, flush=True)
                raise
            await asyncio.sleep(min(5.0, 0.5 * attempt))
        finally:
            await transport.close()
    for event in session.defaults:
        print(f"[sdk] defaulted turn={event.turn} phase={event.phase}: {event.reason}",
              file=sys.stderr, flush=True)


def main_for(policy_factory) -> None:
    """Entrypoint helper for player containers."""
    asyncio.run(run_ws_player(policy_factory()))
