"""Async episode orchestrator for action -> proposal -> debate -> vote -> judge turns."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

from ..engine import (
    HOST_CONSTRAINTS,
    SEAT_COUNT,
    Board,
    default_action,
    default_proposal,
)
from ..judge import (
    ACTION_JUDGE_SYSTEM,
    JUDGE_SYSTEM,
    DeterministicJudge,
    Judge,
    LlmJudge,
    adjudicate,
    adjudicate_action,
)
from ..training import Capture
from .channel import SeatChannel
from .config import GameConfig

# Fixed house names by seat, per Heartleaf lore: House One is Ivan's, and so on.
# A policy may claim its own gnome name through introduce_request; these are the
# defaults for seats whose policy never answers (baselines, vacant seats).
GNOME_SEAT_NAMES = ["Ivan", "Anton", "Yura"]

Broadcast = Callable[[dict[str, Any]], Awaitable[None]]


class Episode:
    def __init__(
        self,
        config: GameConfig,
        channels: list[SeatChannel],
        *,
        seed: int,
        broadcast: Broadcast | None = None,
        teacher_seats: frozenset[int] = frozenset(),
    ) -> None:
        self.config = config
        self.channels = channels
        self.seed = seed
        self.board = Board.initial(seed=seed)
        self._rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._broadcast = broadcast
        self._rid = 0
        self.current_turn = 0
        self.current_phase = "lobby"
        self.winner_slots: list[int] = []
        self.termination = ""
        self.judge_usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
        }
        self.seat_names: list[str] = list(GNOME_SEAT_NAMES)
        self.judge: Judge = (
            DeterministicJudge()
            if config.judge_mode == "deterministic"
            else LlmJudge(config.judge_model)
        )
        self.results: dict[str, Any] | None = None
        self.judge_systems = {"ruling": JUDGE_SYSTEM, "action": ACTION_JUDGE_SYSTEM}
        self.capture = Capture(self, teacher_seats=teacher_seats)
        for channel in self.channels:
            channel.private_receive = lambda packet, seat=channel.seat: (
                self.capture.receive(seat, packet)
            )

    async def _send(self, channel: SeatChannel, message: dict) -> None:
        self.capture.delivered(channel.seat, message)
        await channel.send(message)

    @staticmethod
    def canonical_action(request: dict, seat: int, raw: dict | None) -> dict:
        kind = request["type"]
        if kind == "introduce_request":
            return (
                {"name": raw["name"].strip()[:40]}
                if isinstance(raw, dict) and isinstance(raw.get("name"), str)
                else {}
            )
        if kind in {"action_request", "action_repair_request"}:
            action = Episode._parse_action(raw)
            return {"action": action["text"]}
        if kind == "proposal_request":
            action = Episode._parse_proposal(raw, request["turn"])
            return {"proposal": {k: v for k, v in action.items() if k != "default"}}
        if kind == "debate_request":
            action = Episode._parse_debate(seat, raw)
            return {k: v for k, v in action.items() if k not in {"seat", "default"}}
        if kind == "vote_request":
            action = Episode._parse_vote(seat, raw)
            return {"vote": action["vote"]}
        raise ValueError("Unknown decision request")

    def _next_rid(self) -> int:
        self._rid += 1
        return self._rid

    def _seats(self, *, attributed: bool = False) -> list[dict[str, Any]]:
        """Seat rows for one audience.

        Agents only ever see the gnome names; the spectator stream and the
        replay additionally carry each seat's owning platform player.
        """
        rows: list[dict[str, Any]] = []
        for seat in range(SEAT_COUNT):
            row: dict[str, Any] = {"seat": seat, "policy": self.seat_names[seat]}
            if attributed:
                row["player"] = self.config.players[seat].name
            rows.append(row)
        return rows

    async def _collect_introductions(self) -> None:
        """Ask every seat for its gnome name before the Moot is called to order."""

        async def ask(channel: SeatChannel) -> str | None:
            rid = self._next_rid()
            await self._send(channel, {"type": "lobby", "seat": channel.seat})
            await self._send(
                channel,
                {
                    "type": "introduce_request",
                    "rid": rid,
                    "timeout_s": self.config.introduce_window_s,
                },
            )
            reply = await channel.recv_reply(rid, self.config.introduce_window_s)
            name = reply.get("name") if isinstance(reply, dict) else None
            self.capture.windows[(channel.seat, rid)].submitted = reply
            return name if isinstance(name, str) else None

        raw = await asyncio.gather(*(ask(channel) for channel in self.channels))
        used: dict[str, int] = {}
        names: list[str] = []
        for seat in range(SEAT_COUNT):
            name = "".join(ch for ch in (raw[seat] or "").strip() if ch.isprintable())[
                :40
            ].strip()
            if not name:
                name = GNOME_SEAT_NAMES[seat]
            count = used.get(name.lower(), 0) + 1
            used[name.lower()] = count
            names.append(name if count == 1 else f"{name} ({count})")
        self.seat_names = names
        for (seat, rid), window in self.capture.windows.items():
            if window.request["type"] == "introduce_request":
                self.capture.consumed(
                    seat,
                    rid,
                    window.submitted,
                    {"name": names[seat]},
                    raw[seat] is None,
                )

    async def _emit(self, message: dict[str, Any]) -> None:
        self.events.append(message)
        if self._broadcast is not None:
            await self._broadcast(message)

    async def _send_all(self, message: dict[str, Any]) -> None:
        await asyncio.gather(
            *(self._send(channel, message) for channel in self.channels)
        )

    async def _announce_start(self) -> None:
        await self._collect_introductions()
        session = {
            "id": f"gnomic-{self.seed:08x}",
            "seats": self._seats(),
            "limits": self.config.limits_payload(),
        }
        for seat, channel in enumerate(self.channels):
            await self._send(
                channel,
                {
                    "type": "game_start",
                    "session": session,
                    "you": {"seat": seat},
                    "host_constraints": list(HOST_CONSTRAINTS),
                    "rules": self.board.rules_dict(include_history=False),
                    "state": self.board.state.as_dict(),
                    "history": [],
                },
            )
        await self._emit(
            {
                "type": "game_start",
                "session": {**session, "seats": self._seats(attributed=True)},
                "host_constraints": list(HOST_CONSTRAINTS),
                "rules": self.board.rules_dict(),
                "state": self.board.state.as_dict(),
            }
        )

    @staticmethod
    def _parse_action(raw: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return default_action()
        value = raw.get("action", raw.get("text"))
        if isinstance(value, dict):
            value = value.get("text")
        if not isinstance(value, str) or not value.strip():
            return default_action()
        return {"text": value.strip()[:2_000], "default": False}

    @staticmethod
    def _parse_proposal(raw: dict[str, Any] | None, turn: int) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return default_proposal(turn)
        payload = raw.get("proposal", raw)
        if not isinstance(payload, dict):
            return default_proposal(turn)
        kind = payload.get("kind")
        text = payload.get("text")
        rule_id = payload.get("rule_id")
        rationale = payload.get("rationale", "")
        if kind == "enact" and isinstance(text, str) and 0 < len(text.strip()) <= 2_000:
            return {
                "kind": kind,
                "text": text.strip(),
                "rationale": str(rationale)[:2_000],
                "default": False,
            }
        if (
            kind == "amend"
            and isinstance(rule_id, int)
            and isinstance(text, str)
            and 0 < len(text.strip()) <= 2_000
        ):
            return {
                "kind": kind,
                "rule_id": rule_id,
                "text": text.strip(),
                "rationale": str(rationale)[:2_000],
                "default": False,
            }
        if kind == "repeal" and isinstance(rule_id, int):
            return {
                "kind": kind,
                "rule_id": rule_id,
                "rationale": str(rationale)[:2_000],
                "default": False,
            }
        if kind == "transmute" and isinstance(rule_id, int):
            return {
                "kind": kind,
                "rule_id": rule_id,
                "rationale": str(rationale)[:2_000],
                "default": False,
            }
        return default_proposal(turn)

    @staticmethod
    def _parse_debate(seat: int, raw: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            return {
                "seat": seat,
                "text": "No statement.",
                "vote_intent": "nay",
                "default": True,
            }
        intent = raw.get("vote_intent")
        if intent not in {"aye", "nay"}:
            intent = "nay"
        return {
            "seat": seat,
            "text": raw["text"][:2_000],
            "vote_intent": intent,
            "default": False,
        }

    @staticmethod
    def _parse_vote(seat: int, raw: dict[str, Any] | None) -> dict[str, Any]:
        vote = raw.get("vote") if isinstance(raw, dict) else None
        if vote not in {"aye", "nay"}:
            return {
                "seat": seat,
                "vote": "nay",
                "reason": "Invalid or missing vote.",
                "default": True,
            }
        return {
            "seat": seat,
            "vote": vote,
            "reason": str(raw.get("reason", ""))[:1_000],
            "default": False,
        }

    async def _proposal(self, turn: int, proposer: int) -> dict[str, Any]:
        self.current_phase = "proposal"
        rid = self._next_rid()
        request = {
            "type": "proposal_request",
            "turn": turn,
            "rid": rid,
            "timeout_s": self.config.proposal_window_s,
        }
        await self._send(self.channels[proposer], request)
        raw = await self.channels[proposer].recv_reply(
            rid, self.config.proposal_window_s
        )
        proposal = self._parse_proposal(raw, turn)
        self.capture.consumed(
            proposer,
            rid,
            raw,
            {"proposal": {k: v for k, v in proposal.items() if k != "default"}},
            proposal["default"],
        )
        return proposal

    def _add_judge_usage(self, usage: dict[str, Any]) -> None:
        for key in self.judge_usage:
            self.judge_usage[key] += usage.get(key, 0)

    async def _action_attempt(
        self,
        turn: int,
        proposer: int,
        *,
        attempt: int,
        prior_action: dict[str, Any] | None = None,
        rejection_reason: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.current_phase = "action" if attempt == 1 else "action_repair"
        rid = self._next_rid()
        request: dict[str, Any] = {
            "type": "action_request" if attempt == 1 else "action_repair_request",
            "turn": turn,
            "rid": rid,
            "timeout_s": self.config.action_window_s,
            "attempt": attempt,
        }
        if attempt == 2:
            request["original_action"] = prior_action or default_action()
            request["rejection_reason"] = rejection_reason or "The action was rejected."
        await self._send(self.channels[proposer], request)
        raw = await self.channels[proposer].recv_reply(rid, self.config.action_window_s)
        action = self._parse_action(raw)
        self.capture.consumed(
            proposer, rid, raw, {"action": action["text"]}, action["default"]
        )
        made = {
            "type": "action_made",
            "turn": turn,
            "player": proposer,
            "attempt": attempt,
            "action": action,
        }
        await self._send_all(made)
        await self._emit(made)

        self.current_phase = "action_judge"
        action_record = {
            "turn": turn,
            "player": proposer,
            "attempt": attempt,
            "text": action["text"],
        }
        if rejection_reason:
            action_record["prior_rejection"] = rejection_reason
        ruling = await asyncio.wait_for(
            adjudicate_action(
                self.board,
                action_record=action_record,
                turn=turn,
                turns_max=self.config.turns_max,
                judge=self.judge,
            ),
            timeout=self.config.judge_window_s,
        )
        self._add_judge_usage(ruling["usage"])
        message = {
            "type": "action_ruling",
            "turn": turn,
            "player": proposer,
            "attempt": attempt,
            "valid": ruling["valid"],
            "source": ruling["source"],
            "summary": ruling["summary"],
            "state_ops": ruling["state_ops"],
            "state": self.board.state.as_dict(),
            "winner_slots": ruling["winner_slots"],
        }
        await self._send_all(message)
        await self._emit(message)
        return action, ruling

    async def _action(self, turn: int, proposer: int) -> dict[str, Any]:
        action, ruling = await self._action_attempt(turn, proposer, attempt=1)
        attempts = [{"action": action, "ruling": ruling}]
        if not ruling["valid"]:
            repaired, repaired_ruling = await self._action_attempt(
                turn,
                proposer,
                attempt=2,
                prior_action=action,
                rejection_reason=ruling["summary"],
            )
            attempts.append({"action": repaired, "ruling": repaired_ruling})
            ruling = repaired_ruling
        self.winner_slots = list(ruling["winner_slots"])
        return {"attempts": attempts, "final_valid": ruling["valid"]}

    async def _debate(
        self, turn: int, proposer: int, proposal: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self.current_phase = "debate"

        async def one(seat: int) -> dict[str, Any]:
            rid = self._next_rid()
            request = {
                "type": "debate_request",
                "turn": turn,
                "rid": rid,
                "timeout_s": self.config.debate_window_s,
                "proposer": proposer,
                "proposal": proposal,
                "votes_required": self.board.state.votes_required(
                    turn=turn, proposal_kind=str(proposal.get("kind", ""))
                ),
            }
            await self._send(self.channels[seat], request)
            raw = await self.channels[seat].recv_reply(rid, self.config.debate_window_s)
            action = self._parse_debate(seat, raw)
            self.capture.consumed(
                seat,
                rid,
                raw,
                {k: v for k, v in action.items() if k not in {"seat", "default"}},
                action["default"],
            )
            return action

        # Both debaters receive the same state and cannot condition on each other.
        return list(
            await asyncio.gather(
                *(one(seat) for seat in range(SEAT_COUNT) if seat != proposer)
            )
        )

    async def _vote(
        self, turn: int, proposal: dict[str, Any], debates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.current_phase = "vote"

        async def one(seat: int) -> dict[str, Any]:
            rid = self._next_rid()
            request = {
                "type": "vote_request",
                "turn": turn,
                "rid": rid,
                "timeout_s": self.config.vote_window_s,
                "proposal": proposal,
                "debates": debates,
                "votes_required": self.board.state.votes_required(
                    turn=turn, proposal_kind=str(proposal.get("kind", ""))
                ),
            }
            await self._send(self.channels[seat], request)
            raw = await self.channels[seat].recv_reply(rid, self.config.vote_window_s)
            action = self._parse_vote(seat, raw)
            self.capture.consumed(
                seat, rid, raw, {"vote": action["vote"]}, action["default"]
            )
            return action

        return list(await asyncio.gather(*(one(seat) for seat in range(SEAT_COUNT))))

    async def _run_turn(self, turn: int) -> None:
        self.current_turn = turn
        proposer = self.board.proposer_for_turn(turn)
        votes_required = self.board.state.votes_required(turn=turn)
        start = {
            "type": "turn_start",
            "turn": turn,
            "proposer": proposer,
            "votes_required": votes_required,
            "rules": self.board.rules_dict(include_history=False),
            "state": self.board.state.as_dict(),
        }
        await self._send_all(start)
        await self._emit({**start, "rules": self.board.rules_dict()})

        action_record = await self._action(turn, proposer)
        if self.winner_slots:
            self.history.append(
                {
                    "turn": turn,
                    "proposer": proposer,
                    "action": action_record,
                    "winner_slots": self.winner_slots,
                }
            )
            return

        proposal = await self._proposal(turn, proposer)
        votes_required = self.board.state.votes_required(
            turn=turn, proposal_kind=str(proposal.get("kind", ""))
        )
        proposal_message = {
            "type": "proposal_made",
            "turn": turn,
            "proposer": proposer,
            "proposal": proposal,
        }
        await self._send_all(proposal_message)
        await self._emit(proposal_message)

        debates = await self._debate(turn, proposer, proposal)
        debate_message = {"type": "debate_made", "turn": turn, "statements": debates}
        await self._send_all(debate_message)
        await self._emit(debate_message)

        votes = await self._vote(turn, proposal, debates)
        passed_vote = sum(vote["vote"] == "aye" for vote in votes) >= votes_required
        reveal = {
            "type": "vote_reveal",
            "turn": turn,
            "votes": votes,
            "votes_required": votes_required,
            "passed": passed_vote,
        }
        await self._send_all(reveal)
        await self._emit(reveal)

        turn_record = {
            "turn": turn,
            "proposer": proposer,
            "action": action_record,
            "proposal": proposal,
            "debates": debates,
            "votes": votes,
            "votes_required": votes_required,
            "passed_vote": passed_vote,
            "host_random": self._host_random(),
        }
        self.current_phase = "judge"
        ruling = await asyncio.wait_for(
            adjudicate(
                self.board,
                turn_record=turn_record,
                turn=turn,
                turns_max=self.config.turns_max,
                judge=self.judge,
            ),
            timeout=self.config.judge_window_s,
        )
        self._add_judge_usage(ruling["usage"])
        turn_record["ruling"] = ruling
        self.history.append(turn_record)
        self.winner_slots = ruling["winner_slots"]
        ruling_message = {
            "type": "judge_ruling",
            "turn": turn,
            "passed_vote": passed_vote,
            "adopted": ruling["adopted"],
            "source": ruling["source"],
            "summary": ruling["summary"],
            "rule_ops": ruling["rule_ops"],
            "state_ops": ruling["state_ops"],
            "host_random": turn_record["host_random"],
            "rules": self.board.rules_dict(include_history=False),
            "state": self.board.state.as_dict(),
            "winner_slots": self.winner_slots,
        }
        await self._send_all(ruling_message)
        await self._emit({**ruling_message, "rules": self.board.rules_dict()})

    def _host_random(self) -> dict[str, Any]:
        entropy = self._rng.getrandbits(63)
        return {
            "entropy": entropy,
            "random_seat": self._rng.randrange(SEAT_COUNT),
            "coin": "heads" if entropy % 2 == 0 else "tails",
            "d6": entropy % 6 + 1,
            "d20": entropy % 20 + 1,
            "d100": entropy % 100 + 1,
        }

    async def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        await self._announce_start()
        for turn in range(1, self.config.turns_max + 1):
            await self._run_turn(turn)
            if self.winner_slots:
                self.termination = "constitution_victory"
                break
        if not self.winner_slots:
            self.winner_slots = self.board.cap_victors()
            self.termination = "turn_cap"

        winner_share = 1.0 / len(self.winner_slots)
        scores = [
            winner_share if seat in self.winner_slots else 0.0
            for seat in range(SEAT_COUNT)
        ]
        reason = (
            "Gnome Law declared victory."
            if self.termination == "constitution_victory"
            else "The host turn cap was reached; gnomes tied for most points split one win."
        )
        game_over = {
            "type": "game_over",
            "winner_slots": self.winner_slots,
            "reason": reason,
            "scores": scores,
            "game_points": self.board.state.game_points(),
        }
        await self._send_all(game_over)
        await self._emit(game_over)
        self.current_phase = "done"
        results = {
            "scores": scores,
            "game_points": self.board.state.game_points(),
            "winner_slots": self.winner_slots,
            "turns_played": len(self.history),
            "termination": self.termination,
        }
        replay = {
            "format": "gnomic-replay-v1",
            "seed": self.seed,
            "players": self._seats(attributed=True),
            "events": self.events,
            "turns": self.history,
            "final_board": self.board.as_dict(),
            "judge_usage": self.judge_usage,
            "results": results,
        }
        self.results = results
        return results, replay

    def snapshot_for(self, seat: int) -> dict[str, Any]:
        return {
            "type": "game_start",
            "session": {
                "id": f"gnomic-{self.seed:08x}",
                "seats": self._seats(),
                "limits": self.config.limits_payload(),
            },
            "you": {"seat": seat},
            "host_constraints": list(HOST_CONSTRAINTS),
            "rules": self.board.rules_dict(include_history=False),
            "state": self.board.state.as_dict(),
            "history": self.history[-12:],
        }

    def operator_log(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "turn": self.current_turn,
            "phase": self.current_phase,
            "judge_usage": self.judge_usage,
        }
