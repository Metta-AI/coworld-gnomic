"""Typed public wire protocol shared by the game and player images."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr, ValidationError


class _Msg(BaseModel):
    model_config = ConfigDict(extra="allow")


class Lobby(_Msg):
    type: Literal["lobby"]
    seat: int


class GameStart(_Msg):
    type: Literal["game_start"]
    session: dict[str, Any]
    you: dict[str, Any]
    host_constraints: list[str]
    rules: list[dict[str, Any]]
    state: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)


class TurnStart(_Msg):
    type: Literal["turn_start"]
    turn: int
    proposer: int
    votes_required: int
    rules: list[dict[str, Any]]
    state: dict[str, Any]


class ActionRequest(_Msg):
    type: Literal["action_request"]
    turn: int
    rid: int
    timeout_s: float
    attempt: int = 1


class ActionRepairRequest(_Msg):
    type: Literal["action_repair_request"]
    turn: int
    rid: int
    timeout_s: float
    attempt: int = 2
    original_action: dict[str, Any]
    rejection_reason: str


class ActionMade(_Msg):
    type: Literal["action_made"]
    turn: int
    player: int
    attempt: int
    action: dict[str, Any]


class ActionRuling(_Msg):
    type: Literal["action_ruling"]
    turn: int
    player: int
    attempt: int
    valid: bool
    source: str
    summary: str
    state_ops: list[dict[str, Any]]
    state: dict[str, Any]
    winner_slots: list[int] = Field(default_factory=list)


class ProposalRequest(_Msg):
    type: Literal["proposal_request"]
    turn: int
    rid: int
    timeout_s: float


class ProposalMade(_Msg):
    type: Literal["proposal_made"]
    turn: int
    proposer: int
    proposal: dict[str, Any]


class DebateRequest(_Msg):
    type: Literal["debate_request"]
    turn: int
    rid: int
    timeout_s: float
    proposer: int
    proposal: dict[str, Any]


class DebateMade(_Msg):
    type: Literal["debate_made"]
    turn: int
    statements: list[dict[str, Any]]


class VoteRequest(_Msg):
    type: Literal["vote_request"]
    turn: int
    rid: int
    timeout_s: float
    proposal: dict[str, Any]
    debates: list[dict[str, Any]]


class VoteReveal(_Msg):
    type: Literal["vote_reveal"]
    turn: int
    votes: list[dict[str, Any]]
    votes_required: int
    passed: bool


class JudgeRuling(_Msg):
    type: Literal["judge_ruling"]
    turn: int
    passed_vote: bool
    adopted: bool
    source: str
    summary: str
    rule_ops: list[dict[str, Any]]
    state_ops: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    state: dict[str, Any]
    winner_slots: list[int] = Field(default_factory=list)


class GameOver(_Msg):
    type: Literal["game_over"]
    winner_slots: list[int]
    reason: str
    scores: list[float]
    game_points: list[int]


class Final(_Msg):
    type: Literal["final"]
    scores: list[float] | None = None


class Snapshot(_Msg):
    type: Literal["snapshot"]
    turn: int
    phase: str


class _StopControl(_Msg):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    stop_id: StrictStr = Field(min_length=1, max_length=128)


class Stop(_StopControl):
    type: Literal["stop"] = "stop"


class Stopped(_StopControl):
    type: Literal["stopped"] = "stopped"


SERVER_MESSAGES: dict[str, type[_Msg]] = {
    "lobby": Lobby,
    "game_start": GameStart,
    "turn_start": TurnStart,
    "action_request": ActionRequest,
    "action_repair_request": ActionRepairRequest,
    "action_made": ActionMade,
    "action_ruling": ActionRuling,
    "proposal_request": ProposalRequest,
    "proposal_made": ProposalMade,
    "debate_request": DebateRequest,
    "debate_made": DebateMade,
    "vote_request": VoteRequest,
    "vote_reveal": VoteReveal,
    "judge_ruling": JudgeRuling,
    "game_over": GameOver,
    "final": Final,
    "stop": Stop,
    "snapshot": Snapshot,
}


def parse_server_message(raw: dict[str, Any]) -> _Msg | None:
    model = SERVER_MESSAGES.get(raw.get("type", ""))
    if model is None:
        return None
    return model.model_validate(raw)


class ParsedReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["parsed", "invalid"]
    value: dict[str, JsonValue] | None = None
    error: str | None = None


def parse_reply_text(text: str) -> ParsedReply:
    """Player text is untrusted protocol data; expose an explicit parse outcome."""
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            return ParsedReply(kind="invalid", error="reply-must-be-object")
        return ParsedReply(kind="parsed", value=value)
    except (json.JSONDecodeError, ValidationError) as exc:
        return ParsedReply(kind="invalid", error=type(exc).__name__)


def make_reply(rid: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {"rid": rid, **payload}
