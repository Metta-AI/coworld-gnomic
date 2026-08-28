"""Typed public wire protocol shared by the game and player images."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    "snapshot": Snapshot,
}


def parse_server_message(raw: dict[str, Any]) -> _Msg | None:
    model = SERVER_MESSAGES.get(raw.get("type", ""))
    if model is None:
        return None
    try:
        return model.model_validate(raw)
    except Exception:
        return None


def make_reply(rid: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {"rid": rid, **payload}
