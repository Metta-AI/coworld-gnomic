"""Validated concrete episode configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..engine import HARD_TURN_CAP, SEAT_COUNT


class PlayerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[str] = Field(min_length=SEAT_COUNT, max_length=SEAT_COUNT)
    players: list[PlayerEntry] = Field(min_length=SEAT_COUNT, max_length=SEAT_COUNT)
    num_agents: Literal[SEAT_COUNT] = SEAT_COUNT
    turns_max: int = Field(default=HARD_TURN_CAP, ge=1, le=HARD_TURN_CAP)
    action_window_s: float = Field(default=600.0, ge=0.05, le=600)
    proposal_window_s: float = Field(default=600.0, ge=0.05, le=600)
    debate_window_s: float = Field(default=600.0, ge=0.05, le=600)
    vote_window_s: float = Field(default=600.0, ge=0.05, le=600)
    judge_window_s: float = Field(default=600.0, ge=1, le=600)
    judge_mode: Literal["bedrock", "deterministic"] = "bedrock"
    judge_model: str = "us.anthropic.claude-opus-4-8"
    seed: int | None = None
    player_connect_timeout_seconds: float = Field(default=180.0, ge=0, le=600)
    episode_timeout_seconds: float = Field(default=5_400.0, ge=60, le=7_200)

    @model_validator(mode="after")
    def require_unique_tokens(self) -> "GameConfig":
        if len(set(self.tokens)) != SEAT_COUNT or any(not token for token in self.tokens):
            raise ValueError("tokens must contain three distinct non-empty values")
        return self

    def seat_count(self) -> int:
        return SEAT_COUNT

    def limits_payload(self) -> dict:
        return {
            "turns_max": self.turns_max,
            "action_window_s": self.action_window_s,
            "proposal_window_s": self.proposal_window_s,
            "debate_window_s": self.debate_window_s,
            "vote_window_s": self.vote_window_s,
            "judge_window_s": self.judge_window_s,
        }
