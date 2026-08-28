from __future__ import annotations

import io
import json

import pytest

from gnomic.engine import Board
from gnomic.judge import ACTION_JUDGE_SYSTEM, JUDGE_SYSTEM, BedrockJudge, DeterministicJudge, JudgeError, adjudicate, adjudicate_action


def test_judge_must_preserve_exact_proposer_sequence() -> None:
    assert "Never pad, reorder, or normalize it" in JUDGE_SYSTEM
    assert "host-managed proposer_cursor" in JUDGE_SYSTEM
    assert "Never rewrite, clarify, complete, or" in JUDGE_SYSTEM


def record(*, passed: bool = True) -> dict:
    return {
        "turn": 1,
        "proposer": 0,
        "proposal": {"kind": "enact", "text": "Tea is encouraged."},
        "debates": [],
        "votes": [],
        "votes_required": 2,
        "passed_vote": passed,
        "host_random": {"entropy": 5, "random_seat": 2},
    }


@pytest.mark.asyncio
async def test_deterministic_judge_applies_starting_law() -> None:
    board = Board.initial()
    ruling = await adjudicate(board, turn_record=record(), turn=1, turns_max=9, judge=DeterministicJudge())
    assert ruling["adopted"] is True
    assert board.state.game_points() == [3, 0, 6]
    assert board.find_rule(209).text == "Tea is encouraged."  # type: ignore[union-attr]


class FakeBedrock:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.bodies: list[dict] = []

    def invoke_model(self, *, modelId: str, body: str):
        self.bodies.append(json.loads(body))
        payload = self.payloads.pop(0)
        return {"body": io.BytesIO(json.dumps(payload).encode())}


def response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 4, "output_tokens": 5}}


@pytest.mark.asyncio
async def test_bedrock_judge_repairs_bad_json_once_and_uses_adaptive_high() -> None:
    fixed = {
        "valid": True,
        "adopted": True,
        "summary": "Adopted and scored.",
        "rule_ops": [{"op": "enact", "text": "Tea is encouraged.", "explanation": "harmless"}],
        "state_ops": [
            {
                "op": "increment",
                "scope": "player",
                "seat": 0,
                "key": "points",
                "value": 10,
                "arithmetic_check": "0 + 10 = 10",
            }
        ],
        "winner_slots": [],
        "analysis_complete": True,
    }
    fake = FakeBedrock([response("not json"), response(json.dumps(fixed))])
    board = Board.initial()
    ruling = await adjudicate(board, turn_record=record(), turn=1, turns_max=9, judge=BedrockJudge(client=fake))
    assert ruling["usage"]["calls"] == 2
    assert board.state.points(0) == 10
    assert fake.bodies[0]["thinking"] == {"type": "adaptive"}
    assert fake.bodies[0]["max_tokens"] == 32768
    assert fake.bodies[0]["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 20000},
    }
    assert fake.bodies[0]["anthropic_beta"] == ["task-budgets-2026-03-13"]


@pytest.mark.asyncio
async def test_bedrock_judge_never_adopts_failed_vote() -> None:
    bad = {
        "valid": True,
        "adopted": True,
        "summary": "Wrong.",
        "rule_ops": [{"op": "enact", "text": "Wrong."}],
        "state_ops": [],
        "winner_slots": [],
    }
    fake = FakeBedrock([response(json.dumps(bad)), response(json.dumps(bad))])
    with pytest.raises(JudgeError, match="failed vote"):
        await BedrockJudge(client=fake).rule(Board.initial(), record(passed=False), 1, 9)


@pytest.mark.asyncio
async def test_bedrock_judge_repairs_rewritten_rule_text() -> None:
    rewritten = {
        "valid": True,
        "adopted": True,
        "summary": "Rewritten.",
        "rule_ops": [{"op": "enact", "text": "Tea is encouraged. Tea is mandatory."}],
        "state_ops": [],
        "winner_slots": [],
    }
    exact = {
        **rewritten,
        "summary": "Copied exactly.",
        "rule_ops": [{"op": "enact", "text": "Tea is encouraged."}],
    }
    fake = FakeBedrock([response(json.dumps(rewritten)), response(json.dumps(exact))])
    ruling, usage = await BedrockJudge(client=fake).rule(Board.initial(), record(), 1, 9)
    assert ruling.rule_dicts() == [{"op": "enact", "text": "Tea is encouraged."}]
    assert usage["calls"] == 2


@pytest.mark.asyncio
async def test_action_judge_interprets_natural_language_as_bounded_state_only() -> None:
    action = {
        "valid": True,
        "summary": "Seat 1 spends a key.",
        "state_ops": [
            {"op": "increment", "scope": "player", "seat": 1, "key": "keys", "value": -1}
        ],
        "winner_slots": [],
    }
    fake = FakeBedrock([response(json.dumps(action))])
    board = Board.initial()
    board.state.players[1]["keys"] = 1

    ruling = await adjudicate_action(
        board,
        action_record={"player": 1, "text": "I spend my key."},
        turn=4,
        turns_max=45,
        judge=BedrockJudge(client=fake),
    )

    assert ruling["valid"] is True
    assert board.state.players[1]["keys"] == 0
    assert fake.bodies[0]["system"] == ACTION_JUDGE_SYSTEM
    assert "alter a rule" in ACTION_JUDGE_SYSTEM
