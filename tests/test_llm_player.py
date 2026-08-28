from __future__ import annotations

import asyncio
import io
import json

import pytest

from gnomic.players.llm import ActionOutput, DebateOutput, OpusPolicy, ProposalOutput, VoteOutput, normalize_model_payload


class FakeBedrock:
    def __init__(self, text: str = '{"kind":"enact"}') -> None:
        self.body: dict | None = None
        self.text = text

    def invoke_model(self, *, modelId: str, body: str):
        self.body = json.loads(body)
        payload = {
            "content": [{"type": "text", "text": self.text}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        return {"body": io.BytesIO(json.dumps(payload).encode())}


def test_opus_player_request_uses_adaptive_high_with_task_budget() -> None:
    fake = FakeBedrock()
    policy = OpusPolicy("ivan", client=fake)
    assert policy._invoke("system", "prompt") == '{"kind":"enact"}'
    assert fake.body is not None
    assert fake.body["max_tokens"] == 32768
    assert fake.body["temperature"] == 1
    assert fake.body["thinking"] == {"type": "adaptive"}
    assert fake.body["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 20000},
    }
    assert fake.body["anthropic_beta"] == ["task-budgets-2026-03-13"]
    assert policy.usage == {"calls": 1, "input_tokens": 3, "output_tokens": 4}


def test_common_json_key_synonyms_normalize_before_strict_validation() -> None:
    context = {
        "adoption_outcome": "The game continues.",
        "rejection_outcome": "The game continues.",
        "game_ends_if_adopted": False,
        "my_outcome_if_adopted": "continues",
        "my_outcome_if_rejected": "continues",
    }
    action = normalize_model_payload({"text": "I ring the bell."}, ActionOutput)
    assert ActionOutput.model_validate(action).action == "I ring the bell."
    debate = normalize_model_payload(
        {**context, "public_statement": "This helps us.", "vote_intent": "aye"}, DebateOutput
    )
    assert DebateOutput.model_validate(debate).text == "This helps us."
    invented = normalize_model_payload(
        {**context, "case_against": "This helps only the proposer.", "vote_intent": "nay"}, DebateOutput
    )
    assert DebateOutput.model_validate(invented).text == "This helps only the proposer."
    proposal = normalize_model_payload(
        {
            "proposal": {
                "action": "enact",
                "rule_text": "Tea is served.",
                "rationale": "x" * 2_100,
                "default": False,
            }
        },
        ProposalOutput,
    )
    parsed = ProposalOutput.model_validate(proposal)
    assert parsed.kind == "enact"
    assert parsed.text == "Tea is served."
    assert len(parsed.rationale) == 2_000
    assert parsed.rationale.endswith("…")

    vote = normalize_model_payload(
        {
            **context,
            "vote": "nay",
            "reason": "The proposal creates an immediate loss.",
            "my_intent": "nay",
            "my_intent_matches": True,
        },
        VoteOutput,
    )
    assert VoteOutput.model_validate(vote).vote == "nay"
    assert "my_intent" not in vote
    assert "my_intent_matches" not in vote


def test_debate_rejects_explicit_vote_contradiction() -> None:
    with pytest.raises(ValueError, match="opposite vote"):
        DebateOutput(
            adoption_outcome="Seat 1 wins immediately.",
            rejection_outcome="All seats remain tied.",
            game_ends_if_adopted=True,
            my_outcome_if_adopted="loss",
            my_outcome_if_rejected="co_win",
            text="This makes seat 1 win at my expense, so I vote NAY.",
            vote_intent="aye",
        )


def test_vote_rejects_immediate_own_loss() -> None:
    with pytest.raises(ValueError, match="immediate own loss"):
        VoteOutput(
            adoption_outcome="Seat 1 wins immediately.",
            rejection_outcome="All seats remain tied.",
            game_ends_if_adopted=True,
            my_outcome_if_adopted="loss",
            my_outcome_if_rejected="co_win",
            vote="aye",
            reason="Support the opponent anyway.",
        )


@pytest.mark.asyncio
async def test_game_start_carries_host_constraints_into_strategic_system_prompt() -> None:
    policy = OpusPolicy("ivan", client=FakeBedrock())
    await policy.respond(
        {
            "type": "game_start",
            "you": {"seat": 0},
            "session": {"limits": {"turns_max": 45}},
            "host_constraints": ["Every turn uses parallel debate."],
            "rules": [],
            "state": {"players": [], "common": {}},
            "history": [],
        }
    )

    assert "Every turn uses parallel debate." in policy._system()
    assert "no mutable debate order" in policy._system()
    assert "may omit or repeat seats" in policy._system()
    assert "co-designer" in policy._system()
    assert "one repair attempt" in policy._system()


@pytest.mark.asyncio
async def test_final_vote_reasons_fresh_instead_of_reusing_debate_intent() -> None:
    fake = FakeBedrock(
        json.dumps(
            {
                "adoption_outcome": "Seat 1 reaches the threshold and wins immediately.",
                "rejection_outcome": "The game continues to the final turn.",
                "game_ends_if_adopted": True,
                "my_outcome_if_adopted": "loss",
                "my_outcome_if_rejected": "continues",
                "vote": "nay",
                "reason": "Adoption would immediately eliminate my winning path.",
            }
        )
    )
    policy = OpusPolicy("ivan", client=fake)
    policy.seat = 0
    policy.proposer = 1
    policy.turns_max = 9
    policy.board = {
        "rules": [],
        "state": {
            "players": [{"points": 0}, {"points": 0}, {"points": 0}],
            "common": {"proposer_order": [0, 1, 2], "victory_points": 10},
        },
    }
    policy.vote_intents[8] = "aye"

    reply = await policy.respond(
        {
            "type": "vote_request",
            "turn": 8,
            "rid": 99,
            "proposal": {"kind": "amend", "rule_id": 203, "text": "Ten points wins."},
            "debates": [],
        }
    )

    assert reply == {
        "rid": 99,
        "vote": "nay",
        "reason": "Adoption would immediately eliminate my winning path.",
    }
    assert policy.usage["calls"] == 1
    assert policy._strategic_context(8, 1)["next_proposer_schedule"] == [
        {"turn": 8, "proposer": 1},
        {"turn": 9, "proposer": 2},
    ]


def test_strategic_schedule_uses_variable_order_and_public_cursor() -> None:
    policy = OpusPolicy("ivan", client=FakeBedrock())
    policy.seat = 0
    policy.proposer = 2
    policy.turns_max = 9
    policy.board = {
        "state": {
            "players": [{"points": 4}, {"points": 10}, {"points": 14}],
            "common": {"proposer_order": [2, 0], "proposer_cursor": 1},
        }
    }
    assert policy._strategic_context(4, 2)["next_proposer_schedule"][:4] == [
        {"turn": 4, "proposer": 2},
        {"turn": 5, "proposer": 0},
        {"turn": 6, "proposer": 2},
        {"turn": 7, "proposer": 0},
    ]


def test_reconnect_history_is_compacted_before_future_prompts() -> None:
    policy = OpusPolicy("yura", client=FakeBedrock())
    policy._load_history(
        [
            {
                "turn": 12,
                "proposer": 2,
                "action": {
                    "attempts": [
                        {
                            "action": {"text": "A" * 1_500},
                            "ruling": {"valid": True, "summary": "Done.", "state_ops": []},
                        }
                    ]
                },
                "proposal": {"kind": "enact", "text": "P" * 1_500, "rationale": "R" * 1_500},
                "debates": [{"seat": 0, "text": "D" * 1_500, "vote_intent": "aye"}],
                "votes": [{"seat": 0, "vote": "aye", "reason": "V" * 1_500}],
                "passed_vote": True,
                "ruling": {"adopted": True, "summary": "Adopted.", "winner_slots": []},
            }
        ]
    )

    serialized = json.dumps(policy.history)
    assert len(serialized) < 5_000
    assert any(item["type"] == "judge_ruling" for item in policy.history)


def test_policy_introduces_itself_with_its_gnome_name() -> None:
    policy = OpusPolicy("anton", client=FakeBedrock())
    reply = asyncio.run(policy.respond({"type": "introduce_request", "rid": 7}))
    assert reply == {"rid": 7, "name": "Anton"}
