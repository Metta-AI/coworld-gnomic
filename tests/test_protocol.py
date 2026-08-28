from __future__ import annotations

from gnomic.protocol import SERVER_MESSAGES, parse_server_message


def test_all_declared_message_types_have_models() -> None:
    assert set(SERVER_MESSAGES) == {
        "lobby", "game_start", "turn_start", "proposal_request", "proposal_made",
        "action_request", "action_repair_request", "action_made", "action_ruling",
        "debate_request", "debate_made", "vote_request", "vote_reveal",
        "judge_ruling", "game_over", "final", "snapshot",
    }


def test_unknown_additive_message_is_forward_compatible() -> None:
    assert parse_server_message({"type": "future_message"}) is None
    parsed = parse_server_message({"type": "lobby", "seat": 1, "future": True})
    assert parsed is not None
    assert parsed.model_extra == {"future": True}
