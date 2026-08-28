from __future__ import annotations

import copy

import pytest

from gnomic.engine import Board, HOST_CONSTRAINTS, OperationError, theme_for_seed


def test_initial_board_is_compact_original_inspired_gnomic() -> None:
    board = Board.initial()
    assert len(board.active_rules()) == 13
    assert [rule.id for rule in board.active_rules() if not rule.mutable] == [101, 102, 103, 104, 105]
    assert board.state.game_points() == [0, 0, 0]
    assert board.state.votes_required(turn=1) == 3
    assert board.state.votes_required(turn=7) == 2
    assert board.state.victory_points() == 100
    assert board.state.proposal_points(adopted=True) == 3
    assert board.state.proposal_points(adopted=False) == -1
    assert board.state.proposer_order() == [0, 1, 2]
    assert board.state.proposer_cursor() == 0
    assert board.state.common["muse"] == ["weather", "soup"]
    assert len(HOST_CONSTRAINTS) == 6


def test_rule_ids_stay_stable_across_amend_and_repeal() -> None:
    board = Board.initial()
    board.apply_ops_atomic([{"op": "amend", "rule_id": 201, "text": "Reverse order."}], [], turn=1)
    rule = board.find_rule(201)
    assert rule is not None
    assert rule.version == 2
    assert rule.history[0]["text"].startswith("Turns use")
    board.apply_ops_atomic([{"op": "repeal", "rule_id": 201}], [], turn=2)
    assert board.find_rule(201).active is False  # type: ignore[union-attr]


def test_nested_json_state_and_mutable_runtime_law() -> None:
    board = Board.initial()
    board.apply_ops_atomic(
        [],
        [
            {"op": "set", "scope": "common", "key": "coalitions", "value": {"north": [0, 2]}},
            {"op": "set", "scope": "common", "key": "votes_required", "value": 3},
            {"op": "set", "scope": "common", "key": "proposer_order", "value": [2, 1, 0]},
        ],
        turn=1,
    )
    assert board.state.common["coalitions"] == {"north": [0, 2]}
    assert board.state.votes_required() == 3
    assert board.proposer_for_turn(1) == 2
    assert board.state.proposer_cursor() == 1


def test_proposer_sequence_may_exclude_or_repeat_seats() -> None:
    board = Board.initial()
    board.proposer_for_turn(1)
    board.apply_ops_atomic(
        [],
        [{"op": "set", "scope": "common", "key": "proposer_order", "value": [1, 0]}],
        turn=1,
    )
    assert board.state.proposer_cursor() == 0
    assert [board.proposer_for_turn(turn) for turn in range(2, 6)] == [1, 0, 1, 0]

    board.apply_ops_atomic(
        [],
        [{"op": "set", "scope": "common", "key": "proposer_order", "value": [2, 2, 0]}],
        turn=5,
    )
    assert [board.proposer_for_turn(turn) for turn in range(6, 10)] == [2, 2, 0, 2]


def test_protected_rule_must_be_transmuted_before_amendment() -> None:
    board = Board.initial()
    with pytest.raises(OperationError, match="must be transmuted"):
        board.apply_ops_atomic(
            [{"op": "amend", "rule_id": 104, "text": "Actions are abolished."}], [], turn=1
        )
    board.apply_ops_atomic([{"op": "transmute", "rule_id": 104}], [], turn=1)
    assert board.find_rule(104).mutable is True  # type: ignore[union-attr]
    board.apply_ops_atomic(
        [{"op": "amend", "rule_id": 104, "text": "Actions are abolished."}], [], turn=2
    )
    assert board.find_rule(104).version == 3  # type: ignore[union-attr]


def test_theme_seed_is_deterministic_and_varied() -> None:
    assert theme_for_seed(0) == ["weather", "soup"]
    assert theme_for_seed(1) == ["lanterns", "memory"]
    assert Board.initial(seed=9).state.common["muse"] == ["lanterns", "memory"]


def test_point_victory_is_checked_only_at_circuit_end() -> None:
    board = Board.initial()
    board.state.players[0]["points"] = 100
    assert board.point_victors(turn=2) == []
    assert board.point_victors(turn=3) == [0]


def test_next_proposer_override_does_not_consume_regular_sequence() -> None:
    board = Board.initial()
    board.state.common["next_proposer"] = 2
    assert board.proposer_for_turn(1) == 2
    assert board.state.proposer_cursor() == 0
    assert board.proposer_for_turn(2) == 0


def test_operations_are_atomic() -> None:
    board = Board.initial()
    before = copy.deepcopy(board.as_dict())
    with pytest.raises(OperationError, match="points"):
        board.apply_ops_atomic(
            [{"op": "enact", "text": "A temporary rule."}],
            [{"op": "set", "scope": "player", "seat": 0, "key": "points", "value": "many"}],
            turn=1,
        )
    assert board.as_dict() == before


def test_cap_victors_support_co_winners() -> None:
    board = Board.initial()
    board.state.players[0]["points"] = 20
    board.state.players[1]["points"] = 20
    board.state.players[2]["points"] = 10
    assert board.cap_victors() == [0, 1]


@pytest.mark.parametrize("value", [0, 4, "2", [], [0, 3], [True], list(range(10))])
def test_runtime_control_fields_are_guarded(value: object) -> None:
    board = Board.initial()
    key = "proposer_order" if isinstance(value, list) else "votes_required"
    with pytest.raises(OperationError):
        board.apply_ops_atomic([], [{"op": "set", "scope": "common", "key": key, "value": value}], turn=1)


def test_proposer_cursor_is_host_managed() -> None:
    board = Board.initial()
    with pytest.raises(OperationError, match="managed by the host"):
        board.apply_ops_atomic(
            [],
            [{"op": "set", "scope": "common", "key": "proposer_cursor", "value": 2}],
            turn=1,
        )


@pytest.mark.parametrize("value", [True, 2.5, "4"])
def test_proposal_points_are_guarded(value: object) -> None:
    board = Board.initial()
    with pytest.raises(OperationError):
        board.apply_ops_atomic(
            [],
            [{"op": "set", "scope": "common", "key": "points_per_adopted_proposal", "value": value}],
            turn=1,
        )
