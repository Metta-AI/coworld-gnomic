"""Pure, deterministic state machine for Gnomic.

The engine owns the parts an LLM must never be allowed to bypass: three seats,
bounded JSON state, stable rule identifiers, atomic operations, and the hard
turn cap.  The constitution itself is data and can be amended, repealed, or
unlocked from protected status.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

SEAT_COUNT = 3
HARD_TURN_CAP = 45
MAX_RULES = 64
MAX_RULE_TEXT = 2_000
MAX_KEY_LENGTH = 80
MAX_VALUE_DEPTH = 6
MAX_VALUE_BYTES = 16_384
MAX_STATE_BYTES = 65_536
MAX_OPS_PER_RULING = 24
MAX_PROPOSER_SEQUENCE = 12

THEME_SEEDS = [
    ("weather", "soup"),
    ("lanterns", "memory"),
    ("mushrooms", "ritual"),
    ("burrows", "inheritance"),
    ("harvest", "bureaucracy"),
    ("gardens", "reputation"),
    ("badgers", "etiquette"),
    ("curfew", "hospitality"),
]

HOST_CONSTRAINTS = [
    "Exactly three player seats exist for the entire episode.",
    "Every turn uses action, proposal, parallel debate, secret vote, and judge-ruling phases.",
    "Only an adopted proposal may alter the rulebook; actions may alter bounded public state only.",
    "No rule, action, or ruling may execute code, access secrets, perform I/O, or alter the host protocol.",
    "The episode ends after at most 45 completed turns.",
    "All public state and rulings must fit the bounded JSON schemas enforced by the host.",
]


class OperationError(ValueError):
    """A judge operation was structurally valid but unsafe or inapplicable."""


@dataclass
class Rule:
    id: int
    text: str
    created_turn: int = 0
    version: int = 1
    active: bool = True
    mutable: bool = True
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "created_turn": self.created_turn,
            "version": self.version,
            "active": self.active,
            "mutable": self.mutable,
            "history": copy.deepcopy(self.history),
        }


def initial_rules() -> list[Rule]:
    """A compact, original-Gnomic-inspired starting constitution."""
    protected = [
        "All players and the Judge must obey every active rule. When active rules conflict, the lower-numbered rule takes precedence unless another active rule explicitly resolves that conflict.",
        "A proposal changes exactly one rule: it may enact one new mutable rule, amend or repeal one active mutable rule, or transmute one active protected rule into a mutable rule. Rule changes take effect in the ruling that adopts them.",
        "A protected rule cannot be amended or repealed. A proposal to transmute a protected rule requires all three AYE votes and does nothing except make that rule mutable; a later proposal is required to amend or repeal it.",
        "The active player may state one natural-language game action before proposing. The action changes public state only when authorized by active rules. If rejected as illegal or unclear, that player receives one repair attempt; passing is always legal.",
        "The Judge interprets the active Gnome Law faithfully. It may reject only an illegal, unclear, incoherent, impossible, or host-unsafe action or proposal, never one it merely considers unwise.",
    ]
    mutable = [
        "Turns use the repeating proposer order [0, 1, 2]. A valid amendment to proposer order takes effect next turn at the first listed seat; the non-empty sequence may contain up to twelve seat numbers and may omit or repeat seats.",
        "A proposal needs three AYE votes during turns 1 through 6. Beginning on turn 7 it needs two AYE votes, unless this rule has changed. The two non-proposers debate in parallel and all three players then vote secretly.",
        "An adopted proposal gives its proposer 3 points. A proposal that fails its vote or is vetoed costs its proposer 1 point. No player receives points merely for voting AYE.",
        "After each proposal ruling, Fate rolls one six-sided die and independently selects one of the three seats uniformly at random; the selected player gains the roll in points. The host supplies and records the random draws, and the Judge applies this rule after any newly adopted amendment.",
        "Players begin with 0 points. At the end of each complete three-turn circuit, every player with at least 100 points wins; players reaching the threshold in that same circuit co-win. There is no point victory check between circuit ends.",
        "The action phase may be used for any move that an active rule authorizes. An action cannot itself enact, amend, repeal, or transmute a rule, and a newly adopted rule cannot retroactively authorize the action earlier in that turn.",
        "Each game has two public muse words. They are nonbinding creative prompts: players should use, combine, subvert, or ignore them while trying to evolve a coherent and entertaining game.",
        "If no rule has produced a winner by the 45-turn host cap, every player tied for the most points co-wins. Co-winners split one tournament win point equally.",
    ]
    rules = [
        Rule(id=101 + index, text=text, mutable=False)
        for index, text in enumerate(protected)
    ]
    rules.extend(Rule(id=201 + index, text=text) for index, text in enumerate(mutable))
    return rules


def theme_for_seed(seed: int) -> list[str]:
    """Select a deterministic, nonbinding muse pair without consuming game RNG."""
    return list(THEME_SEEDS[abs(seed) % len(THEME_SEEDS)])


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_VALUE_DEPTH:
        raise OperationError(f"JSON value exceeds maximum depth {MAX_VALUE_DEPTH}")
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise OperationError("non-finite numbers are not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
                raise OperationError("object keys must be short non-empty strings")
            _validate_json_value(item, depth=depth + 1)
        return
    raise OperationError(f"unsupported JSON value type: {type(value).__name__}")


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _valid_proposer_order(value: Any) -> bool:
    return (
        isinstance(value, list)
        and 1 <= len(value) <= MAX_PROPOSER_SEQUENCE
        and all(
            isinstance(seat, int) and not isinstance(seat, bool) and 0 <= seat < SEAT_COUNT
            for seat in value
        )
    )


@dataclass
class WorldState:
    players: list[dict[str, Any]]
    common: dict[str, Any]

    @classmethod
    def initial(cls, *, seed: int = 0) -> "WorldState":
        return cls(
            players=[{"points": 0} for _ in range(SEAT_COUNT)],
            common={
                "votes_required": 3,
                "majority_begins_turn": 7,
                "transmutation_votes_required": 3,
                "proposer_order": [0, 1, 2],
                "proposer_cursor": 0,
                "points_per_adopted_proposal": 3,
                "points_per_rejected_proposal": -1,
                "victory_points": 100,
                "victory_check_every": 3,
                "fate_die_sides": 6,
                "fate_recipient": "random",
                "muse": theme_for_seed(seed),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {"players": copy.deepcopy(self.players), "common": copy.deepcopy(self.common)}

    def points(self, seat: int) -> int:
        value = self.players[seat].get("points", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def game_points(self) -> list[int]:
        return [self.points(seat) for seat in range(SEAT_COUNT)]

    def votes_required(self, *, turn: int | None = None, proposal_kind: str | None = None) -> int:
        value = self.common.get("votes_required", 2)
        if isinstance(value, int) and not isinstance(value, bool):
            required = min(SEAT_COUNT, max(1, value))
        else:
            required = 2
        majority_turn = self.common.get("majority_begins_turn")
        if (
            turn is not None
            and isinstance(majority_turn, int)
            and not isinstance(majority_turn, bool)
            and turn >= majority_turn
        ):
            required = min(required, 2)
        if proposal_kind == "transmute":
            transmutation = self.common.get("transmutation_votes_required", 3)
            if isinstance(transmutation, int) and not isinstance(transmutation, bool):
                required = max(required, min(SEAT_COUNT, max(1, transmutation)))
        return required

    def victory_points(self) -> int:
        value = self.common.get("victory_points", 100)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        return 100

    def proposal_points(self, *, adopted: bool) -> int:
        key = "points_per_adopted_proposal" if adopted else "points_per_rejected_proposal"
        fallback = 3 if adopted else -1
        value = self.common.get(key, fallback)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return fallback

    def victory_check_every(self) -> int:
        value = self.common.get("victory_check_every", 3)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        return 3

    def victory_check_due(self, turn: int) -> bool:
        return turn % self.victory_check_every() == 0

    def proposer_order(self) -> list[int]:
        value = self.common.get("proposer_order")
        if _valid_proposer_order(value):
            return list(value)
        return list(range(SEAT_COUNT))

    def proposer_cursor(self) -> int:
        order = self.proposer_order()
        value = self.common.get("proposer_cursor", 0)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(order):
            return value
        return 0


@dataclass
class Board:
    rules: list[Rule]
    state: WorldState
    next_rule_id: int = 209

    @classmethod
    def initial(cls, *, seed: int = 0) -> "Board":
        return cls(rules=initial_rules(), state=WorldState.initial(seed=seed))

    def as_dict(self) -> dict[str, Any]:
        return {
            "host_constraints": list(HOST_CONSTRAINTS),
            "rules": self.rules_dict(),
            "state": self.state.as_dict(),
            "next_rule_id": self.next_rule_id,
        }

    def rules_dict(self, *, include_history: bool = True) -> list[dict[str, Any]]:
        rules = [rule.as_dict() for rule in self.rules]
        if not include_history:
            for rule in rules:
                rule["history"] = []
        return rules

    def active_rules(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.active]

    def find_rule(self, rule_id: int, *, active_only: bool = False) -> Rule | None:
        for rule in self.rules:
            if rule.id == rule_id and (rule.active or not active_only):
                return rule
        return None

    def _apply_rule_op(self, op: dict[str, Any], turn: int) -> None:
        kind = op.get("op")
        if kind == "enact":
            text = op.get("text")
            if not isinstance(text, str) or not text.strip():
                raise OperationError("enact requires non-empty text")
            if len(text) > MAX_RULE_TEXT:
                raise OperationError(f"rule text exceeds {MAX_RULE_TEXT} characters")
            if len(self.rules) >= MAX_RULES:
                raise OperationError("rulebook is full")
            self.rules.append(Rule(id=self.next_rule_id, text=text.strip(), created_turn=turn))
            self.next_rule_id += 1
            return

        if kind == "transmute":
            rule_id = op.get("rule_id")
            if not isinstance(rule_id, int):
                raise OperationError("transmute requires integer rule_id")
            rule = self.find_rule(rule_id, active_only=True)
            if rule is None:
                raise OperationError(f"active rule {rule_id} does not exist")
            if rule.mutable:
                raise OperationError(f"rule {rule_id} is already mutable")
            rule.history.append(
                {
                    "version": rule.version,
                    "text": rule.text,
                    "mutable": rule.mutable,
                    "ended_turn": turn,
                }
            )
            rule.version += 1
            rule.mutable = True
            return

        if kind not in {"amend", "repeal"}:
            raise OperationError(f"unknown rule operation {kind!r}")
        rule_id = op.get("rule_id")
        if not isinstance(rule_id, int):
            raise OperationError(f"{kind} requires integer rule_id")
        rule = self.find_rule(rule_id, active_only=True)
        if rule is None:
            raise OperationError(f"active rule {rule_id} does not exist")
        if not rule.mutable:
            raise OperationError(f"protected rule {rule_id} must be transmuted before {kind}")
        rule.history.append(
            {
                "version": rule.version,
                "text": rule.text,
                "mutable": rule.mutable,
                "ended_turn": turn,
            }
        )
        rule.version += 1
        if kind == "repeal":
            rule.active = False
            return
        text = op.get("text")
        if not isinstance(text, str) or not text.strip():
            raise OperationError("amend requires non-empty text")
        if len(text) > MAX_RULE_TEXT:
            raise OperationError(f"rule text exceeds {MAX_RULE_TEXT} characters")
        rule.text = text.strip()

    def _apply_state_op(self, op: dict[str, Any]) -> None:
        scope = op.get("scope")
        key = op.get("key")
        if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
            raise OperationError("state key must be a short non-empty string")
        if scope == "common":
            target = self.state.common
        elif scope == "player":
            seat = op.get("seat")
            if not isinstance(seat, int) or not (0 <= seat < SEAT_COUNT):
                raise OperationError("player state operation requires a valid seat")
            target = self.state.players[seat]
        else:
            raise OperationError(f"unknown state scope {scope!r}")

        kind = op.get("op", "set")
        if scope == "common" and key == "proposer_cursor":
            raise OperationError("proposer_cursor is managed by the host")
        if scope == "common" and key == "proposer_order" and kind not in {"set", "delete"}:
            raise OperationError("proposer_order supports only set or delete")
        if kind == "delete":
            if key == "points" and scope == "player":
                raise OperationError("player points cannot be deleted")
            target.pop(key, None)
        elif kind == "increment":
            delta = op.get("value")
            current = target.get(key, 0)
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise OperationError("increment value must be numeric")
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                raise OperationError("increment target must be numeric")
            target[key] = current + delta
        elif kind == "set":
            value = copy.deepcopy(op.get("value"))
            _validate_json_value(value)
            if _json_size(value) > MAX_VALUE_BYTES:
                raise OperationError(f"state value exceeds {MAX_VALUE_BYTES} bytes")
            target[key] = value
        else:
            raise OperationError(f"unknown state operation {kind!r}")

        if scope == "common" and key == "proposer_order":
            target["proposer_cursor"] = 0
        self._validate_runtime_fields()

    def _validate_runtime_fields(self) -> None:
        for player in self.state.players:
            points = player.get("points")
            if isinstance(points, bool) or not isinstance(points, int):
                raise OperationError("every player's points must remain an integer")
        common = self.state.common
        if "votes_required" in common:
            value = common["votes_required"]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= SEAT_COUNT:
                raise OperationError("votes_required must be an integer from 1 to 3")
        for key in ("majority_begins_turn", "victory_check_every"):
            if key in common:
                value = common[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise OperationError(f"{key} must be a positive integer")
        if "transmutation_votes_required" in common:
            value = common["transmutation_votes_required"]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= SEAT_COUNT:
                raise OperationError("transmutation_votes_required must be an integer from 1 to 3")
        if "victory_points" in common:
            value = common["victory_points"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise OperationError("victory_points must be a positive integer")
        for key in ("points_per_adopted_proposal", "points_per_rejected_proposal"):
            if key in common:
                value = common[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise OperationError(f"{key} must be an integer")
        if "fate_die_sides" in common:
            value = common["fate_die_sides"]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000:
                raise OperationError("fate_die_sides must be an integer from 1 to 1000")
        if "fate_recipient" in common:
            value = common["fate_recipient"]
            if not isinstance(value, str) or not value or len(value) > MAX_KEY_LENGTH:
                raise OperationError("fate_recipient must be a short non-empty mode name")
        if "proposer_order" in common and not _valid_proposer_order(common["proposer_order"]):
            raise OperationError(
                f"proposer_order must be a non-empty sequence of at most {MAX_PROPOSER_SEQUENCE} valid seats"
            )
        if "proposer_cursor" in common:
            value = common["proposer_cursor"]
            order = self.state.proposer_order()
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(order)
            ):
                raise OperationError("proposer_cursor must index proposer_order")
        if "next_proposer" in common:
            value = common["next_proposer"]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SEAT_COUNT:
                raise OperationError("next_proposer must be a valid seat")
        if _json_size(self.state.as_dict()) > MAX_STATE_BYTES:
            raise OperationError(f"world state exceeds {MAX_STATE_BYTES} bytes")

    def apply_ops_atomic(
        self, rule_ops: list[dict[str, Any]], state_ops: list[dict[str, Any]], *, turn: int
    ) -> None:
        """Validate and apply a ruling atomically; any bad op rejects all ops."""
        if len(rule_ops) + len(state_ops) > MAX_OPS_PER_RULING:
            raise OperationError(f"ruling exceeds {MAX_OPS_PER_RULING} operations")
        candidate = copy.deepcopy(self)
        for op in rule_ops:
            if not isinstance(op, dict):
                raise OperationError("rule operations must be JSON objects")
            candidate._apply_rule_op(op, turn)
        for op in state_ops:
            if not isinstance(op, dict):
                raise OperationError("state operations must be JSON objects")
            candidate._apply_state_op(op)
        self.rules = candidate.rules
        self.state = candidate.state
        self.next_rule_id = candidate.next_rule_id

    def proposal_rule_op(self, proposal: dict[str, Any]) -> dict[str, Any]:
        kind = proposal.get("kind")
        if kind == "enact":
            return {"op": "enact", "text": proposal.get("text", "")}
        if kind == "amend":
            return {"op": "amend", "rule_id": proposal.get("rule_id"), "text": proposal.get("text", "")}
        if kind == "repeal":
            return {"op": "repeal", "rule_id": proposal.get("rule_id")}
        if kind == "transmute":
            return {"op": "transmute", "rule_id": proposal.get("rule_id")}
        raise OperationError(f"unknown proposal kind {kind!r}")

    def proposer_for_turn(self, turn: int) -> int:
        order = self.state.proposer_order()
        override = self.state.common.pop("next_proposer", None)
        if isinstance(override, int) and not isinstance(override, bool) and 0 <= override < SEAT_COUNT:
            return override
        cursor = self.state.proposer_cursor()
        proposer = order[cursor]
        self.state.common["proposer_cursor"] = (cursor + 1) % len(order)
        return proposer

    def point_victors(self, *, turn: int, force: bool = False) -> list[int]:
        if not force and not self.state.victory_check_due(turn):
            return []
        threshold = self.state.victory_points()
        return [seat for seat in range(SEAT_COUNT) if self.state.points(seat) >= threshold]

    def cap_victors(self) -> list[int]:
        points = self.state.game_points()
        best = max(points)
        return [seat for seat, score in enumerate(points) if score == best]


def default_proposal(turn: int) -> dict[str, Any]:
    return {
        "kind": "enact",
        "text": f"Rule proposed on turn {turn}: no game effect.",
        "rationale": "Defaulted because the proposer did not return a valid proposal.",
        "default": True,
    }


def default_action() -> dict[str, Any]:
    return {"text": "pass", "default": True}
