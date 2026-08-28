"""Scribe: the deterministic, zero-model baseline.

Cycles through templated proposals, debates with a fixed formula, and votes by a
stable hash of the proposal text — bit-identical behavior across runs and
container restarts (all pseudo-randomness via sha256, never builtin hash()).
Must be able to fill every seat of an unattended episode (the hosted upload
smoke test runs scribe-vs-scribe-vs-scribe).
"""

from __future__ import annotations

import json

from .sdk import GameView, Policy, main_for, stable_rng_int

PROPOSAL_TEMPLATES = [
    "Enact: every player gains 1 point at the end of each turn.",
    "Enact: the proposer of a proposal that passes gains 2 additional points.",
    "Enact: a player who votes nay on a proposal that passes loses 1 point.",
    "Enact: the victory threshold recorded in the common key 'victory_points' is reduced by 2.",
    "Enact: whenever a proposal fails, every non-proposer gains 1 point.",
    "Enact: the player with the fewest points gains 2 points at the end of each turn.",
    "Enact: seat {seat} gains 1 extra point whenever one of their proposals passes.",
]


class ScribePolicy(Policy):
    def propose(self, view: GameView) -> dict:
        idx = stable_rng_int("propose", view.seat, view.turn, mod=len(PROPOSAL_TEMPLATES))
        text = PROPOSAL_TEMPLATES[idx].format(seat=view.seat)
        if text.startswith("Enact: "):
            text = text[len("Enact: "):]
        return {"kind": "enact", "text": text, "rationale": "Deterministic scribe proposal."}

    def debate(self, view: GameView) -> dict:
        support = self._supports(view)
        stance = "support" if support else "oppose"
        return {
            "text": (f"As seat {view.seat} with {view.my_points()} points, I {stance} this proposal. "
                     f"I favor rules that raise my standing without handing the lead away."),
            "vote_intent": "aye" if support else "nay",
        }

    def vote(self, view: GameView) -> str:
        return "aye" if self._supports(view) else "nay"

    def _supports(self, view: GameView) -> bool:
        if view.proposer == view.seat:
            return True
        proposal_text = json.dumps(view.proposal, ensure_ascii=False).lower()
        mentions_me = f"seat {view.seat}" in proposal_text
        if mentions_me:
            return True
        universal = "every player" in proposal_text or "each turn" in proposal_text
        if universal:
            return True
        return stable_rng_int("vote", view.seat, view.proposal, mod=2) == 0


if __name__ == "__main__":
    main_for(ScribePolicy)
