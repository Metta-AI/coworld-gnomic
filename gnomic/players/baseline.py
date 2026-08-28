"""Deterministic, no-network baseline used by certification."""

from __future__ import annotations

from typing import Any

from .client import main


class BaselinePolicy:
    def __init__(self) -> None:
        self.seat = 0
        self.proposer = 0

    async def respond(self, message: dict[str, Any]) -> dict[str, Any] | None:
        kind = message.get("type")
        if kind == "game_start":
            self.seat = int(message["you"]["seat"])
        elif kind == "turn_start":
            self.proposer = int(message["proposer"])
        elif kind in {"action_request", "action_repair_request"}:
            return {"rid": message["rid"], "action": "pass"}
        elif kind == "proposal_request":
            turn = int(message["turn"])
            return {
                "rid": message["rid"],
                "proposal": {
                    "kind": "enact",
                    "text": f"Ceremonial rule {turn}: players should explain their choices concisely.",
                    "rationale": "A harmless deterministic certification proposal.",
                },
            }
        elif kind == "debate_request":
            return {
                "rid": message["rid"],
                "text": "This proposal is coherent, host-safe, and preserves playable Gnomic.",
                "vote_intent": "aye",
            }
        elif kind == "vote_request":
            return {"rid": message["rid"], "vote": "aye", "reason": "Valid and harmless."}
        return None


if __name__ == "__main__":
    main(BaselinePolicy())
