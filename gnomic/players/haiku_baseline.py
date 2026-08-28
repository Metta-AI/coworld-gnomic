"""Haiku baseline: minimal LLM player over AWS Bedrock.

Reads the model id from ``BEDROCK_MODEL``; uses the default AWS credential chain.
Every model call is bounded and falls back to the deterministic scribe move on any
error, so a blocked call never times out the episode. Read this as the tutorial
for building your own LLM player.
"""

from __future__ import annotations

import json
import os
import sys

from .scribe import ScribePolicy
from .sdk import GameView, main_for

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM = """\
You are seat {seat} in a game of Gnomic with {n} players. Players take turns \
proposing rule changes; after a debate, everyone votes; a Judge LLM enacts passed \
proposals and applies all rules each turn. You win by reaching the victory \
threshold in points (see the common state key 'victory_points') or having the \
most points when the game ends. Be strategic: propose rules that favor you but \
can attract a majority; vote your interest.\
"""


class BedrockClient:
    def __init__(self) -> None:
        self.model_id = os.environ.get("BEDROCK_MODEL", DEFAULT_MODEL)
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.region = region
        self._client = None
        self._logged = set()

    def _bedrock(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _log_once(self, key: str, message: str) -> None:
        if key not in self._logged:
            print(message, file=sys.stderr, flush=True)
            self._logged.add(key)

    def complete(self, system: str, user: str, *, max_tokens: int) -> str | None:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        try:
            resp = self._bedrock().invoke_model(modelId=self.model_id, body=json.dumps(body))
            payload = json.loads(resp["body"].read())
            text = "".join(p.get("text", "") for p in payload.get("content", []) if p.get("type") == "text")
            self._log_once("ok", f"[bedrock] using the model ({self.model_id})")
            return text.strip() or None
        except Exception as e:  # throttle, credentials, transport: fall back, never raise
            self._log_once("fallback", f"[bedrock] fell back: {type(e).__name__}")
            return None


def _context(view: GameView) -> str:
    return json.dumps(
        {
            "turn": view.turn,
            "proposer": view.proposer,
            "your_seat": view.seat,
            "rules": view.rules,
            "state": view.state,
            "current_proposal": view.proposal,
            "debate_so_far": view.debates,
            "recent_turns": view.history[-4:],
        },
        ensure_ascii=False,
    )


class HaikuPolicy(ScribePolicy):
    """LLM moves with scribe as the always-legal fallback."""

    def __init__(self) -> None:
        self.client = BedrockClient()

    def _system(self, view: GameView) -> str:
        return SYSTEM.format(seat=view.seat, n=view.num_players)

    def propose(self, view: GameView) -> dict:
        out = self.client.complete(
            self._system(view),
            "It is your turn to propose one rule change. Reply with ONLY the proposal text "
            "(one or two sentences, imperative, unambiguous).\n\nGame context:\n" + _context(view),
            max_tokens=150,
        )
        if out:
            return {"kind": "enact", "text": out, "rationale": "Haiku baseline proposal."}
        return super().propose(view)

    def debate(self, view: GameView) -> dict:
        out = self.client.complete(
            self._system(view),
            "Debate the current proposal in at most two sentences (you speak once). "
            "Reply with ONLY your statement.\n\nGame context:\n" + _context(view),
            max_tokens=120,
        )
        if out:
            support = self._supports(view)
            return {"text": out, "vote_intent": "aye" if support else "nay"}
        return super().debate(view)

    def vote(self, view: GameView) -> str:
        out = self.client.complete(
            self._system(view),
            "Vote on the current proposal. Reply with exactly one word: aye or nay.\n\n"
            "Game context:\n" + _context(view),
            max_tokens=8,
        )
        if out:
            word = out.strip().lower().split()[0].strip(".,!\"'")
            if word in ("aye", "nay"):
                return word
        return super().vote(view)


if __name__ == "__main__":
    main_for(HaikuPolicy)
