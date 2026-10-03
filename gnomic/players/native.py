"""Saved-model native player with the same prompt and parser as teacher export."""

from __future__ import annotations

import os
import time
from typing import Any

from gnomic.llm_transport import complete_native, current_window
from gnomic.protocol import parse_reply_text

from .posttrain_prompt import messages_for
from .sdk import GameView, Policy, main_for


class NativePolicy(Policy):
    async def _reply(self, view: GameView) -> dict[str, Any]:
        window = current_window.get()
        messages = messages_for(view)
        text = (
            await complete_native(
                {
                    "system": messages[0]["content"],
                    "messages": messages[1:],
                    "max_tokens": int(
                        os.environ.get("GNOMIC_LEARNER_MAX_TOKENS", "1024")
                    ),
                },
                os.environ.get("COWORLD_LLM_MODEL", "anthropic/claude-opus-4.7"),
                purpose="learner",
                slot=window.seat,
                timeout=window.deadline - time.monotonic(),
            )
        ).text
        result = parse_reply_text(text)
        if result.kind == "invalid":
            raise ValueError(f"Model reply is invalid: {result.error}")
        reply = result.value
        assert reply is not None
        if reply["rid"] != view.request["rid"]:
            raise ValueError("Model reply has the wrong request id")
        return reply

    async def introduce(self, view: GameView) -> str:
        name = (await self._reply(view))["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model gnome name must be nonempty text")
        return name

    async def action(self, view: GameView) -> str:
        action = (await self._reply(view))["action"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Model action must be nonempty text")
        return action

    async def propose(self, view: GameView) -> dict:
        proposal = (await self._reply(view))["proposal"]
        if not isinstance(proposal, dict):
            raise ValueError("Model proposal must be an object")
        return proposal

    async def debate(self, view: GameView) -> dict:
        reply = await self._reply(view)
        if not isinstance(reply["text"], str) or reply["vote_intent"] not in {
            "aye",
            "nay",
        }:
            raise ValueError("Model debate reply has invalid fields")
        return {"text": reply["text"], "vote_intent": reply["vote_intent"]}

    async def vote(self, view: GameView) -> str:
        vote = (await self._reply(view))["vote"]
        if vote not in {"aye", "nay"}:
            raise ValueError("Model vote must be aye or nay")
        return vote


if __name__ == "__main__":
    main_for(NativePolicy)
