"""One player-visible prompt for Gnomic post-training and model inference."""

from __future__ import annotations

import json

from .sdk import GameView

SYSTEM = (
    "You are a Gnomic player. Answer the current request using only your player-visible "
    "view. Return one JSON reply with the request's rid and the fields required by the "
    "Gnomic player protocol."
)


def messages_for(view: GameView) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": view.prompt_selection.rules(
                SYSTEM,
                (
                    "You play Gnomic. Use only the player-visible view when answering the current request.",
                    "Return one JSON reply containing the request's rid and the fields required by the Gnomic player protocol.",
                ),
                (
                    "For this Gnomic decision, respond to the current request from the information visible to your player alone.",
                    "Your response must be a single JSON reply with that request's rid and all fields the Gnomic player protocol requires.",
                ),
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "seat": view.seat,
                    "turn": view.turn,
                    "proposer": view.proposer,
                    "host_constraints": view.host_constraints,
                    "limits": view.limits,
                    "rules": view.rules,
                    "state": view.state,
                    "proposal": view.proposal,
                    "debates": view.debates,
                    "recent_turns": view.history[-2:],
                    "request": view.request,
                },
                ensure_ascii=False,
            ),
        },
    ]
