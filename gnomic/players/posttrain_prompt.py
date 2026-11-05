"""One player-visible prompt for Gnomic post-training and model inference."""

from __future__ import annotations

import json

from .sdk import GameView

SYSTEM = (
    "You are a Gnomic player. Answer the current request using only your player-visible "
    "view. Return one JSON reply with the request's rid and the fields required by the "
    "Gnomic player protocol."
)

REPLY_PROTOCOL = """
Reply with one JSON object only. Copy the current request's integer rid exactly.
Use the shape for that request; replace example values with your own choices:
- introduce_request: {"rid": R, "name": "your gnome name"}
- action_request or action_repair_request: {"rid": R, "action": "your natural-language action"}
  The action value must be a nonempty string, never an object or list.
- proposal_request: {"rid": R, "proposal": {"kind": "enact", "text": "new rule text", "rationale": "your reason"}}
  Other proposal kinds are amend (integer rule_id and text), repeal (integer
  rule_id), and transmute (integer rule_id); rationale is optional for each.
- debate_request: {"rid": R, "text": "your statement", "vote_intent": "aye"}
  vote_intent must be either "aye" or "nay".
- vote_request: {"rid": R, "vote": "aye"}; vote must be either "aye" or "nay".
Do not wrap the reply in Markdown or another object. These examples describe
field types, not moves to copy. Choose values under the active game rules.
""".strip()


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
            )
            + "\n\n"
            + REPLY_PROTOCOL,
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
