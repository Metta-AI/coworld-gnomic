"""A model decision uses the same prompt and ordinary player SDK as the exporter."""

from __future__ import annotations

import asyncio
import json

import pytest

from gnomic.llm_transport import Response
from gnomic.players.native import NativePolicy
from gnomic.players.sdk import GameView, InProcessTransport, PlayerSession
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode


def scripted_generation(messages: list[dict[str, str]], _deadline: float) -> str:
    context = json.loads(messages[1]["content"])
    request = context["request"]
    assert request["rid"] > 0
    if request["type"] != "introduce_request":
        assert context["host_constraints"]
    reply = {"rid": request["rid"]}
    if request["type"] == "introduce_request":
        reply["name"] = "Model Gnome"
    elif request["type"] in {"action_request", "action_repair_request"}:
        reply["action"] = "pass"
    elif request["type"] == "proposal_request":
        reply["proposal"] = {
            "kind": "enact",
            "text": "Every player gains 1 point after each turn.",
            "rationale": "A common rule.",
        }
    elif request["type"] == "debate_request":
        reply.update({"text": "I support this common rule.", "vote_intent": "aye"})
    elif request["type"] == "vote_request":
        reply["vote"] = "aye"
    return json.dumps(reply)


@pytest.mark.asyncio
async def test_model_policy_completes_normal_three_seat_game(monkeypatch) -> None:
    async def complete(body, model, **kwargs):
        text = scripted_generation(
            [{"role": "system", "content": body["system"]}, *body["messages"]], 0
        )
        return Response(
            model=model,
            content=[{"type": "text", "text": text}],
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr("gnomic.players.native.complete_native", complete)
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions = [
        PlayerSession(NativePolicy(), InProcessTransport(channel))
        for channel in channels
    ]
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate(
        {
            "tokens": ["a", "b", "c"],
            "players": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "judge_mode": "deterministic",
            "turns_max": 3,
            "action_window_s": 2,
            "proposal_window_s": 2,
            "debate_window_s": 2,
            "vote_window_s": 2,
            "judge_window_s": 2,
        }
    )
    results, replay = await Episode(config, channels, seed=42).run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    assert results["turns_played"] == 3
    assert all(not session.defaults for session in sessions)
    assert all(
        event["action"] == {"text": "pass", "default": False}
        for event in replay["events"]
        if event["type"] == "action_made"
    )


@pytest.mark.asyncio
async def test_model_policy_rejects_wrong_request_id(monkeypatch) -> None:
    async def wrong_reply(body, model, **kwargs):
        return Response(
            model=model,
            content=[{"type": "text", "text": '{"rid":99,"action":"pass"}'}],
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr("gnomic.players.native.complete_native", wrong_reply)
    channel = InProcessChannel(0)
    session = PlayerSession(
        NativePolicy(),
        InProcessTransport(channel),
    )
    session.view = GameView(seat=0, turn=1)
    await session.handle(
        {"type": "action_request", "turn": 1, "rid": 1, "timeout_s": 2}
    )
    assert len(session.defaults) == 1
    assert "ValueError" in session.defaults[0].reason
    assert channel._queue.empty()
