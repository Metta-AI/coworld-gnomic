from __future__ import annotations

import asyncio

import pytest

from gnomic.players.baseline import BaselinePolicy
from gnomic.judge import ActionRuling, DeterministicJudge
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode


def config(**overrides) -> GameConfig:
    values = {
        "tokens": ["a", "b", "c"],
        "players": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}],
        "judge_mode": "deterministic",
        "proposal_window_s": 0.5,
        "debate_window_s": 0.5,
        "vote_window_s": 0.5,
        "judge_window_s": 2,
        "episode_timeout_seconds": 60,
    }
    values.update(overrides)
    return GameConfig.model_validate(values)


async def drive(channel: InProcessChannel) -> None:
    policy = BaselinePolicy()
    while True:
        message = await channel.player_recv()
        if message.get("type") == "final":
            return
        reply = await policy.respond(message)
        if reply is not None:
            await channel.player_send(reply)


@pytest.mark.asyncio
async def test_full_certification_episode_reaches_circuit_victory() -> None:
    channels = [InProcessChannel(i) for i in range(3)]
    tasks = [asyncio.create_task(drive(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(), channels, seed=42, broadcast=broadcast)
    results, replay = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    assert results["termination"] == "constitution_victory"
    assert results["turns_played"] % 3 == 0
    assert results["turns_played"] <= 45
    assert all(results["game_points"][seat] >= 100 for seat in results["winner_slots"])
    assert sum(results["scores"]) == pytest.approx(1)
    assert replay["format"] == "gnomic-replay-v1"
    assert replay["judge_usage"]["calls"] == 0
    assert sum(event["type"] == "action_made" for event in emitted) == results["turns_played"]
    assert sum(event["type"] == "debate_made" for event in emitted) == results["turns_played"]


@pytest.mark.asyncio
async def test_cap_produces_three_way_co_win_when_every_vote_fails() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def nay_driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            reply = await policy.respond(message)
            if reply is not None:
                if message["type"] == "vote_request":
                    reply["vote"] = "nay"
                await channel.player_send(reply)

    tasks = [asyncio.create_task(nay_driver(channel)) for channel in channels]
    results, _ = await Episode(config(turns_max=2), channels, seed=1).run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    points = results["game_points"]
    assert results["winner_slots"] == [seat for seat, value in enumerate(points) if value == max(points)]
    assert sum(results["scores"]) == pytest.approx(1)
    assert results["termination"] == "turn_cap"


@pytest.mark.asyncio
async def test_rejected_action_gets_one_player_repair_attempt() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def action_driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            reply = await policy.respond(message)
            if message.get("type") == "action_request":
                reply = {"rid": message["rid"], "action": "I steal every point without a rule."}
            if reply is not None:
                await channel.player_send(reply)

    class RepairJudge(DeterministicJudge):
        async def act(self, board, action_record, turn, turns_max):  # type: ignore[override]
            return (
                ActionRuling(
                    valid=False,
                    summary="No active rule authorizes stealing points.",
                    state_ops=[],
                    winner_slots=[],
                ),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0},
            )

    tasks = [asyncio.create_task(action_driver(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(turns_max=1), channels, seed=3, broadcast=broadcast)
    episode.judge = RepairJudge()
    results, _ = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    actions = [event for event in emitted if event["type"] == "action_made"]
    rulings = [event for event in emitted if event["type"] == "action_ruling"]
    assert [event["attempt"] for event in actions] == [1, 2]
    assert actions[1]["action"]["text"] == "pass"
    assert [event["valid"] for event in rulings] == [False, True]
    assert any(event["type"] == "proposal_made" for event in emitted)
