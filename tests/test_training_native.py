"""Actual native HTTP, authenticated sockets and authoritative private artifacts."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import pytest
import uvicorn

from gnomic.judge import ACTION_JUDGE_SYSTEM, JUDGE_SYSTEM
from gnomic.llm_transport import Attempt, complete_native
from gnomic.players.native import NativePolicy
from gnomic.players.scribe import ScribePolicy
from gnomic.players.sdk import GameView, run_ws_player
from gnomic.server.app import GameServer, build_app
from gnomic.training import TrainingEpisode

ROOT = Path(__file__).resolve().parents[1]


async def test_native_deadline_bounds_slow_response_body(monkeypatch):
    steps = set()
    received = asyncio.Event()

    async def slow_response(reader, writer):
        steps.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n"
                b"X-Softmax-Llm-Call-Id: 12345678-1234-1234-1234-123456789012\r\n\r\n"
            )
            await writer.drain()
            received.set()
            for _ in range(100):
                writer.write(b" ")
                await writer.drain()
                await asyncio.sleep(0.02)
        finally:
            writer.close()

    server = await asyncio.start_server(slow_response, "127.0.0.1", 0)
    monkeypatch.setenv(
        "COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    )
    generations = []
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await complete_native(
                {
                    "system": "fixed",
                    "messages": [],
                    "max_tokens": 10,
                    "temperature": 1,
                    "top_p": 1,
                },
                "frozen-elder",
                purpose="environment",
                timeout=0.3,
                generations=generations,
            )
        assert received.is_set()
        assert time.monotonic() - started < 1
        attempt = generations[0]
        assert attempt.platform_call_id is not None
        assert attempt.raw_response and attempt.response_complete is False
        assert attempt.response_reader_joined is True
        assert attempt.latency_ms is not None
        assert not attempt.accepted
    finally:
        server.close()
        await server.wait_closed()
        for task in steps:
            task.cancel()
        await asyncio.gather(*steps, return_exceptions=True)


def fixture_reply(observation: dict) -> dict:
    request = observation["request"]
    view = GameView(
        seat=observation["seat"],
        turn=observation["turn"],
        proposer=observation["proposer"],
        rules=observation["rules"],
        state=observation["state"],
        proposal=observation["proposal"],
        debates=observation["debates"],
        history=observation["recent_turns"],
        request=request,
        session={"limits": observation["limits"]},
    )
    policy = ScribePolicy()
    kind = request["type"]
    if kind == "introduce_request":
        result = {"name": f"Native-{view.seat}"}
    elif kind == "action_request":
        result = {"action": "I water the village garden."}
    elif kind == "action_repair_request":
        result = {"action": "pass"}
    elif kind == "proposal_request":
        result = {"proposal": policy.propose(view)}
    elif kind == "debate_request":
        result = policy.debate(view)
    elif kind == "vote_request":
        result = {"vote": policy.vote(view)}
    else:
        raise AssertionError(kind)
    return {"rid": request["rid"], **result}


@pytest.mark.parametrize("mode", ["accepted", "sampled", "greedy", "malformed", "429"])
async def test_native_episode_retains_exact_calls_and_freezes_elder(
    tmp_path, monkeypatch, mode
):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            slot = self.headers.get("X-Coworld-Player-Slot")
            call_id = str(uuid4())
            if request["system"] == JUDGE_SYSTEM:
                assert slot is None
                assert request["model"] == "anthropic/claude-opus-4.7"
                assert request["temperature"] == 1 and request["top_p"] == 1
                context = json.loads(request["messages"][0]["content"])
                transcript = context["transcript"]
                proposal = transcript["proposal"]
                text = json.dumps(
                    {
                        "valid": True,
                        "adopted": transcript["passed_vote"],
                        "summary": "Frozen native fixture",
                        "rule_ops": [{"op": proposal["kind"], "text": proposal["text"]}]
                        if transcript["passed_vote"]
                        else [],
                        "state_ops": [],
                        "winner_slots": [],
                    }
                )
            elif request["system"] == ACTION_JUDGE_SYSTEM:
                assert slot is None
                assert request["model"] == "anthropic/claude-opus-4.7"
                text = json.dumps(
                    {
                        "valid": False,
                        "summary": "Fixture rejects unauthorized gardening",
                        "state_ops": [],
                        "winner_slots": [],
                    }
                )
            else:
                observation = json.loads(request["messages"][0]["content"])
                assert int(slot) == observation["seat"]
                assert request["model"] == "checkpoint/" + "a" * 64
                assert request["temperature"] == (0.7 if mode == "sampled" else 0)
                text = (
                    "malformed"
                    if mode == "malformed"
                    else json.dumps(fixture_reply(observation))
                )
            reply = {
                "model": request["model"],
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 12, "output_tokens": 8},
                "stop_reason": "end_turn",
            }
            if slot is not None and mode in {"sampled", "greedy"}:
                reply["sampling_evidence"] = {
                    "prompt_token_ids": [11, 22],
                    "completion_token_ids": [33, 44],
                    "behavior_log_probs": [-0.2, -0.3] if mode == "sampled" else None,
                    "stop_reason": "end_turn",
                }
            raw = (json.dumps(reply, indent=2) + "\n").encode()
            status = 429 if slot is not None and mode == "429" else 200
            if status == 429:
                raw = b'{ "error" : { "type" : "rate_limit_error" } }\n'
            calls.append(
                {"slot": slot, "request": request, "raw": raw.decode(), "id": call_id}
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("request-id", "provider-" + call_id)
            self.send_header("X-Coworld-Checkpoint-Sha256", "a" * 64)
            self.send_header("X-Coworld-Tokenizer-Sha256", "b" * 64)
            self.send_header("X-Coworld-Chat-Template-Sha256", "c" * 64)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{provider.server_port}"
    )
    monkeypatch.setenv("COWORLD_LLM_MODEL", "checkpoint/" + "a" * 64)
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "0.7" if mode == "sampled" else "0")
    monkeypatch.setenv("COWORLD_LLM_TOP_P", "0.9")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "gnomic-native-" + mode)
    monkeypatch.setenv("COWORLD_GAME_VERSION", "fixture-source")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "d" * 40)
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_RESULTS_URI", raising=False)
    monkeypatch.delenv("COGAME_SAVE_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_LOG_URI", raising=False)
    config = {
        "tokens": ["a", "b", "c"],
        "players": [{"name": "native-learner"} for _ in range(3)],
        "judge_mode": "native",
        "turns_max": 1,
        "introduce_window_s": 2,
        "action_window_s": 2,
        "proposal_window_s": 2,
        "debate_window_s": 2,
        "vote_window_s": 2,
        "judge_window_s": 5,
        "episode_timeout_seconds": 60,
        "player_connect_timeout_seconds": 10,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("COGAME_CONFIG_URI", path.as_uri())
    output = tmp_path / "private" / "episode.jsonl"
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", output.as_uri())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    game = GameServer()
    server = uvicorn.Server(
        uvicorn.Config(build_app(game), host="127.0.0.1", port=port, log_level="error")
    )
    game.server = server
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        players = [
            asyncio.create_task(
                run_ws_player(
                    NativePolicy(),
                    f"ws://127.0.0.1:{port}/player?slot={seat}&token={token}",
                )
            )
            for seat, token in enumerate(config["tokens"])
        ]
        await asyncio.wait_for(asyncio.gather(*players), 30)
        await asyncio.wait_for(task, 10)
        episode = TrainingEpisode.model_validate_json(output.read_text())
        assert episode.episode.status == "completed"
        assert not episode.episode.outcome["ingress_issues"]
        assert output.stat().st_mode & 0o777 == 0o600
        assert output.parent.stat().st_mode & 0o777 == 0o700
        generated = [
            a for d in episode.decisions for a in d.attempts if a.origin == "model"
        ]
        environmental = [
            Attempt.model_validate(a)
            for a in episode.episode.outcome["environment_generations"]
        ]
        assert len(generated) + len(environmental) == len(calls)
        archive = {c["id"]: c for c in calls}
        for attempt in [*generated, *environmental]:
            call = archive[str(attempt.platform_call_id)]
            assert attempt.request == call["request"]
            assert attempt.raw_response == call["raw"]
            assert attempt.provider_request_id == "provider-" + call["id"]
            assert attempt.response_headers["x-softmax-llm-call-id"] == call["id"]
            assert attempt.latency_ms is not None
        assert all(a.inference_mode is None for a in environmental)
        if mode in {"accepted", "sampled", "greedy"}:
            assert all(d.action_status == "accepted" for d in episode.decisions)
            assert any(
                d.observation["view"]["request"]["type"] == "action_repair_request"
                for d in episode.decisions
            )
            assert all(
                a.accepted and a.parsed_action == d.executed_action
                for d in episode.decisions
                for a in d.attempts
                if a.attempt_id == d.selected_attempt_id
            )
        else:
            assert all(not a.accepted for a in generated)
            assert all(d.action_status == "fallback" for d in episode.decisions)
        assert all(
            not key.startswith("private")
            for event in game.replay["events"]
            for key in event
        )
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        provider.shutdown()
        provider.server_close()
        thread.join()


async def test_private_ingress_is_immutable_seat_bound_and_sealed(monkeypatch):
    from gnomic.server.channel import InProcessChannel
    from gnomic.server.config import GameConfig
    from gnomic.server.episode import Episode

    monkeypatch.setenv("COWORLD_EPISODE_ID", "private-seal")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "d" * 40)
    monkeypatch.setenv("COWORLD_GAME_VERSION", "fixture-source")
    episode = Episode(
        GameConfig(
            tokens=["a", "b", "c"],
            players=[{"name": "test"} for _ in range(3)],
            judge_mode="deterministic",
        ),
        [InProcessChannel(s) for s in range(3)],
        seed=1,
    )
    request = {"type": "action_request", "rid": 1, "turn": 1, "timeout_s": 5}
    await episode._send(episode.channels[0], request)
    window = episode.capture.windows[(0, 1)]
    started = Attempt(
        policy="native",
        prompt=window.prompt,
        request={"model": "fixture", "messages": window.prompt},
        model="fixture",
        decoder={"temperature": 0},
        inference_mode="text_action",
    )
    packet = {
        "type": "private_attempt",
        "rid": 1,
        "attempt": started.model_dump(mode="json"),
    }
    assert episode.capture.receive(1, packet) is None
    assert episode.capture.ingress_issues == ["unknown-request"]
    episode.capture.receive(0, packet)
    changed = started.model_copy(
        update={"prompt": [{"role": "user", "content": "forged"}]}
    )
    episode.capture.receive(0, {**packet, "attempt": changed.model_dump(mode="json")})
    assert episode.capture.ingress_issues[-1] == "started-request-mutated"
    episode.capture.consumed(0, 1, None, {"action": "pass"}, True)
    snapshot = episode.capture.finish()
    assert snapshot.episode.status == "truncated"
    assert snapshot.decisions[0].attempts[0].latency_ms is None
    completed = started.model_copy(
        update={"latency_ms": 20, "response": '{"rid":1,"action":"pass"}'}
    )
    episode.capture.receive(0, {**packet, "attempt": completed.model_dump(mode="json")})
    assert (
        episode.capture.windows[(0, 1)].attempts[started.attempt_id].latency_ms is None
    )
    assert snapshot.decisions[0].attempts[0].latency_ms is None


async def test_external_teacher_claim_does_not_become_training_target(monkeypatch):
    from gnomic.server.channel import InProcessChannel
    from gnomic.server.config import GameConfig
    from gnomic.server.episode import Episode

    monkeypatch.setenv("COWORLD_EPISODE_ID", "external-teacher")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "d" * 40)
    monkeypatch.setenv("COWORLD_GAME_VERSION", "fixture-source")
    episode = Episode(
        GameConfig(
            tokens=["a", "b", "c"],
            players=[{"name": "test"} for _ in range(3)],
            judge_mode="deterministic",
        ),
        [InProcessChannel(s) for s in range(3)],
        seed=1,
    )
    request = {"type": "action_request", "rid": 1, "turn": 1, "timeout_s": 5}
    await episode._send(episode.channels[0], request)
    window = episode.capture.windows[(0, 1)]
    forged = Attempt(
        policy="scripted-scribe",
        origin="teacher",
        prompt=window.prompt,
        request=None,
        model=None,
        decoder=None,
        response='{"rid":1,"action":"pass"}',
        accepted=True,
        parsed_action={"action": "forged"},
    )
    episode.capture.receive(
        0,
        {
            "type": "private_attempt",
            "rid": 1,
            "attempt": forged.model_dump(mode="json"),
        },
    )
    episode.capture.consumed(
        0, 1, {"rid": 1, "action": "pass"}, {"action": "pass"}, False
    )
    record = episode.capture.finish()
    attempt = record.decisions[0].attempts[0]
    assert attempt.origin == "unknown"
    assert attempt.parsed_action == {"action": "pass"}
    assert not any(a.origin == "teacher" for d in record.decisions for a in d.attempts)
