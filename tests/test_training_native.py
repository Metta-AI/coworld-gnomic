"""Actual native HTTP, authenticated sockets and authoritative private artifacts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import httpx
import pytest
import uvicorn
import websockets

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


@pytest.mark.parametrize(
    "mode", ["accepted", "sampled", "greedy", "context_metadata", "malformed", "429"]
)
async def test_native_episode_retains_exact_calls_and_freezes_elder(
    tmp_path, monkeypatch, mode, *, seed=42, turns=1
):
    calls = []
    call_lock = Lock()
    (tmp_path / "attempt-start.json").write_text(
        json.dumps(
            {
                "started_monotonic": time.monotonic(),
                "seed": seed,
                "turns": turns,
                "mode": mode,
            }
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(body)
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
            if slot is not None and mode in {"sampled", "greedy", "context_metadata"}:
                reply["sampling_evidence"] = {
                    "prompt_token_ids": list(range(32768))
                    if mode == "context_metadata"
                    else [11, 22],
                    "completion_token_ids": [33, 44],
                    "behavior_log_probs": [-0.2, -0.3] if mode == "sampled" else None,
                    "stop_reason": "end_turn",
                }
            raw = (json.dumps(reply, indent=2) + "\n").encode()
            status = 429 if slot is not None and mode == "429" else 200
            if status == 429:
                raw = b'{ "error" : { "type" : "rate_limit_error" } }\n'
            call = {
                "slot": slot,
                "request": request,
                "raw": raw.decode(),
                "id": call_id,
                "request_body_b64": base64.b64encode(body).decode(),
                "response_body_b64": base64.b64encode(raw).decode(),
                "status": status,
                "fixture_only": True,
                "safe_request_headers": {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower()
                    in {
                        "content-type",
                        "anthropic-version",
                        "accept-encoding",
                        "x-coworld-player-slot",
                    }
                },
            }
            with call_lock:
                calls.append(call)
                with (tmp_path / "native-http.jsonl").open("a") as archive:
                    archive.write(json.dumps(call) + "\n")
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
    monkeypatch.setenv("COWORLD_EPISODE_ID", f"gnomic-native-{mode}-{seed}-{turns}")
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
    if turns == 45:
        manifest = json.loads((ROOT / "coworld_manifest_template.json").read_text())
        standard = next(
            v["game_config"]
            for v in manifest["variants"]
            if v["id"] == "standard-3-opus"
        )
        config = {**standard, "tokens": config["tokens"], "players": config["players"]}
    config.update(seed=seed, turns_max=turns)
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
    observer = None
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(trust_env=False) as client:
            health = await client.get(f"http://127.0.0.1:{port}/healthz")
            assert health.status_code == 200
            assert (
                await client.get(f"http://127.0.0.1:{port}/client/global")
            ).status_code == 200
        with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/player?slot=0&token=wrong"
            ):
                pytest.fail("invalid token joined a seat")
        assert refused.value.response.status_code == 403

        async def observe():
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/global", max_size=16 * 1024 * 1024
            ) as websocket:
                first = json.loads(await websocket.recv())
                assert first["type"] == "hello" and first["seats_total"] == 3
                await asyncio.wait_for(await websocket.ping(), 2)
                with (tmp_path / "global-events.jsonl").open("w") as archive:
                    archive.write(json.dumps(first) + "\n")
                    async for raw in websocket:
                        event = json.loads(raw)
                        archive.write(json.dumps(event) + "\n")
                        assert not any(key.startswith("private") for key in event)
                        if event["type"] == "final":
                            return

        observer = asyncio.create_task(observe())
        await asyncio.sleep(0)
        players = [
            asyncio.create_task(
                run_ws_player(
                    NativePolicy(),
                    f"ws://127.0.0.1:{port}/player?slot={seat}&token={token}",
                )
            )
            for seat, token in enumerate(config["tokens"])
        ]
        await asyncio.wait_for(asyncio.gather(*players), 180 if turns == 45 else 30)
        await asyncio.wait_for(observer, 5)
        await asyncio.wait_for(task, 10)
        assert game.results is not None and game.replay is not None
        (tmp_path / "results.json").write_text(json.dumps(game.results))
        (tmp_path / "replay.json").write_text(json.dumps(game.replay))
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
            assert (
                base64.b64decode(attempt.response_body_b64, validate=True)
                == call["raw"].encode()
            )
        assert all(a.inference_mode is None for a in environmental)
        if mode == "context_metadata":
            assert all(
                65536 < len(a.model_dump_json().encode()) < 16 * 1024 * 1024
                for a in generated
            )
            assert generated and all(
                a.prompt_token_ids == list(range(32768)) for a in generated
            )
            assert all(
                a.response_complete is True and a.response_reader_joined is True
                for a in generated
            )
            assert all(
                a.response_body_b64 is not None and a.http_status == 200
                for a in generated
            )
        if mode in {"accepted", "sampled", "greedy", "context_metadata"}:
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
        (tmp_path / "attempt-complete.json").write_text(
            json.dumps(
                {
                    "turns_played": game.results["turns_played"],
                    "HTTP_calls": len(calls),
                    "decisions": len(episode.decisions),
                    "status": episode.episode.status,
                    "family": episode.episode.seed_family,
                    "fixture_only": True,
                }
            )
        )
    finally:
        if observer is not None and not observer.done():
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        provider.shutdown()
        provider.server_close()
        thread.join()


async def test_signed_seed_aliases_share_the_complete_native_world_family(
    tmp_path, monkeypatch
):
    from gnomic.engine import Board

    records = []
    worlds = []
    for seed in (42, -42):
        directory = tmp_path / str(seed)
        directory.mkdir()
        with monkeypatch.context() as patch:
            await test_native_episode_retains_exact_calls_and_freezes_elder(
                directory, patch, "accepted", seed=seed, turns=45
            )
        episode = TrainingEpisode.model_validate_json(
            (directory / "private" / "episode.jsonl").read_text()
        )
        replay = json.loads((directory / "replay.json").read_text())
        calls = [
            json.loads(line)
            for line in (directory / "native-http.jsonl").read_text().splitlines()
        ]
        assert replay["results"]["turns_played"] == 45
        first = next(
            event for event in replay["events"] if event["type"] == "game_start"
        )
        action_call = next(
            call for call in calls if call["request"]["system"] == ACTION_JUDGE_SYSTEM
        )
        first_board = json.loads(action_call["request"]["messages"][0]["content"])[
            "board"
        ]
        initial = {key: first[key] for key in ("host_constraints", "rules", "state")}
        initial["next_rule_id"] = first_board["next_rule_id"]
        assert initial == Board.initial(seed=seed).as_dict()
        fate = [turn["host_random"] for turn in replay["turns"]]
        assert len(fate) == 45
        worlds.append({"initial": initial, "fate": fate})
        records.append(episode)
        (directory / "native-world-witness.json").write_text(
            json.dumps(worlds[-1], sort_keys=True)
        )
        assert len(calls) == 453 and len(episode.decisions) == 363

        def digest(value):
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        root = episode.episode.outcome["native_scenario_root"]
        assert root == {
            "version": "gnomic-native-root-v1",
            "hard_turn_cap": 45,
            "initial_board_sha256": digest(initial),
            "host_random_sha256": digest(fate),
        }
        assert episode.episode.seed_family == "gnomic-native-root-v1-" + digest(root)
    assert worlds[0] == worlds[1]
    (tmp_path / "signed-pair-observed-proof.json").write_text(
        json.dumps(
            {
                "same_initial_world_and_all45_Fate": True,
                "families": [record.episode.seed_family for record in records],
                "full_native_games": 2,
                "turns_each": 45,
                "fixture_only": True,
                "world_sha256": hashlib.sha256(
                    json.dumps(worlds[0], sort_keys=True).encode()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    assert records[0].episode.seed_family == records[1].episode.seed_family

    # Constructor controls are part of the same native system proof, not extra games.
    import random

    from gnomic.server.channel import InProcessChannel
    from gnomic.server.config import GameConfig
    from gnomic.server.episode import Episode

    controls = []
    for seed, horizon, wording in [
        (42, 1, "plain"),
        (-42, 9, "heldout-briefing"),
        (43, 45, "canonical"),
    ]:
        config = GameConfig(
            tokens=["x", "y", "z"],
            players=[{"name": "alias"}] * 3,
            turns_max=horizon,
            prompt_selection={"variant": wording},
            action_window_s=1,
        )
        control = Episode(
            config, [InProcessChannel(seat) for seat in range(3)], seed=seed
        )
        assert control._rng.getstate() == random.Random(seed).getstate()
        assert (
            not control.history
            and not control.events
            and control.board.state.proposer_cursor() == 0
        )
        assert (control.scenario_family == records[0].episode.seed_family) == (
            seed != 43
        )
        controls.append(
            {
                "seed": seed,
                "horizon": horizon,
                "wording": wording,
                "root": control.scenario_root,
                "family": control.scenario_family,
            }
        )
    (tmp_path / "constructor-controls.json").write_text(json.dumps(controls, indent=2))


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
