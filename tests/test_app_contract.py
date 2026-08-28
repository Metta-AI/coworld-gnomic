from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gnomic.server.app import GameServer, build_app


def write_config(tmp_path) -> str:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                "judge_mode": "deterministic",
                "player_connect_timeout_seconds": 600,
            }
        )
    )
    return str(path)


def test_live_http_and_global_websocket_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    game = GameServer()
    with TestClient(build_app(game)) as client:
        assert client.get("/healthz").json() == {"status": "ok", "mode": "live"}
        for route in ("/client/global", "/client/player?slot=0&token=a", "/client/replay"):
            response = client.get(route)
            assert response.status_code == 200
            assert "GNOMIC MOOT" in response.text
            assert 'aria-label="Replay controls"' in response.text
        with client.websocket_connect("/global") as websocket:
            assert websocket.receive_json()["type"] == "hello"


def test_player_auth_rejects_wrong_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    with TestClient(build_app(GameServer())) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/player?slot=0&token=wrong") as websocket:
                websocket.receive_json()


def test_replay_contract(tmp_path, monkeypatch) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"format": "gnomic-replay-v1", "events": []}))
    monkeypatch.setenv("COGAME_LOAD_REPLAY_URI", str(replay))
    monkeypatch.delenv("COGAME_CONFIG_URI", raising=False)
    with TestClient(build_app(GameServer())) as client:
        assert client.get("/healthz").json()["mode"] == "replay"
        with client.websocket_connect("/replay") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "replay"
            assert message["data"]["format"] == "gnomic-replay-v1"


async def test_certification_probe_survives_fast_episode(tmp_path, monkeypatch) -> None:
    """The hosted certifier connects to /global before the episode starts but
    pings only later; a fast episode must not tear the socket down at done.
    Reproduces the 0.2.2 certification failure (cow_b12a3042, "no close frame
    received or sent"): with the old close-at-done behavior the ping below
    fails; with the spectator drain the pong arrives and the server still
    exits once the client leaves."""
    import socket

    import uvicorn
    import websockets

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                "judge_mode": "deterministic",
                "turns_max": 1,
                "action_window_s": 0.05,
                "proposal_window_s": 0.05,
                "debate_window_s": 0.05,
                "vote_window_s": 0.05,
                "judge_window_s": 1,
                "player_connect_timeout_seconds": 0.2,
                "episode_timeout_seconds": 60,
            }
        )
    )
    monkeypatch.setenv("COGAME_CONFIG_URI", str(path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_RESULTS_URI", raising=False)
    monkeypatch.delenv("COGAME_SAVE_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_LOG_URI", raising=False)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    game = GameServer()
    server = uvicorn.Server(uvicorn.Config(build_app(game), host="127.0.0.1", port=port, log_level="error"))
    game.server = server
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    async with websockets.connect(f"ws://127.0.0.1:{port}/global") as websocket:
        hello = json.loads(await asyncio.wait_for(websocket.recv(), 5))
        assert hello["type"] == "hello"
        while not game.done:
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.5)  # past the old 1s grace in which the server used to exit
        pong_waiter = await websocket.ping(b"coworld-certification-ping")
        await asyncio.wait_for(pong_waiter, 2)

    await asyncio.wait_for(serve_task, 10)


def test_fatal_operator_log_preserves_root_cause(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    log_path = tmp_path / "operator.json"
    monkeypatch.setenv("COGAME_LOG_URI", str(log_path))
    game = GameServer()
    game.episode = SimpleNamespace(operator_log=lambda: {"turn": 7, "phase": "judge"})  # type: ignore[assignment]
    game.fatal_error = "JudgeError: model returned no text"

    game._write_operator_log()

    assert json.loads(log_path.read_text()) == {
        "turn": 7,
        "phase": "judge",
        "fatal_error": "JudgeError: model returned no text",
    }
