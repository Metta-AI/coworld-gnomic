"""FastAPI implementation of the current CoWorld GAME contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Response, WebSocket
from starlette.websockets import WebSocketDisconnect

from .channel import NullSeatChannel, SeatChannel
from .config import GameConfig
from .episode import Episode
from .io import artifact_method, maybe_decompress, read_data, write_data
from .ws_channel import WebSocketSeatChannel

VIEWER_PATH = Path(__file__).parent / "viewer" / "index.html"

# How long a finished episode waits for connected /global spectators to leave
# before the server exits. The certification probe disconnects as soon as it
# has its ping-pong and first message, so certification pays seconds, not the
# cap; the cap only bounds a live human viewer holding a finished game open.
SPECTATOR_DRAIN_SECONDS = 120.0


class GameServer:
    def __init__(self) -> None:
        self.replay_mode = bool(os.environ.get("COGAME_LOAD_REPLAY_URI"))
        self.config: GameConfig | None = None
        self.tokens: list[str] = []
        self.channels: dict[int, SeatChannel] = {}
        self.started = False
        self.done = False
        self.fatal_error: str | None = None
        self.episode: Episode | None = None
        self.results: dict[str, Any] | None = None
        self.replay: dict[str, Any] | None = None
        self.loaded_replay: dict[str, Any] | None = None
        self.event_log: list[dict[str, Any]] = []
        self.spectators: set[WebSocket] = set()
        self.server: uvicorn.Server | None = None
        self._start_lock = asyncio.Lock()

        if self.replay_mode:
            raw = maybe_decompress(read_data(os.environ["COGAME_LOAD_REPLAY_URI"])
            )
            self.loaded_replay = json.loads(raw)
        else:
            config_uri = os.environ.get("COGAME_CONFIG_URI")
            if not config_uri:
                raise RuntimeError("COGAME_CONFIG_URI is required outside replay mode")
            self.config = GameConfig.model_validate_json(read_data(config_uri))
            self.tokens = self.config.tokens

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.event_log.append(message)
        dead: list[WebSocket] = []
        for websocket in self.spectators:
            try:
                await websocket.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(websocket)
        for websocket in dead:
            self.spectators.discard(websocket)

    def _seed(self) -> int:
        assert self.config is not None
        if self.config.seed is not None:
            return self.config.seed
        return int(hashlib.sha256("|".join(self.tokens).encode()).hexdigest(), 16) % (2**31)

    async def maybe_start(self) -> None:
        async with self._start_lock:
            if self.started or self.config is None:
                return
            if len(self.channels) == self.config.seat_count():
                self.started = True
                asyncio.create_task(self._run())

    async def start_after_timeout(self) -> None:
        assert self.config is not None
        await asyncio.sleep(self.config.player_connect_timeout_seconds)
        async with self._start_lock:
            if not self.started:
                self.started = True
                asyncio.create_task(self._run())

    def _episode_channels(self) -> list[SeatChannel]:
        assert self.config is not None
        return [self.channels.get(seat) or NullSeatChannel(seat) for seat in range(self.config.seat_count())]

    async def _run(self) -> None:
        assert self.config is not None
        channels = self._episode_channels()
        self.episode = Episode(self.config, channels, seed=self._seed(), broadcast=self.broadcast)
        try:
            self.results, self.replay = await asyncio.wait_for(
                self.episode.run(), timeout=self.config.episode_timeout_seconds
            )
            self._write_artifacts()
            await asyncio.gather(
                *(channel.send({"type": "final", "scores": self.results["scores"]}) for channel in channels)
            )
            await self.broadcast({"type": "final", "scores": self.results["scores"]})
            self.done = True
            await asyncio.sleep(1)
        except Exception as exc:
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.done = True
            try:
                self._write_operator_log()
            except Exception as log_exc:
                self.fatal_error += f"; operator log write failed: {type(log_exc).__name__}: {log_exc}"
            await self.broadcast({"type": "fatal_error", "error": self.fatal_error})
            # Deliberately leave result/replay artifacts absent: a production judge
            # outage is an episode failure, not a mechanical game with different law.
        finally:
            # Hold the server up until /global spectators disconnect (capped).
            # The hosted certification probe connects to /global before the
            # episode starts but pings only after it observes every player pod
            # started; a fast episode that exits at done wins that race and the
            # probe sees a dead socket ("no close frame received or sent",
            # certification of 0.2.2, cow_b12a3042). With no spectators the
            # loop is a no-op, so ordinary episodes exit as promptly as before.
            if self.server is not None:
                deadline = asyncio.get_event_loop().time() + SPECTATOR_DRAIN_SECONDS
                while self.spectators and asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(0.2)
                self.server.should_exit = True

    def _write_artifacts(self) -> None:
        if self.results is not None and os.environ.get("COGAME_RESULTS_URI"):
            write_data(
                os.environ["COGAME_RESULTS_URI"],
                json.dumps(self.results),
                content_type="application/json",
                http_method=artifact_method("COGAME_RESULTS_METHOD"),
            )
        if self.replay is not None and os.environ.get("COGAME_SAVE_REPLAY_URI"):
            write_data(
                os.environ["COGAME_SAVE_REPLAY_URI"],
                json.dumps(self.replay),
                content_type="application/json",
                http_method=artifact_method("COGAME_SAVE_REPLAY_METHOD"),
            )
        self._write_operator_log()

    def _write_operator_log(self) -> None:
        if self.episode is not None and os.environ.get("COGAME_LOG_URI"):
            payload = self.episode.operator_log()
            if self.fatal_error is not None:
                payload["fatal_error"] = self.fatal_error
            write_data(
                os.environ["COGAME_LOG_URI"],
                json.dumps(payload),
                content_type="application/json",
                http_method=artifact_method("COGAME_LOG_METHOD"),
            )


def _viewer_html() -> str:
    return VIEWER_PATH.read_text() if VIEWER_PATH.exists() else "<h1>Gnomic</h1>"


async def _snapshot(game: GameServer, channel: SeatChannel, seat: int) -> None:
    if game.episode is None:
        await channel.send({"type": "lobby", "seat": seat})
        return
    await channel.send(game.episode.snapshot_for(seat))
    await channel.send(
        {"type": "snapshot", "turn": game.episode.current_turn, "phase": game.episode.current_phase}
    )


def build_app(game: GameServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not game.replay_mode:
            asyncio.create_task(game.start_after_timeout())
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "mode": "replay" if game.replay_mode else "live"}

    @app.get("/client/global")
    @app.get("/client/player")
    @app.get("/client/replay")
    def viewer() -> Response:
        return Response(_viewer_html(), media_type="text/html")

    @app.get("/client/art/{name}.png")
    def viewer_art(name: str) -> Response:
        path = VIEWER_PATH.parent / "art" / f"{name}.png"
        if not name.isidentifier() or not path.exists():
            return Response(status_code=404)
        return Response(path.read_bytes(), media_type="image/png")

    @app.websocket("/player")
    async def player_ws(websocket: WebSocket) -> None:
        try:
            seat = int(websocket.query_params.get("slot", "-1"))
        except ValueError:
            await websocket.close(code=1008)
            return
        token = websocket.query_params.get("token", "")
        if seat < 0 or seat >= len(game.tokens) or token != game.tokens[seat]:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        channel = WebSocketSeatChannel(seat, websocket)
        prior = game.channels.get(seat)
        if prior is not None:
            prior.close_reader()
        game.channels[seat] = channel
        await _snapshot(game, channel, seat)
        await game.maybe_start()
        try:
            while channel.connected and not game.done:
                await asyncio.sleep(0.2)
        finally:
            channel.close_reader()

    @app.websocket("/global")
    async def global_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        game.spectators.add(websocket)
        await websocket.send_json(
            {
                "type": "hello",
                "started": game.started,
                "done": game.done,
                "seats_connected": len(game.channels),
                "seats_total": game.config.seat_count() if game.config else None,
            }
        )
        for event in list(game.event_log):
            await websocket.send_json(event)
        try:
            # Hold the socket until the CLIENT leaves rather than until the
            # game is done: the certification probe pings this socket after a
            # fast episode may already have finished, and closing at done
            # raced it. Server exit is driven by _run's spectator drain.
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            game.spectators.discard(websocket)

    @app.websocket("/replay")
    async def replay_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "replay", "data": game.loaded_replay or {}})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    game = GameServer()
    app = build_app(game)
    config = uvicorn.Config(
        app,
        host=os.environ.get("COGAME_HOST", "0.0.0.0"),
        port=int(os.environ.get("COGAME_PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
    server = uvicorn.Server(config)
    game.server = server
    server.run()
    if game.fatal_error:
        raise SystemExit(game.fatal_error)


if __name__ == "__main__":
    main()
