"""FastAPI implementation of the current CoWorld GAME contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Response, WebSocket
from starlette.websockets import WebSocketDisconnect

from gnomic.lifecycle import (
    CLEANUP_SECONDS,
    main_owned,
    owned_children,
    settle,
    shutdown_deadline,
)
from gnomic.training import validate_capture_environment, write_private_episode

from .channel import NullSeatChannel, SeatChannel, settle_channels
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
        self.run_task: asyncio.Task | None = None
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
        self.start_task: asyncio.Task | None = None
        self.stopping = False
        self.private_written = False
        self.registered_channels: list[SeatChannel] = []
        self.owner_deadline: float | None = None
        self.cleanup_task: asyncio.Task[bool] | None = None
        self.episode_task: asyncio.Task | None = None
        self.ownership_joined = False
        self.native_owned_tasks: set[asyncio.Task] = set()

        if self.replay_mode:
            raw = maybe_decompress(read_data(os.environ["COGAME_LOAD_REPLAY_URI"]))
            self.loaded_replay = json.loads(raw)
        else:
            config_uri = os.environ.get("COGAME_CONFIG_URI")
            if not config_uri:
                raise RuntimeError("COGAME_CONFIG_URI is required outside replay mode")
            if "COGAME_SAVE_TRAJECTORY_URI" in os.environ:
                validate_capture_environment()
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
        return int(hashlib.sha256("|".join(self.tokens).encode()).hexdigest(), 16) % (
            2**31
        )

    async def maybe_start(self) -> None:
        async with self._start_lock:
            if self.stopping or self.started or self.config is None:
                return
            if len(self.channels) == self.config.seat_count():
                self._start_episode()

    async def start_after_timeout(self) -> None:
        assert self.config is not None
        await asyncio.sleep(self.config.player_connect_timeout_seconds)
        async with self._start_lock:
            if not self.stopping and not self.started:
                self._start_episode()

    def _episode_channels(self) -> list[SeatChannel]:
        assert self.config is not None
        return [
            self.channels.get(seat) or NullSeatChannel(seat)
            for seat in range(self.config.seat_count())
        ]

    def _start_episode(self) -> None:
        assert self.config is not None
        self.started = True
        self.episode = Episode(
            self.config,
            self._episode_channels(),
            seed=self._seed(),
            broadcast=self.broadcast,
        )
        self.run_task = asyncio.create_task(self._run())

    def request_stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        if self.owner_deadline is None:
            self.owner_deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
            inherited = shutdown_deadline.get()
            if inherited is not None and inherited[0] is not None:
                self.owner_deadline = min(self.owner_deadline, inherited[0])
        for task in (self.start_task, self.run_task):
            if task is not None and not task.done():
                task.cancel()

    async def _settle_ownership(self) -> bool:
        if self.owner_deadline is None:
            self.owner_deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
        deadline = self.owner_deadline
        registered = list(self.registered_channels)
        # In-process fixtures and collectors register channels without a WS route.
        registered.extend(c for c in self.channels.values() if c not in registered)
        channel_owner = asyncio.create_task(settle_channels(registered, deadline))
        engine_tasks = set(self.native_owned_tasks)
        if self.episode_task is not None:
            engine_tasks.add(self.episode_task)
        engine_joined = await settle(engine_tasks, deadline, cancel=True)
        engine_joined = (
            await settle(set(self.native_owned_tasks), deadline, cancel=True)
            and engine_joined
        )
        channel_joined = await settle({channel_owner}, deadline, cancel=False)
        if not channel_joined:
            for channel in registered:
                channel.seal()
            await settle({channel_owner}, deadline, cancel=True)
        self.ownership_joined = (
            engine_joined
            and channel_joined
            and (
                not channel_owner.cancelled()
                and channel_owner.exception() is None
                and channel_owner.result()
            )
        )
        return self.ownership_joined

    async def _join_ownership(self) -> bool:
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._settle_ownership())
        assert self.owner_deadline is not None or self.cleanup_task is not None
        # The cleanup task itself owns the single deadline, not its callers.
        if self.owner_deadline is None:
            self.owner_deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
        done, _ = await asyncio.wait(
            {self.cleanup_task},
            timeout=max(0, self.owner_deadline - asyncio.get_running_loop().time()),
        )
        if not done:
            for channel in self.registered_channels:
                channel.seal()
            self.ownership_joined = False
            return False
        return self.cleanup_task.result()

    async def stop(self) -> None:
        self.request_stop()
        assert self.owner_deadline is not None
        await self._join_ownership()
        tasks = {t for t in (self.start_task, self.run_task) if t is not None}
        joined = await settle(tasks, self.owner_deadline, cancel=True)
        self.ownership_joined = self.ownership_joined and joined
        if not self.ownership_joined:
            self.results = self.replay = None
        if self.started and not self.private_written:
            assert self.episode is not None
            self._write_interrupted()

    def _write_private(
        self,
        status: Literal["completed", "failed", "truncated"],
        failure_kind: str | None = None,
    ) -> None:
        assert self.episode is not None
        if not self.private_written and "COGAME_SAVE_TRAJECTORY_URI" in os.environ:
            record = self.episode.capture.finish(
                status=status,
                failure_kind=failure_kind,
                ownership_joined=self.ownership_joined,
            )
            write_private_episode(os.environ["COGAME_SAVE_TRAJECTORY_URI"], record)
            self.private_written = True

    def _write_interrupted(self) -> None:
        self._write_private("truncated", "ownership-interrupted")

    async def _run(self) -> None:
        assert self.config is not None and self.episode is not None
        channels = self._episode_channels()
        try:
            token = owned_children.set(self.native_owned_tasks)
            self.episode_task = asyncio.create_task(self.episode.run())
            owned_children.reset(token)
            done, _ = await asyncio.wait(
                {self.episode_task}, timeout=self.config.episode_timeout_seconds
            )
            if not done:
                raise TimeoutError("episode deadline exceeded")
            self.results, self.replay = self.episode_task.result()
            if not await self._join_ownership():
                self.results = self.replay = None
                self._write_interrupted()
                raise RuntimeError("episode ownership remains unresolved")
            private = self.episode.capture.finish()
            if private.episode.status != "completed":
                self.results = self.replay = None
                self._write_interrupted()
                raise RuntimeError("private episode has unresolved native ownership")
            self._write_private("completed")
            self._write_artifacts()
            await asyncio.gather(
                *(
                    c.send({"type": "final", "scores": self.results["scores"]})
                    for c in channels
                )
            )
            await self.broadcast({"type": "final", "scores": self.results["scores"]})
            self.done = True
            await asyncio.sleep(1)
        except Exception as exc:
            await self._join_ownership()
            self.results = self.replay = None
            self.fatal_error = type(exc).__name__
            self.done = True
            self._write_private(
                "failed" if self.ownership_joined else "truncated", type(exc).__name__
            )
            self._write_operator_log()
            await self.broadcast({"type": "fatal_error", "error": self.fatal_error})
        finally:
            await self._join_ownership()
            if self.stopping and not self.private_written:
                self._write_interrupted()
            if self.server is not None:
                deadline = asyncio.get_running_loop().time() + SPECTATOR_DRAIN_SECONDS
                while (
                    not self.stopping
                    and self.ownership_joined
                    and self.spectators
                    and asyncio.get_running_loop().time() < deadline
                ):
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
        {
            "type": "snapshot",
            "turn": game.episode.current_turn,
            "phase": game.episode.current_phase,
        }
    )


def build_app(game: GameServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not game.replay_mode:
            game.start_task = asyncio.create_task(game.start_after_timeout())
        try:
            yield
        finally:
            await game.stop()

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
        if game.stopping or game.cleanup_task is not None:
            await websocket.close(code=1008)
            return
        channel = WebSocketSeatChannel(seat, websocket)
        game.registered_channels.append(channel)
        prior = game.channels.get(seat)
        if prior is not None:
            await prior.close_reader()
        game.channels[seat] = channel
        await _snapshot(game, channel, seat)
        await game.maybe_start()
        try:
            while channel.connected and not game.done:
                await asyncio.sleep(0.2)
        finally:
            if game.cleanup_task is None:
                await channel.close_reader()

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
        ws_max_size=16 * 1024 * 1024,
    )

    class OwnedServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            # main_owned owns first/repeated signals and the finite cleanup budget.
            yield

    config.timeout_graceful_shutdown = 2
    server = OwnedServer(config)
    game.server = server

    async def serve_owned() -> None:
        try:
            await server.serve()
        finally:
            await game.stop()

    main_owned(serve_owned(), game.request_stop)
    if game.started and not game.ownership_joined:
        raise SystemExit("game ownership did not settle")
    if game.fatal_error:
        raise SystemExit(game.fatal_error)


if __name__ == "__main__":
    main()
