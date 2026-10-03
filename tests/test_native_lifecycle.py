"""Owned socket prefixes and nonce-bound shutdown cannot fabricate release."""

import asyncio
import base64
import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager

import pytest

from gnomic.lifecycle import OwnershipUnsettled, player_loop
from gnomic.llm_transport import Attempt, LearnerWindow, complete_native, current_window
from gnomic.server.channel import InProcessChannel, settle_channels


async def test_actual_partial_invalid_utf8_survives_deadline_and_reader_release(
    monkeypatch,
):
    handlers = set()
    release = asyncio.Event()
    snapshots = []

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nX-Softmax-Llm-Call-Id: 12345678-1234-1234-1234-123456789012\r\n\r\n\xffprefix"
            )
            await writer.drain()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    async def publish(attempt):
        snapshots.append(attempt.model_copy(deep=True))

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    monkeypatch.setenv(
        "COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    )
    token = current_window.set(
        LearnerWindow(2, "native", time.monotonic() + 1, publish)
    )
    try:
        with pytest.raises(TimeoutError):
            await complete_native(
                {"system": "actual", "messages": [], "max_tokens": 8},
                "native",
                purpose="learner",
                timeout=0.15,
            )
        final = snapshots[-1]
        assert final.response_complete is False
        assert final.response_reader_joined is True
        assert final.http_status == 200
        assert final.raw_response is None
        assert base64.b64decode(final.response_body_b64) == b"\xffprefix"
        assert final.platform_call_id is not None
        assert final._received_header_pairs.pairs
        assert snapshots[0].http_status is None
        assert snapshots[0].response_complete is None
    finally:
        current_window.reset(token)
        release.set()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)


async def test_native_uncooperative_close_is_bounded_and_truthful(monkeypatch):
    import httpx

    steps = set()
    original = httpx.AsyncClient.aclose

    async def send(client, request, **kwargs):
        return httpx.Response(
            200,
            request=request,
            stream=httpx.ByteStream(
                b'{"model":"native","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}'
            ),
        )

    # A reader whose close task ignores cancellation remains visibly unresolved.
    class SuppressingFuture(asyncio.Future):
        def cancel(self, *args, **kwargs):
            return False

    blocker = SuppressingFuture()

    async def close(client):
        steps.add(asyncio.current_task())
        await blocker
        await original(client)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", "http://not-contacted")
    generations = []
    started = time.monotonic()
    try:
        with pytest.raises(OwnershipUnsettled):
            await complete_native(
                {
                    "system": "frozen",
                    "messages": [],
                    "temperature": 1,
                    "top_p": 1,
                    "max_tokens": 8,
                },
                "frozen",
                purpose="environment",
                timeout=0.2,
                generations=generations,
            )
        assert time.monotonic() - started < 1.5
        assert generations[0].response_reader_joined is False
    finally:
        blocker.set_result(None)
        await asyncio.gather(*steps, return_exceptions=True)


async def test_stop_ack_denies_observed_unjoined_native_reader():
    channel = InProcessChannel(0)
    channel.register_window(1)
    attempt = Attempt(
        policy="native",
        prompt=[],
        request={},
        model="native",
        decoder={},
        response_reader_joined=False,
    )
    await channel.player_send(
        {"type": "private_attempt", "rid": 1, "attempt": attempt.model_dump()}
    )
    channel.stop_id = "engine"
    channel.stop_deadline = asyncio.get_running_loop().time() + 1
    with pytest.raises(ValueError, match="contradicts unsettled"):
        await channel.player_send({"type": "stopped", "stop_id": "engine"})
    assert not channel.stopped.is_set()


async def test_all_player_owners_join_before_nonce_ack_and_final():
    channel = InProcessChannel(0)
    active = asyncio.Event()
    settled = asyncio.Event()

    async def handle(packet):
        if packet["type"] == "action_request":
            active.set()
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()

    player = asyncio.create_task(
        player_loop(channel.player_recv, channel.player_send, handle)
    )
    await channel.send({"type": "action_request"})
    await active.wait()
    assert await settle_channels([channel], asyncio.get_running_loop().time() + 1)
    assert settled.is_set() and channel.stopped.is_set()
    await channel.send({"type": "final"})
    await player


async def test_cancellation_suppressing_callback_withholds_ack():
    channel = InProcessChannel(0)
    blocker = None

    class SuppressingFuture(asyncio.Future):
        def cancel(self, *args, **kwargs):
            return False

    blocker = SuppressingFuture()
    entered = asyncio.Event()

    async def handle(packet):
        entered.set()
        await blocker

    player = asyncio.create_task(
        player_loop(channel.player_recv, channel.player_send, handle)
    )
    await channel.send({"type": "action_request"})
    await entered.wait()
    try:
        assert not await settle_channels(
            [channel], asyncio.get_running_loop().time() + 0.05
        )
        assert not channel.stopped.is_set()
    finally:
        blocker.set_result(None)
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)


@asynccontextmanager
async def partial_response(
    monkeypatch,
    prefix: bytes,
    *,
    status: int = 200,
    complete: bool = False,
    extra_headers: bytes = b"",
):
    release = asyncio.Event()
    observed = asyncio.Event()
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            writer.write(
                f"HTTP/1.1 {status} Fixture\r\nContent-Length: {len(prefix) if complete else 1000}\r\n"
                "Content-Type: application/json\r\n"
                "X-Softmax-Llm-Call-Id: 12345678-1234-1234-1234-123456789012\r\n".encode()
                + extra_headers
                + b"\r\n"
                + prefix
            )
            await writer.drain()
            observed.set()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    monkeypatch.setenv(
        "COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    )
    try:
        yield observed
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)


@pytest.mark.parametrize("suppress_close", [False, True])
async def test_actual_server_repeated_sigterm_joins_native_judge_and_seals(
    monkeypatch, tmp_path, suppress_close
):
    config = tmp_path / "signal-config.json"
    config.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "vacant"}] * 3,
                "judge_mode": "native",
                "turns_max": 1,
                "player_connect_timeout_seconds": 0,
            }
        )
    )
    output = tmp_path / "signal-private.jsonl"
    received = tmp_path / "received-prefix.json"
    # Observe the actual child-owned reader, rather than the fixture's earlier TCP write.
    runner = """
import os
import asyncio
from contextlib import suppress
import httpx
from pathlib import Path
if os.environ['TEST_SUPPRESS_CLOSE'] == '1':
    original_close = httpx.AsyncClient.aclose
    async def suppressing_close(self):
        await original_close(self)
        with suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        await asyncio.Event().wait()
    httpx.AsyncClient.aclose = suppressing_close
from gnomic.judge import LlmJudge
from gnomic.lifecycle import owned_task
original_invoke = LlmJudge._invoke
async def instrument(self, *args, **kwargs):
    async def observe():
        while not self.generations or self.generations[-1].response_body_b64 != 'eyJwYXJ0aWFsIjoiww==':
            await asyncio.sleep(.005)
        with os.fdopen(os.open(os.environ['TEST_RECEIVED_PREFIX'], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
            output.write(self.generations[-1].model_dump_json())
    watcher = owned_task(observe())
    try:
        return await original_invoke(self, *args, **kwargs)
    finally:
        await watcher
LlmJudge._invoke = instrument
from gnomic.server.app import main
main()
"""
    async with partial_response(monkeypatch, b'{"partial":"\xc3') as observed:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            runner,
            env={
                **os.environ,
                "COGAME_CONFIG_URI": config.as_uri(),
                "COGAME_SAVE_TRAJECTORY_URI": output.as_uri(),
                "COGAME_PORT": "0",
                "COWORLD_EPISODE_ID": "actual-signal-episode",
                "COWORLD_GAME_VERSION": "0.2.1",
                "COWORLD_SOURCE_REVISION": "a" * 40,
                "LOG_LEVEL": "error",
                "TEST_RECEIVED_PREFIX": str(received),
                "TEST_SUPPRESS_CLOSE": "1" if suppress_close else "0",
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(observed.wait(), 5)
            async with asyncio.timeout(5):
                while not received.exists():
                    await asyncio.sleep(0.01)
            snapshot = json.loads(received.read_bytes())
            assert snapshot["response_reader_joined"] is False
            assert snapshot["response_complete"] is False
            process.send_signal(signal.SIGTERM)
            await asyncio.sleep(0.01)
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        finally:
            if process.returncode is None:
                process.terminate()
            await process.wait()
    assert process.returncode != 0 if suppress_close else process.returncode == 0, (
        process.returncode,
        stdout,
        stderr,
    )
    episode = json.loads(output.read_bytes())
    assert episode["episode"]["status"] == "truncated"
    attempts = episode["episode"]["outcome"]["environment_generations"]
    assert attempts and attempts[-1]["response_reader_joined"] is (not suppress_close)
    assert attempts[-1]["response_complete"] is False
    assert base64.b64decode(attempts[-1]["response_body_b64"]) == b'{"partial":"\xc3'
    assert attempts[-1]["raw_response"] is None
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "status,body", [(500, b'{"error":"observed failure"}'), (200, b"")]
)
async def test_observed_error_or_empty_eof_remains_complete_received_body(
    monkeypatch, status, body
):
    generations = []
    async with partial_response(monkeypatch, body, status=status, complete=True):
        with pytest.raises(ValueError):
            await complete_native(
                {
                    "system": "private",
                    "messages": [],
                    "max_tokens": 1,
                    "temperature": 1,
                    "top_p": 1,
                },
                "fixture",
                purpose="environment",
                generations=generations,
                timeout=1,
            )
    attempt = generations[0]
    assert attempt.response_complete is True and attempt.response_reader_joined is True
    assert attempt.http_status == status
    assert attempt.response_body_b64 == base64.b64encode(body).decode()
    assert attempt.raw_response == body.decode()
    assert attempt.response is None


async def test_unsettled_server_reader_seals_once_without_public_artifacts(
    monkeypatch, tmp_path
):
    from contextlib import suppress

    from gnomic.lifecycle import settle
    from gnomic.server.app import GameServer
    from gnomic.server.channel import InProcessChannel
    from gnomic.server.episode import Episode

    release = asyncio.Event()
    readers = set()

    class UnsettledChannel(InProcessChannel):
        async def stop_player(self, deadline):
            return True

        async def close_reader(self):
            readers.add(asyncio.current_task())
            with suppress(asyncio.CancelledError):
                await release.wait()
            await release.wait()

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "judge_mode": "deterministic",
                "players": [{"name": "native"}] * 3,
            }
        )
    )
    private = tmp_path / "private.jsonl"
    results = tmp_path / "results.json"
    replay = tmp_path / "replay.json"
    for key, value in {
        "COGAME_CONFIG_URI": config.as_uri(),
        "COGAME_SAVE_TRAJECTORY_URI": private.as_uri(),
        "COGAME_RESULTS_URI": results.as_uri(),
        "COGAME_SAVE_REPLAY_URI": replay.as_uri(),
        "COWORLD_SOURCE_REVISION": "a" * 40,
        "COWORLD_GAME_VERSION": "0.2.1",
        "COWORLD_EPISODE_ID": "unsettled-reader",
    }.items():
        monkeypatch.setenv(key, value)

    async def completed(self):
        await self._send(
            self.channels[0],
            {"type": "action_request", "turn": 1, "rid": 1, "timeout_s": 1},
        )
        self.channels[0].consume_progress(
            {"type": "private_attempt", "rid": 1, "attempt": started.model_dump()}
        )
        self.results = {"scores": [0, 0, 0], "game_points": [0, 0, 0]}
        return self.results, {"events": []}

    monkeypatch.setattr(Episode, "run", completed)
    game = GameServer()
    channel = UnsettledChannel(0)
    channel.register_window(1)
    started = Attempt(
        policy="native",
        origin="model",
        prompt=[],
        request={"model": "native"},
        model="native",
        decoder={},
        response_body_b64="ww==",
        http_status=200,
        response_complete=False,
        response_reader_joined=False,
    )
    channel.consume_progress(
        {"type": "private_attempt", "rid": 1, "attempt": started.model_dump()}
    )
    game.channels[0] = channel
    game.registered_channels.append(channel)
    game.owner_deadline = asyncio.get_running_loop().time() + 0.05
    game._start_episode()
    await asyncio.wait_for(game.run_task, 1.3)
    frozen = private.read_bytes()
    captured = json.loads(frozen)
    assert captured["episode"]["status"] == "truncated"
    assert captured["decisions"][0]["attempts"][0]["response_body_b64"] == "ww=="
    assert captured["decisions"][0]["attempts"][0]["response_reader_joined"] is False
    assert not results.exists() and not replay.exists()
    assert game.results is None and game.replay is None
    assert not game.ownership_joined and channel.sealed
    channel.consume_progress({"rid": 1, "action": "late"})
    release.set()
    assert await settle(readers, asyncio.get_running_loop().time() + 0.2, cancel=False)
    await game.stop()
    assert private.read_bytes() == frozen
    assert not results.exists() and not replay.exists()


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":["PRIVATE_NATIVE_SCHEMA_SENTINEL"],"content":[{"type":"text","text":"PRIVATE_NATIVE_SCHEMA_SENTINEL"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}',
        b'{"malformed":"PRIVATE_NATIVE_SCHEMA_SENTINEL",',
    ],
)
async def test_malformed_native_validation_traceback_hides_private_response(
    monkeypatch, body
):
    import traceback

    generations = []
    async with partial_response(monkeypatch, body, complete=True):
        with pytest.raises(ValueError) as failure:
            await complete_native(
                {
                    "system": "private",
                    "messages": [],
                    "max_tokens": 8,
                    "temperature": 1,
                    "top_p": 1,
                },
                "fixture",
                purpose="environment",
                generations=generations,
                timeout=1,
            )
        attempt = generations[0]
    rendered = "".join(traceback.format_exception(failure.value))
    assert "PRIVATE_NATIVE_SCHEMA_SENTINEL" not in rendered
    assert attempt.raw_response == body.decode()
    assert base64.b64decode(attempt.response_body_b64) == body
    assert attempt.response_complete is True and attempt.response_reader_joined is True


def test_private_attempt_validation_traceback_hides_received_payload():
    import traceback

    from pydantic import ValidationError

    with pytest.raises(ValidationError) as failure:
        Attempt.model_validate(
            {
                "policy": "native",
                "origin": "model",
                "prompt": [],
                "request": {},
                "model": "native",
                "decoder": {},
                "sampled_token_ids": ["PRIVATE_INGRESS_SENTINEL"],
                "raw_response": "actual private body",
            }
        )
    assert "PRIVATE_INGRESS_SENTINEL" not in "".join(
        traceback.format_exception(failure.value)
    )


@pytest.mark.parametrize("model_path", ["judge", "player"])
def test_private_output_schema_traceback_hides_model_response(model_path):
    import traceback

    from pydantic import ValidationError

    from gnomic.judge import ActionRuling
    from gnomic.players.llm import ActionOutput

    model = ActionRuling if model_path == "judge" else ActionOutput
    with pytest.raises(ValidationError) as failure:
        model.model_validate_json(
            '{"action":["PRIVATE_OUTPUT_SCHEMA_SENTINEL"],"valid":"PRIVATE_OUTPUT_SCHEMA_SENTINEL"}'
        )
    assert "PRIVATE_OUTPUT_SCHEMA_SENTINEL" not in "".join(
        traceback.format_exception(failure.value)
    )


async def test_empty_completion_error_does_not_echo_private_stop_metadata(monkeypatch):
    import traceback

    from gnomic.llm_transport import Response
    from gnomic.players.llm import OpusPolicy

    async def complete(body, model, **kwargs):
        return Response(
            content=[],
            model=model,
            stop_reason="PRIVATE_STOP_METADATA_SENTINEL",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr("gnomic.players.llm.complete_native", complete)
    policy = OpusPolicy()
    with pytest.raises(ValueError, match="model returned no text content") as failure:
        await policy._invoke("private system", "private prompt")
    assert "PRIVATE_STOP_METADATA_SENTINEL" not in "".join(
        traceback.format_exception(failure.value)
    )


@pytest.mark.parametrize("case", ["early", "stale", "late", "duplicate"])
async def test_stopped_ack_is_bound_to_engine_nonce_and_absolute_window(case):
    channel = InProcessChannel(0)
    now = asyncio.get_running_loop().time()
    if case != "early":
        channel.stop_id = "engine-nonce"
        channel.stop_deadline = now + 1
    if case == "late":
        channel.stop_deadline = now - 1
    if case == "duplicate":
        channel.stopped.set()
    nonce = "other" if case == "stale" else "engine-nonce"
    with pytest.raises(ValueError, match="stopped acknowledgement"):
        channel.consume_progress({"type": "stopped", "stop_id": nonce})


@pytest.mark.parametrize("field", ["response_body_b64", "response_headers_b64"])
async def test_received_bytes_require_monotonic_prefix_and_freeze_at_eof(field):
    channel = InProcessChannel(0)
    channel.register_window(1)
    before = Attempt(
        policy="native",
        prompt=[],
        request={},
        model="native",
        decoder={},
        **{field: base64.b64encode(b"actual").decode()},
    )
    packet = {"type": "private_attempt", "rid": 1, "attempt": before.model_dump()}
    channel.consume_progress(packet)
    invalid = before.model_copy(update={field: base64.b64encode(b"rewrite").decode()})
    with pytest.raises(ValueError, match="cannot be rewritten"):
        channel.consume_progress({**packet, "attempt": invalid.model_dump()})
    final = before.model_copy(
        update={
            field: base64.b64encode(b"actual-complete").decode(),
            "response_complete": True,
        }
    )
    channel.consume_progress({**packet, "attempt": final.model_dump()})
    extended = final.model_copy(
        update={field: base64.b64encode(b"actual-complete-late").decode()}
    )
    with pytest.raises(ValueError, match="completed response"):
        channel.consume_progress({**packet, "attempt": extended.model_dump()})
    assert channel.evidence_for(1)[0] == final


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["eof", "final", "stalled"])
@pytest.mark.parametrize("surface", ["merchant", "sdk"])
async def test_joined_stop_transport_exit_without_final(monkeypatch, ending, surface):
    import json

    import websockets

    from gnomic import lifecycle
    from gnomic.players.client import run_policy
    from gnomic.players.sdk import Policy, run_ws_player

    monkeypatch.setattr(lifecycle, "CLEANUP_SECONDS", 0.2)
    connections = []
    received_final = []

    class Merchant:
        async def respond(self, message):
            received_final.append(message)
            return None

    class SdkPolicy(Policy):
        def on_message(self, view, message):
            received_final.append(message)

    async def game(connection):
        connections.append(connection)
        await connection.send(json.dumps({"type": "stop", "stop_id": "owned-stop"}))
        control = json.loads(await connection.recv())
        assert control == {"type": "stopped", "stop_id": "owned-stop"}
        if ending == "final":
            await connection.send(json.dumps({"type": "final", "scores": [1, 0, 0]}))
        if ending == "stalled":
            await connection.wait_closed()

    async with websockets.serve(game, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        work = (
            run_policy(Merchant(), url)
            if surface == "merchant"
            else run_ws_player(SdkPolicy(), url, max_attempts=1)
        )
        await asyncio.wait_for(lifecycle.run_owned(work), timeout=1)
    assert len(connections) == 1
    assert bool(received_final) == (ending == "final")
