"""Real HTTP system checks for optional authenticated native learner serving."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from gnomic.llm_transport import CheckpointAccess, complete_native


KEY = "synthetic-native-checkpoint-canary"
MODEL = "checkpoint/" + "a" * 64
IDENTITY = {
    "model": MODEL,
    "checkpoint_sha256": "a" * 64,
    "tokenizer_sha256": "b" * 64,
    "chat_template_sha256": "c" * 64,
}
HEADER_FIELDS = {
    "X-Coworld-Checkpoint-Sha256": "checkpoint_sha256",
    "X-Coworld-Tokenizer-Sha256": "tokenizer_sha256",
    "X-Coworld-Chat-Template-Sha256": "chat_template_sha256",
}


async def call(body, purpose, attempts):
    response = await complete_native(
        body,
        MODEL,
        purpose=purpose,
        slot=2 if purpose == "learner" else None,
        timeout=2,
        generations=attempts,
    )
    return response.model_dump()


@pytest.mark.parametrize(
    "mode",
    [
        "accepted",
        "checkpoint_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "missing",
        "wrong-model",
        "wrong-key",
        "401",
        "redirect",
    ],
)
async def test_checkpoint_real_http_preserves_native_request_and_refuses_mismatch(
    monkeypatch, mode
):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/messages"
            assert self.headers["Authorization"] == "Bearer " + (
                "wrong" if mode == "wrong-key" else KEY
            )
            assert self.headers["X-Coworld-Player-Slot"] == "2"
            received.append(body)
            response = {
                "model": "other" if mode == "wrong-model" else MODEL,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            raw = json.dumps(response).encode()
            self.send_response(
                401
                if mode in {"wrong-key", "401"}
                else 302
                if mode == "redirect"
                else 200
            )
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if mode == "redirect":
                self.send_header("Location", "/credential-must-not-reach-this-route")
            if mode != "missing":
                for header, name in HEADER_FIELDS.items():
                    self.send_header(
                        header, "f" * 64 if mode == name else IDENTITY[name]
                    )
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", MODEL)
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "0")
    monkeypatch.setenv("COWORLD_LLM_TOP_P", "1")
    monkeypatch.setenv(
        "COWORLD_CHECKPOINT_API_KEY", "wrong" if mode == "wrong-key" else KEY
    )
    monkeypatch.setenv("COWORLD_CHECKPOINT_ATTESTATION", json.dumps(IDENTITY))
    body = {
        "system": "original native system",
        "messages": [{"role": "user", "content": "original private view"}],
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 1,
    }
    attempts = []
    try:
        async with asyncio.timeout(5):
            if mode == "accepted":
                response = await call(body, "learner", attempts)
                assert response["model"] == MODEL
                assert attempts[0].response == "ok"
                assert attempts[0].model_identity == IDENTITY["checkpoint_sha256"]
                assert attempts[0].tokenizer_identity == IDENTITY["tokenizer_sha256"]
                assert (
                    attempts[0].chat_template_sha256 == IDENTITY["chat_template_sha256"]
                )
            else:
                with pytest.raises(ValueError):
                    await call(body, "learner", attempts)
                assert attempts[0].response is None and not attempts[0].accepted
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert not thread.is_alive()
    assert received == [{**body, "model": MODEL}], (
        "No redirect, fallback provider or retry may forward credentials"
    )
    assert (
        attempts[0].response_complete is True
        and attempts[0].response_reader_joined is True
    )
    assert KEY not in attempts[0].model_dump_json()
    assert KEY not in repr(CheckpointAccess.from_environment(MODEL))


@pytest.mark.parametrize(
    "mode",
    [
        "missing-key",
        "missing-attestation",
        "invalid-key",
        "invalid-json",
        "extra-field",
        "bad-hash",
        "request-model",
        "endpoint-userinfo",
    ],
)
async def test_invalid_checkpoint_configuration_refuses_before_http(monkeypatch, mode):
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("COWORLD_LLM_MODEL", MODEL)
    monkeypatch.setenv("COWORLD_CHECKPOINT_API_KEY", KEY)
    identity = dict(IDENTITY)
    if mode == "extra-field":
        identity["extra"] = "forbidden"
    if mode == "bad-hash":
        identity["tokenizer_sha256"] = "bad"
    if mode == "request-model":
        identity["model"] = "different"
    monkeypatch.setenv(
        "COWORLD_CHECKPOINT_ATTESTATION",
        "[" if mode == "invalid-json" else json.dumps(identity),
    )
    if mode == "missing-key":
        monkeypatch.delenv("COWORLD_CHECKPOINT_API_KEY")
    if mode == "missing-attestation":
        monkeypatch.delenv("COWORLD_CHECKPOINT_ATTESTATION")
    if mode == "invalid-key":
        monkeypatch.setenv("COWORLD_CHECKPOINT_API_KEY", KEY + "\n")
    if mode == "endpoint-userinfo":
        monkeypatch.setenv("COWORLD_LLM_ENDPOINT", "http://user@127.0.0.1:1")
    body = {
        "system": "original",
        "messages": [],
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 1,
    }
    with pytest.raises(ValueError):
        await call(body, "learner", [])


async def test_environment_call_ignores_learner_checkpoint_configuration(monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers.get("Authorization") is None
            assert self.headers.get("X-Coworld-Player-Slot") is None
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            raw = json.dumps(
                {
                    "model": MODEL,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_CHECKPOINT_API_KEY", KEY)
    monkeypatch.setenv(
        "COWORLD_CHECKPOINT_ATTESTATION", "invalid ignored learner configuration"
    )
    body = {
        "system": "frozen elder",
        "messages": [],
        "max_tokens": 128,
        "temperature": 1,
        "top_p": 1,
    }
    try:
        response = await call(body, "environment", [])
        assert response["content"][0]["text"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert not thread.is_alive() and received == [{**body, "model": MODEL}]
