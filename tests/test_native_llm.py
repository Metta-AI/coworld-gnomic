import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from gnomic.judge import LlmJudge
from gnomic.llm_transport import LearnerWindow, complete_native, current_window
from gnomic.players.haiku_baseline import LlmClient
from gnomic.players.llm import OpusPolicy


async def test_native_judge_and_players_route_without_aws_credentials(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                (self.path, self.headers.get("X-Coworld-Player-Slot"), payload)
            )
            reply = json.dumps(
                {
                    "model": payload["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/"
    )
    monkeypatch.setenv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
    try:
        judge = LlmJudge("local-model")
        assert judge.model_id == "local-model"
        assert (await judge._invoke([{"role": "user", "content": "proposal"}]))[
            0
        ] == "ok"
        policy = OpusPolicy()
        assert policy.model == "anthropic/claude-sonnet-4.6"
        assert await policy._invoke("rules", "action") == "ok"

        async def publish(attempt):
            pass

        token = current_window.set(
            LearnerWindow(1, "haiku", time.monotonic() + 5, publish)
        )
        try:
            assert await LlmClient().complete("rules", "action", max_tokens=64) == "ok"
        finally:
            current_window.reset(token)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == 3
    assert [slot for _, slot, _ in requests] == [None, "0", "1"]
    for path, _, payload in requests:
        assert path == "/v1/messages"
        assert payload["model"] in {"local-model", "anthropic/claude-sonnet-4.6"}
        assert "anthropic_version" not in payload
        assert "task_budget" not in payload.get("output_config", {})
        assert "anthropic_beta" not in payload


def test_native_opus_defaults_preserve_game_model_family(monkeypatch):
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", "http://sidecar")
    monkeypatch.delenv("COWORLD_LLM_MODEL", raising=False)
    assert LlmJudge().model_id == "anthropic/claude-opus-4.7"
    assert OpusPolicy().model == "anthropic/claude-opus-4.7"
    assert LlmClient().model_id == "anthropic/claude-haiku-4.5"


@pytest.mark.parametrize(
    "model,purpose,temperature,top_p,checkpoint,expected_status",
    [
        ("anthropic/claude-opus-4.7", "environment", 1, 1, False, 200),
        ("anthropic/claude-opus-4.8", "learner", 1, 1, False, 200),
        ("anthropic/claude-opus-4.7", "learner", 0, 1, False, 404),
        ("anthropic/claude-opus-4.7", "environment", 1, 0.9, False, 404),
        ("anthropic/claude-sonnet-4.6", "learner", 0.3, 0.9, False, 200),
        ("other/model", "learner", 1, 1, False, 200),
        ("anthropic/claude-opus-4.7", "learner", 1, 1, True, 200),
    ],
)
async def test_real_http_legacy_defaults_preserve_controls_and_original_evidence(
    monkeypatch, model, purpose, temperature, top_p, checkpoint, expected_status
):
    requests = []
    attempts = []
    identity = {
        "model": model,
        "checkpoint_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            assert self.headers["anthropic-version"] == "2023-06-01"
            strict = model.startswith("anthropic/claude-opus-4.") and not checkpoint
            status = 404 if strict and {"temperature", "top_p"} & set(payload) else 200
            response = {
                "model": model,
                "content": [{"type": "text", "text": "synthetic HTTP fixture"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            raw = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            if checkpoint:
                assert (
                    self.headers["Authorization"]
                    == "Bearer synthetic-routing-checkpoint-key"
                )
                for header, key in [
                    ("X-Coworld-Checkpoint-Sha256", "checkpoint_sha256"),
                    ("X-Coworld-Tokenizer-Sha256", "tokenizer_sha256"),
                    ("X-Coworld-Chat-Template-Sha256", "chat_template_sha256"),
                ]:
                    self.send_header(header, identity[key])
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", model)
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", str(temperature))
    monkeypatch.setenv("COWORLD_LLM_TOP_P", str(top_p))
    if checkpoint:
        monkeypatch.setenv(
            "COWORLD_CHECKPOINT_API_KEY", "synthetic-routing-checkpoint-key"
        )
        monkeypatch.setenv("COWORLD_CHECKPOINT_ATTESTATION", json.dumps(identity))
    else:
        monkeypatch.delenv("COWORLD_CHECKPOINT_API_KEY", raising=False)
        monkeypatch.delenv("COWORLD_CHECKPOINT_ATTESTATION", raising=False)
    body = {
        "system": "Synthetic native routing fixture",
        "messages": [{"role": "user", "content": "fixture"}],
        "max_tokens": 8192,
        "temperature": temperature,
        "top_p": top_p,
    }
    original = json.loads(json.dumps(body))
    try:
        if expected_status == 404:
            with pytest.raises(ValueError, match="status 404"):
                await complete_native(
                    body,
                    model,
                    purpose=purpose,
                    slot=0 if purpose == "learner" else None,
                    timeout=2,
                    generations=attempts,
                )
        else:
            response = await complete_native(
                body,
                model,
                purpose=purpose,
                slot=0 if purpose == "learner" else None,
                timeout=2,
                generations=attempts,
            )
            assert response.text == "synthetic HTTP fixture"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive() and len(requests) == len(attempts) == 1
    assert body == original
    assert attempts[0].request == requests[0]
    assert attempts[0].http_status == expected_status
    for name, value in [("temperature", temperature), ("top_p", top_p)]:
        if checkpoint or not model.startswith("anthropic/claude-") or value != 1:
            assert requests[0][name] == value
        else:
            assert name not in requests[0]
