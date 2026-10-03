import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from gnomic.judge import LlmJudge
from gnomic.llm_transport import LearnerWindow, current_window
from gnomic.players.haiku_baseline import LlmClient
from gnomic.players.llm import OpusPolicy


def test_native_judge_and_players_route_without_aws_credentials(monkeypatch):
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
        assert judge._invoke([{"role": "user", "content": "proposal"}])[0] == "ok"
        policy = OpusPolicy()
        assert policy.model == "anthropic/claude-sonnet-4.6"
        assert policy._invoke("rules", "action") == "ok"
        token = current_window.set(
            LearnerWindow(1, "haiku", time.monotonic() + 5, lambda attempt: None)
        )
        try:
            assert LlmClient().complete("rules", "action", max_tokens=64) == "ok"
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
