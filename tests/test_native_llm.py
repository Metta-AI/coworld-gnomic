import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from gnomic.judge import LlmJudge
from gnomic.players.llm import OpusPolicy
from gnomic.players.haiku_baseline import LlmClient


def test_native_judge_and_players_route_without_aws_credentials(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("X-Coworld-Player-Slot"), payload))
            reply = json.dumps({"content": [{"type": "text", "text": "ok"}], "usage": {}}).encode()
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
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
    try:
        judge = LlmJudge("local-model")
        assert judge.model_id == "anthropic/claude-sonnet-4.6"
        assert judge._invoke([{"role": "user", "content": "proposal"}], slot=1)[0] == "ok"
        policy = OpusPolicy()
        assert policy.model == "anthropic/claude-sonnet-4.6"
        assert policy._invoke("rules", "action") == "ok"
        assert LlmClient().complete("rules", "action", max_tokens=64) == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == 3
    assert [slot for _, slot, _ in requests] == ["1", None, None]
    for path, _, payload in requests:
        assert path == "/v1/messages"
        assert payload["model"] == "anthropic/claude-sonnet-4.6"
        assert "anthropic_version" not in payload
        assert "output_config" not in payload
        assert "anthropic_beta" not in payload
