import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mbench import shim


class Upstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, format, *args):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part in ("data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"):
                self.wfile.write(part.encode())
                self.wfile.flush()
            return
        reply = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def post(port, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def test_chat_requests_get_the_run_effort_but_keep_client_settings():
    upstream = serve()
    server, port = shim.start(f"http://127.0.0.1:{upstream.server_address[1]}",
                              {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}})
    try:
        post(port, {"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": False}})
        assert Upstream.seen[-1]["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "xhigh"}
    finally:
        server.shutdown()
        upstream.shutdown()


def test_streams_pass_through_intact():
    upstream = serve()
    server, port = shim.start(f"http://127.0.0.1:{upstream.server_address[1]}", {"reasoning_effort": "high"})
    try:
        body = post(port, {"model": "m", "messages": [], "stream": True})
        assert body == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
        assert Upstream.seen[-1]["reasoning_effort"] == "high"
    finally:
        server.shutdown()
        upstream.shutdown()


def test_trim_drops_the_generated_token_and_rebuilds_offsets():
    payload = {"choices": [{"text": "The cat sat.\n\n", "logprobs": {
        "tokens": ["The", " cat", " sat", ".", "\n\n"], "token_logprobs": [None, -1.0, -2.0, -0.5, -0.1],
        "top_logprobs": [None, {}, {}, {}, {}], "text_offset": [-1, -1, -1, -1, -1]}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}
    trimmed = shim.trim_generated_token(payload)
    choice = trimmed["choices"][0]
    assert choice["text"] == "The cat sat."
    assert choice["logprobs"]["tokens"] == ["The", " cat", " sat", "."]
    assert choice["logprobs"]["text_offset"] == [0, 3, 7, 11]
    assert trimmed["usage"] == {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}
