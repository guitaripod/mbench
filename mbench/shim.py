import itertools
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def rebuild_offsets(tokens, offsets):
    """SGLang reports text_offset as -1; log-likelihood scorers need real character offsets to find the continuation."""
    if offsets and any(offset != -1 for offset in offsets):
        return offsets
    return [0, *itertools.accumulate(len(token or "") for token in tokens[:-1])] if tokens else []


def trim_generated_token(payload):
    """Drops the one token SGLang must generate, so the reply matches an echo-only (max_tokens=0) completion."""
    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        last = tokens[-1] if tokens else ""
        for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
            if isinstance(logprobs.get(key), list) and logprobs[key]:
                logprobs[key] = logprobs[key][:-1]
        if "tokens" in logprobs:
            logprobs["text_offset"] = rebuild_offsets(logprobs["tokens"], logprobs.get("text_offset"))
        text = choice.get("text")
        if isinstance(text, str) and last and text.endswith(last):
            choice["text"] = text[: -len(last)]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        usage["total_tokens"] = usage.get("prompt_tokens", usage.get("total_tokens"))
        usage["completion_tokens"] = 0
    return payload


def inject_into(body, inject):
    """Adds the run's thinking setting to a request that lacks one; nested settings merge and anything the client sent wins."""
    for key, value in inject.items():
        if isinstance(value, dict):
            body[key] = {**value, **(body.get(key) or {})}
        else:
            body.setdefault(key, value)
    return body


class LogprobShim(BaseHTTPRequestHandler):
    """Sits between lmx and llama-swap: passes lmx's zero-token HellaSwag scoring through SGLang and gives chat requests the run's effort."""

    upstream = ""
    inject = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def read_body(self):
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def open_upstream(self, method, body=None):
        request = urllib.request.Request(self.upstream + self.path, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            return urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as error:
            print(f"shim: upstream {error.code} for {method} {self.path}", file=sys.stderr, flush=True)
            return error

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def relay_stream(self, response):
        """Passes a server-sent-event stream on line by line, so the client sees each token when the server produces it."""
        self.send_response(response.status)
        self.send_header("Content-Type", response.headers.get("Content-Type", "text/event-stream"))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        while line := response.readline():
            self.wfile.write(line)
            self.wfile.flush()

    def do_GET(self):
        response = self.open_upstream("GET")
        self.reply(getattr(response, "status", None) or response.code, response.read())

    def do_POST(self):
        body = self.read_body()
        path = self.path.rstrip("/")
        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            request = None
        trim = False
        streaming = False
        if isinstance(request, dict):
            if path.endswith("chat/completions") and self.inject:
                inject_into(request, self.inject)
                body = json.dumps(request).encode()
            elif path.endswith("/completions") and request.get("echo") and request.get("max_tokens") == 0:
                request.update(max_tokens=1, temperature=0)
                body = json.dumps(request).encode()
                trim = True
            streaming = bool(request.get("stream"))
        response = self.open_upstream("POST", body)
        status = getattr(response, "status", None) or response.code
        if streaming and status == 200:
            self.relay_stream(response)
            return
        payload = response.read()
        if trim and status == 200:
            payload = json.dumps(trim_generated_token(json.loads(payload))).encode()
        self.reply(status, payload)


def start(upstream, inject=None):
    """Serves the shim on a free loopback port chosen by the OS; returns the server (call shutdown()) and its port."""
    handler = type("BoundLogprobShim", (LogprobShim,), {"upstream": upstream.rstrip("/"), "inject": dict(inject or {})})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
