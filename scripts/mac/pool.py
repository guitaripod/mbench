"""A concurrent front for mlx_lm.server, which serves one request at a time. Starts a pool of servers, each holding
its own copy of the model, and hands every request to a free one. Quality runs send four at once; without this they
queue behind each other for hours."""
import http.server
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8085
CONTEXT = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
BASE = PORT + 100
PYTHON = os.path.expanduser("~/Dev/mlx-serve/.venv/bin/python")
LOGS = os.path.expanduser("~/Dev/mlx-serve/logs")

try:
    from importlib.metadata import version as _version
    VERSION = _version("mlx-lm")
except Exception:
    VERSION = "unknown"

os.makedirs(LOGS, exist_ok=True)
free = queue.Queue()
children = []
WORKER_TIMEOUT_S = 7200


def revive(index, port):
    """A worker that died takes its port with it. mlx_lm.server dies on a Metal resource limit and never comes back
    on its own, and a run that waits on a dead worker waits forever, so the worker is started again and the request
    that found it dead is failed rather than hung."""
    child = children[index] if index < len(children) else None
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except Exception:
            child.kill()
    start_worker(index)
    return wait_for(port, seconds=900)


def start_worker(index):
    port = BASE + index
    log = open(f"{LOGS}/worker-{index}.log", "w")
    child = subprocess.Popen(
        [PYTHON, os.path.expanduser("~/Dev/mlx-serve/serve.py"), MODEL, str(port)],
        stdout=log, stderr=subprocess.STDOUT)
    while len(children) <= index:
        children.append(None)
    children[index] = child
    return port


def wait_for(port, seconds=1800):
    deadline = time.time() + seconds
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self.reply(200, b'{"status":"ok"}')
            return
        if self.path.rstrip("/") == "/props":
            self.reply(200, json.dumps({
                "version": VERSION,
                "build_info": f"mlx-lm {VERSION}",
                "model_path": MODEL,
                "total_slots": WORKERS,
                "default_generation_settings": {"n_ctx": CONTEXT},
            }).encode())
            return
        if self.path.rstrip("/").endswith("models"):
            self.forward(body=None)
            return
        self.reply(404, json.dumps({"error": {"message": f"no route for {self.path}"}}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.forward(body=self.rfile.read(length))

    def forward(self, body):
        index, port = free.get()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}{self.path}", data=body,
                headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
            with urllib.request.urlopen(request, timeout=WORKER_TIMEOUT_S) as response:
                streaming = "text/event-stream" in (response.headers.get("Content-Type") or "")
                self.send_response(200)
                self.send_header("Content-Type", response.headers.get("Content-Type") or "application/json")
                if streaming:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for line in response:
                        self.wfile.write(b"%x\r\n" % len(line) + line + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    payload = response.read()
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.reply(error.code, payload or json.dumps({"error": str(error)}).encode())
        except Exception as error:
            self.reply(502, json.dumps({"error": {"message": f"{type(error).__name__}: {error}"}}).encode())
            if children[index] is None or children[index].poll() is not None:
                print(f"worker {index} on {port} died; starting it again", flush=True)
                revive(index, port)
        finally:
            free.put((index, port))

    def reply(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


class Pool(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def shutdown(*_):
    for child in children:
        if child is not None:
            child.terminate()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    ports = [start_worker(index) for index in range(WORKERS)]
    for index, port in enumerate(ports):
        if not wait_for(port):
            print(f"worker on {port} never came up", flush=True)
            shutdown()
        free.put((index, port))
        print(f"worker ready on {port}", flush=True)
    print(f"pool of {WORKERS} listening on {PORT} for {MODEL}", flush=True)
    Pool(("0.0.0.0", PORT), Handler).serve_forever()
