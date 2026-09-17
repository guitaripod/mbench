"""One MLX model, one request at a time, over the OpenAI shape mbench speaks.

mlx_lm.server batches requests through a generator that keeps buffers alive across them; after a few dozen long
answers the Metal device's resource limit (499000 live buffers) is reached and the generation thread dies while the
process stays up, so every later request hangs forever. This serves one request at a time and clears MLX's cache
after each, which keeps the buffer count flat. Concurrency comes from running several of these behind pool.py.
"""
import http.server
import json
import sys
import time
import uuid

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

MODEL_PATH = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8185

model, tokenizer = load(MODEL_PATH)
print(f"loaded {MODEL_PATH}", flush=True)


def render(messages, tools):
    """The prompt the model sees, with the tools the request offered, through the model's own chat template."""
    return tokenizer.apply_chat_template(readable(messages), tools=tools, add_generation_prompt=True)


def readable(messages):
    """OpenAI sends a tool call's arguments as a JSON string; a chat template expects the mapping it encodes, and
    refuses the whole conversation otherwise — which would score every tool episode zero."""
    found = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            found.append(message)
            continue
        rewritten = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    function["arguments"] = {}
            rewritten.append({**call, "function": function})
        found.append({**message, "tool_calls": rewritten})
    return found


def split_thinking(text):
    """Thinking apart from the answer, the way a server with a reasoning parser reports it."""
    if not tokenizer.has_thinking:
        return "", text
    start, end = tokenizer.think_start, tokenizer.think_end
    if end not in text:
        return (text.replace(start, ""), "") if start in text else ("", text)
    thinking, answer = text.split(end, 1)
    return thinking.replace(start, "").strip(), answer.strip()


def split_tool_calls(text):
    """Structured calls out of the text the model wrote, using the parser the tokenizer names for this family."""
    if not tokenizer.has_tool_calling or tokenizer.tool_call_start not in text:
        return text, []
    head, rest = text.split(tokenizer.tool_call_start, 1)
    body = rest.split(tokenizer.tool_call_end, 1)[0] if tokenizer.tool_call_end else rest
    try:
        parsed = tokenizer.tool_parser(body)
    except Exception:
        return text, []
    calls = parsed if isinstance(parsed, list) else [parsed]
    found = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or call
        arguments = function.get("arguments")
        found.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": function.get("name"),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
            },
        })
    return head, found


def answer(body):
    messages = body.get("messages") or []
    tools = body.get("tools")
    prompt = render(messages, tools)
    sampler = make_sampler(
        temp=float(body.get("temperature") or 0.0),
        top_p=float(body.get("top_p") or 0.0),
        top_k=int(body.get("top_k") or 0),
    )
    limit = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
    text, prompt_tokens, completion_tokens, finish = "", 0, 0, "stop"
    for response in stream_generate(model, tokenizer, prompt, max_tokens=limit, sampler=sampler):
        text += response.text
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.generation_tokens
        finish = response.finish_reason or finish
    mx.clear_cache()
    thinking, visible = split_thinking(text)
    visible, calls = split_tool_calls(visible)
    message = {"role": "assistant", "content": visible.strip()}
    if thinking:
        message["reasoning_content"] = thinking
    if calls:
        message["tool_calls"] = calls
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_PATH,
        "choices": [{"index": 0, "message": message, "finish_reason": "length" if finish == "length" else finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        """Only the two paths this server really answers. A server that says something to every GET is read by a
        client as whichever kind of server asked first, and mbench then believes it has no slots."""
        path = self.path.rstrip("/")
        if path.endswith("models"):
            self.reply(200, {"object": "list", "data": [{"id": MODEL_PATH, "object": "model"}]})
        elif path in ("", "/health"):
            self.reply(200, {"status": "ok"})
        else:
            self.reply(404, {"error": {"message": f"no route for {self.path}"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.reply(400, {"error": {"message": "the request body is not JSON"}})
            return
        try:
            self.reply(200, answer(body))
        except Exception as error:
            mx.clear_cache()
            self.reply(500, {"error": {"message": f"{type(error).__name__}: {error}"}})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        return


if __name__ == "__main__":
    print(f"serving {MODEL_PATH} on {PORT}", flush=True)
    http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
