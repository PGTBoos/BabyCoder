"""
tests/mock_server.py

A scripted stand-in for LM Studio, speaking the streaming (SSE) protocol the
toolkit uses. No model needed.

    server = MockLMStudio().start()
    configure_model(url=server.url)
    server.script = [reply(content="hello"), reply(reasoning="hmm...", content="hi")]
    ...
    server.seen[-1]   # the last request payload

Each request pops the next scripted reply. A reply can carry reasoning
(sent on the reasoning_content channel before the content), content, and a
status: status=400 answers with an error instead, the way LM Studio answers a
payload key it does not accept.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply(content="", reasoning="", status=200, reject_if=None):
    """One scripted answer. reject_if(payload) -> True answers 400 instead,
    without using up this reply."""
    return {"content": content, "reasoning": reasoning, "status": status, "reject_if": reject_if}


def envelope(calls=(), final=None, notes="n"):
    """The structured-output JSON a model returns in structured mode."""
    return json.dumps({"notes": notes, "calls": list(calls), "final_answer": final})


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class MockLMStudio:
    def __init__(self):
        self.script = []
        self.seen = []
        # Payload keys to refuse with a 400, the way LM Studio refuses a
        # thinking switch it does not know. Checked before the script.
        self.reject = lambda payload: False
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def handle(self):
                # The toolkit hangs up mid-stream on purpose (runaway thinking,
                # someone typing). That is the behaviour under test, not noise
                # worth a traceback.
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError):
                    pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.seen.append(payload)
                if server.reject(payload):
                    return self._error(400, b"unsupported")
                step = server.script.pop(0) if server.script else reply("ready")
                if step["status"] != 200:
                    return self._error(step["status"], b"scripted error")
                self._stream(step["reasoning"], step["content"])

            def _error(self, status, body):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _chunk(self, raw):
                self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                self.wfile.flush()

            def _event(self, delta, finish=None):
                data = json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
                self._chunk(f"data: {data}\n\n".encode())

            def _stream(self, reasoning, content):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for i in range(0, len(reasoning), 40):
                        self._event({"reasoning_content": reasoning[i:i + 40]})
                        time.sleep(0.001)
                    for i in range(0, len(content), 20):
                        self._event({"content": content[i:i + 20]})
                        time.sleep(0.001)
                    self._event({}, "stop")
                    self._chunk(b"data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass   # the client walked away on purpose (runaway, interrupt)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/v1/chat/completions"

    def start(self):
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self._httpd.shutdown()
