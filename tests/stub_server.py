"""
Test-only stub for the inference service. NOT the serving engine.
SPEC: the serving engine is vLLM (or TGI). Start it with start_server.py.

This stub implements the same HTTP contract (/health, /v1/completions)
so task 1.1 tests can run without a live vLLM process.
Task 1.4: /health validates stack (vLLM + PEFT + single base model).
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from health_check import check_stack


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            ok, details = check_stack()
            status = 200 if ok else 503
            body = json.dumps(details).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") == "/v1/completions":
            body = json.dumps({"choices": [{"text": " completion"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run(port: int = 8000) -> HTTPServer:
    server = HTTPServer(("", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
