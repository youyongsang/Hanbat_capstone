"""HTTP API — `Handler`(BaseHTTPRequestHandler)만 여기 있다.

GET/POST 라우팅과 요청/응답 직렬화만 하고, 실제 로직(부하 제어·연결확인)은 demo_load.py로,
공유 상태 읽기는 demo_state.py로 전부 위임한다. 엔드포인트 계약은 API.md 참고.
"""
from __future__ import annotations

import json
import queue
from http.server import BaseHTTPRequestHandler

import demo_state
from demo_state import HERE, DEFAULT_PACKET_SIZE
from demo_load import check_connectivity, phone_signals, set_phone_load


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            with demo_state._lock:
                self._send(200, json.dumps({"ready": demo_state._state.get("ready")}).encode(), "application/json")
        elif self.path == "/signal":
            self._send(200, json.dumps(phone_signals(), ensure_ascii=False).encode("utf-8"), "application/json")
        elif self.path == "/check":
            self._send(200, json.dumps(check_connectivity(), ensure_ascii=False).encode("utf-8"), "application/json")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=50)
            with demo_state._lock:
                demo_state._clients.append(q)
                snap = dict(demo_state._state)
            try:
                self.wfile.write(f"data: {json.dumps(snap, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        chunk = q.get(timeout=15)
                        self.wfile.write(chunk)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with demo_state._lock:
                    if q in demo_state._clients:
                        demo_state._clients.remove(q)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/load":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        try:
            packet_size = int(body.get("packet_size", DEFAULT_PACKET_SIZE))
        except (TypeError, ValueError):
            packet_size = -1  # set_phone_load 가 ALLOWED_PACKET_SIZES 로 걸러서 400 반환
        res = set_phone_load(str(body.get("phone", "")), str(body.get("rate", "off")), packet_size)
        self._send(200 if res.get("ok") else 400,
                   json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")
