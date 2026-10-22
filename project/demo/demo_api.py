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
        self.send_header("Access-Control-Allow-Origin", "*")   # 2026-10-06: 비교 화면(/dual)이 다른 포트 서버(:8001)도 부른다
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802  — POST /load(JSON)의 CORS 사전 요청
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/dual" or self.path.startswith("/dual?"):
            # 2026-10-06: 2.4GHz·5GHz 비교 화면 — 이 서버(보통 :8000, 2.4GHz)와 옆 서버(:8001, 5GHz)를 한 화면에.
            self._send(200, (HERE / "demo_dual.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/assets_taste/"):
            # 2026-10-02: 화면이 쓰는 내장 글꼴·아이콘(오프라인 랩실에서도 동작). 파일 이름만 허용.
            name = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
            f = HERE / "assets_taste" / name
            ctype = {"woff2": "font/woff2", "txt": "text/plain; charset=utf-8"}.get(name.rsplit(".", 1)[-1])
            if ctype and "/" not in name and ".." not in name and f.is_file():
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, b"not found", "text/plain")
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
            self.send_header("Access-Control-Allow-Origin", "*")
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

    def _action(self, body: dict) -> None:
        """10-22 화면 버튼. do = find(후보 찾기, 쿨다운 무시) | apply(지난 결과 실행), kind = channel | 5g."""
        acts = demo_state._actions
        if acts is None:
            self._send(400, json.dumps({"ok": False, "msg": "실시간 MT6000 모드에서만 쓸 수 있어요"}, ensure_ascii=False).encode("utf-8"),
                       "application/json")
            return
        r = acts["runner"] if body.get("kind") == "channel" else acts["steer"]
        other = acts["steer"] if r is acts["runner"] else acts["runner"]
        if other.running:
            ok, msg = False, "다른 조치가 진행 중이에요"
        elif body.get("do") == "find":
            ok = r.start(force=True); msg = "" if ok else "이미 진행 중이에요"
        elif body.get("do") == "apply":
            ok = r.apply_last(); msg = "" if ok else "실행할 후보가 없어요"
        else:
            ok, msg = False, "알 수 없는 요청"
        self._send(200 if ok else 409, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False).encode("utf-8"), "application/json")

    def do_POST(self):  # noqa: N802
        if self.path == "/action":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            self._action(body)
            return
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
        try:
            duration_s = int(body["duration_s"]) if body.get("duration_s") is not None else None
        except (TypeError, ValueError):
            duration_s = None
        res = set_phone_load(str(body.get("phone", "")), str(body.get("rate", "off")), packet_size, duration_s)
        self._send(200 if res.get("ok") else 400,
                   json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")
