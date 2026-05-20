from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from jumpshot_predictor.service import add_session, get_summary, score_jumpshot


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


class JumpshotHandler(BaseHTTPRequestHandler):
    server_version = "JumpshotCreator/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/summary":
            self._send_json(get_summary())
            return
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/jumpshot_predictor/exploit_chart.png":
            self._send_file(ROOT / "jumpshot_predictor" / "exploit_chart.png")
            return
        self._send_static(path)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/jumpshot/score":
                self._send_json(score_jumpshot(payload))
                return
            if path == "/api/sessions":
                self._send_json(add_session(payload), status=HTTPStatus.CREATED)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_json({"error": f"server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        if path in ("", "/"):
            target = FRONTEND_DIR / "index.html"
        else:
            target = (FRONTEND_DIR / path.lstrip("/")).resolve()
            if FRONTEND_DIR.resolve() not in target.parents:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
        self._send_file(target)

    def _send_file(self, target: Path) -> None:
        if not target.exists() or not target.is_file():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), JumpshotHandler)
    print(f"NBA 2K26 jumpshot creator running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()

