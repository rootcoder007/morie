"""A transfer that drops part-way is resumed with a Range request; a server that ignores Range is read again from the start."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from morie._progress import download_url

BODY = bytes(range(256)) * 400  # 102,400 bytes


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/file.bin"


class _Dropping(BaseHTTPRequestHandler):
    """First request: promises everything, sends half, closes. Second: honours Range with a 206."""

    hits: list[str] = []

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        self.hits.append(rng or "")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            rest = BODY[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(rest)))
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            self.end_headers()
            self.wfile.write(rest)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY[: len(BODY) // 2])
        self.wfile.flush()
        self.close_connection = True
        self.connection.close()


class _IgnoresRange(_Dropping):
    """Drops once, then serves the whole body with a 200 whatever the Range header says."""

    hits: list[str] = []

    def do_GET(self):
        self.hits.append(self.headers.get("Range") or "")
        if len(self.hits) == 1:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY[:1000])
            self.wfile.flush()
            self.close_connection = True
            self.connection.close()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)


def test_dropped_transfer_is_resumed_with_range(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MORIE_NO_PROGRESS", "1")
    monkeypatch.setattr("time.sleep", lambda s: None)
    _Dropping.hits = []
    srv, url = _serve(_Dropping)
    try:
        dest = tmp_path / "f.bin"
        n = download_url(url, dest, "f", timeout=10)
    finally:
        srv.shutdown()
    assert n == len(BODY)
    assert dest.read_bytes() == BODY
    assert _Dropping.hits[0] == "" and _Dropping.hits[1].startswith("bytes=")
    assert "resuming from" in capsys.readouterr().err


def test_server_that_ignores_range_is_read_again_from_the_start(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_NO_PROGRESS", "1")
    monkeypatch.setattr("time.sleep", lambda s: None)
    _IgnoresRange.hits = []
    srv, url = _serve(_IgnoresRange)
    try:
        dest = tmp_path / "g.bin"
        n = download_url(url, dest, "g", timeout=10)
    finally:
        srv.shutdown()
    assert n == len(BODY)
    assert dest.read_bytes() == BODY
    assert len(_IgnoresRange.hits) == 2


def test_http_errors_are_not_retried(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    class _NotFound(BaseHTTPRequestHandler):
        hits = 0

        def log_message(self, *a):
            pass

        def do_GET(self):
            _NotFound.hits += 1
            self.send_error(404)

    srv, url = _serve(_NotFound)
    try:
        with pytest.raises(HTTPError):
            download_url(url, tmp_path / "h.bin", "h", timeout=10)
    finally:
        srv.shutdown()
    assert _NotFound.hits == 1
