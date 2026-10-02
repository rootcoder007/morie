"""Download progress: milestone lines off a terminal, silence under MORIE_NO_PROGRESS, streaming copies."""

from __future__ import annotations

import io

from morie._progress import Progress, fmt_bytes, stream_to_file


class _Resp:
    def __init__(self, payload: bytes, length: int | None):
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(length)} if length is not None else {}

    def read(self, n: int) -> bytes:
        return self._buf.read(n)


def test_fmt_bytes():
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(1536) == "1.5 KB"
    assert fmt_bytes(583 * 1024 * 1024) == "583.0 MB"


def test_milestones_off_a_terminal(monkeypatch):
    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    out = io.StringIO()
    with Progress("ocp21", total=1000, stream=out) as p:
        for _ in range(10):
            p.update(100)
    text = out.getvalue()
    assert text.startswith("ocp21: downloading 1000 B\n")
    assert " 50%" in text and "100%" in text
    assert text.rstrip().endswith("s") and "ocp21: 1000 B in" in text


def test_unknown_total_counts_rows(monkeypatch):
    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    out = io.StringIO()
    with Progress("ckan", unit="rows", stream=out) as p:
        p.update(250_000)
    assert "250,000 rows" in out.getvalue()


def test_silenced(monkeypatch):
    monkeypatch.setenv("MORIE_NO_PROGRESS", "1")
    out = io.StringIO()
    with Progress("x", total=10, stream=out) as p:
        p.update(10)
    assert out.getvalue() == ""


def test_stream_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MORIE_NO_PROGRESS", "1")
    payload = b"a" * (3 * 1024 * 1024 + 7)
    dest = tmp_path / "f.bin"
    n = stream_to_file(_Resp(payload, len(payload)), dest, "f.bin", chunk_size=1 << 20)
    assert n == len(payload) and dest.read_bytes() == payload


def test_run_step_off_a_terminal(monkeypatch, capsys):
    import sys

    from morie._progress import run_step

    monkeypatch.delenv("MORIE_NO_PROGRESS", raising=False)
    rc = run_step([sys.executable, "-c", "print('hello from the step')"], "a quick step")
    err = capsys.readouterr().err
    assert rc == 0 and err.startswith("-> a quick step ...") and "ok  a quick step (" in err
    rc = run_step([sys.executable, "-c", "raise SystemExit(3)"], "a failing step")
    assert rc == 3 and "FAILED  a failing step" in capsys.readouterr().err
