"""Download progress on stderr, one look across morie, rmorie, rmoriebricklayer and rmoriedata.

A live line with a bar, percent, bytes and rate while stderr is a terminal; milestone lines
(every 10 %, or every 50 MB when the size is unknown) when it is not, so logs stay readable.
``MORIE_NO_PROGRESS=1`` silences it.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO

_SPIN = "|/-\\"


def _progress_off() -> bool:
    """``MORIE_NO_PROGRESS`` set to a true value (1, yes, true, on); ``0``/``false``/``no``/``off`` keep the bars."""
    return os.environ.get("MORIE_NO_PROGRESS", "").strip().lower() not in ("", "0", "false", "no", "off")


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class Progress:
    """``update(n)`` advances by ``n`` units; ``close()`` prints the total. Usable as a context manager."""

    def __init__(self, label: str, total: int | None = None, *, unit: str = "B", stream: IO[str] | None = None):
        self.label = label
        self.total = int(total) if total and total > 0 else None
        self.unit = unit
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = not _progress_off()
        self.tty = bool(self.enabled and getattr(self.stream, "isatty", lambda: False)())
        self.done = 0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._milestone = 0
        self._spin = 0
        self._width = 0
        if self.enabled and not self.tty:
            size = f" {self._fmt(self.total)}" if self.total else ""
            self._emit(f"{label}: downloading{size}\n")

    def _fmt(self, n: float) -> str:
        return fmt_bytes(n) if self.unit == "B" else f"{int(n):,} {self.unit}"

    def _line(self) -> str:
        elapsed = time.monotonic() - self.t0
        rate = f"{self._fmt(self.done / elapsed)}/s" if elapsed >= 0.5 else ""
        if self.total:
            pct = min(100, int(100 * self.done / self.total))
            filled = pct // 4
            return f"{self.label}  [{'#' * filled}{'.' * (25 - filled)}] {pct:3d}%  {self._fmt(self.done)} / {self._fmt(self.total)}  {rate}"
        self._spin += 1
        return f"{self.label}  {_SPIN[self._spin % 4]} {self._fmt(self.done)}  {rate}"

    def update(self, n: int) -> None:
        self.done += n
        if not self.enabled:
            return
        if self.tty:
            now = time.monotonic()
            if now - self._last < 0.1:
                return
            self._last = now
            line = self._line()
            self._emit("\r" + line + " " * max(0, self._width - len(line)))
            self._width = max(self._width, len(line))
            return
        if self.total:
            step = self.done * 10 // self.total
        else:
            step = self.done // (50 << 20) if self.unit == "B" else self.done // 100_000
        if step > self._milestone:
            self._milestone = step
            self._emit("  " + self._line() + "\n")

    def close(self) -> None:
        if not self.enabled:
            return
        if self.tty:
            self._emit("\r" + " " * self._width + "\r")
        self._emit(f"{self.label}: {self._fmt(self.done)} in {time.monotonic() - self.t0:.0f} s\n")

    def _emit(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:  # noqa: BLE001 - a closed stderr must never break a download
            self.enabled = False

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def run_step(cmd: list[str], label: str, *, env: dict | None = None, cwd: str | None = None) -> int:
    """Run a long subprocess (an R install, a pip upgrade) behind a spinner.

    On a terminal: one live line with a spinner, the elapsed seconds and the command's last
    output line; the full output is kept and shown only when the command fails. Off a
    terminal (logs, CI): the output streams through unchanged between a start and an end line.
    """
    import subprocess
    import tempfile

    stream = sys.stderr
    tty = bool(getattr(stream, "isatty", lambda: False)()) and not _progress_off()
    t0 = time.monotonic()
    if not tty:
        stream.write(f"-> {label} ...\n")
        stream.flush()
        rc = subprocess.run(cmd, env=env, cwd=cwd).returncode
        stream.write(f"{'ok' if rc == 0 else 'FAILED'}  {label} ({time.monotonic() - t0:.0f} s)\n")
        stream.flush()
        return rc
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        i = 0
        width = 0
        last = ""
        while proc.poll() is None:
            i += 1
            try:
                pos = log.tell()
                log.seek(max(0, pos - 400))
                tail = log.read().rstrip().splitlines()
                last = tail[-1].strip()[:60] if tail else last
                log.seek(0, 2)
            except (OSError, ValueError):
                pass
            line = f"  {_SPIN[i % 4]} {label}  {time.monotonic() - t0:.0f} s  {last}"
            stream.write("\r" + line + " " * max(0, width - len(line)))
            stream.flush()
            width = max(width, len(line))
            time.sleep(0.1)
        rc = proc.returncode
        stream.write("\r" + " " * width + "\r")
        stream.write(f"{'ok' if rc == 0 else 'FAILED'}  {label} ({time.monotonic() - t0:.0f} s)\n")
        if rc != 0:
            log.seek(0)
            tail = log.read().splitlines()[-30:]
            stream.write("".join("      " + t + "\n" for t in tail))
        stream.flush()
        return rc


def stream_to_file(resp, dest, label: str, chunk_size: int = 1 << 20, offset: int = 0) -> int:
    """Copy an HTTP response body to ``dest`` with progress; returns the bytes written.

    ``offset`` > 0 appends a resumed transfer (a 206 response) to what is already on disk.
    """
    try:
        total: int | None = int(resp.headers.get("Content-Length") or 0) or None
    except (AttributeError, TypeError, ValueError):
        total = None
    if total is not None and offset:
        total += offset
    with open(dest, "ab" if offset else "wb") as fh, Progress(label, total) as prog:
        if offset:
            prog.update(offset)
        while chunk := resp.read(chunk_size):
            fh.write(chunk)
            prog.update(len(chunk))
        if total is not None and prog.done < total:
            # a connection that closes early reads as a short body, not an error:
            # say so, or a truncated file would pass as complete
            from http.client import IncompleteRead

            raise IncompleteRead(b"", total - prog.done)
        return prog.done


def _no_route(exc: BaseException) -> bool:
    """True when the machine has no network or the host name does not resolve (nothing to resume)."""
    import errno
    import socket

    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.gaierror):
        return True
    return isinstance(reason, OSError) and reason.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH)


def download_url(
    url: str, dest, label: str, timeout: int = 60, headers: dict | None = None, attempts: int = 3, opener=None
) -> int:
    """Stream ``url`` to ``dest`` with progress; a transfer that drops part-way is resumed, up to ``attempts`` times.

    The retry asks for the remainder with an HTTP Range header; a server that ignores it
    (answers 200 instead of 206) is read again from the start. HTTP errors (404, 401, ...)
    are not retried. ``opener(request, timeout)`` replaces ``urlopen`` (tests, authenticated hubs).
    Returns the bytes on disk.
    """
    from http.client import IncompleteRead
    from pathlib import Path
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    dest = Path(dest)
    opener = opener or (lambda req, timeout: urlopen(req, timeout=timeout))
    written = 0
    for attempt in range(1, attempts + 1):
        hdrs = dict(headers or {})
        if written:
            hdrs["Range"] = f"bytes={written}-"
        try:
            # a plain URL when nothing is to be added, so callers' own openers see the string they expect
            with opener(Request(url, headers=hdrs) if hdrs else url, timeout) as resp:
                if written and getattr(resp, "status", 200) != 206:
                    written = 0  # the server ignored the range: start over
                return stream_to_file(resp, dest, label, offset=written)
        except HTTPError:
            raise
        except (IncompleteRead, ConnectionError, TimeoutError, OSError) as exc:
            if attempt == attempts or _no_route(exc):
                raise  # no network / no such host: resuming cannot help
            written = dest.stat().st_size if dest.exists() else 0
            sys.stderr.write(
                f"{label}: the transfer dropped ({type(exc).__name__}); "
                f"resuming from {fmt_bytes(written)}, attempt {attempt + 1} of {attempts}\n"
            )
            time.sleep(attempt)
    return written


class Stages:
    """Numbered stage lines on stderr for a long computation with no byte count to show.

    ``Stages("logistic-models", 4).step("fitting the interaction model")`` prints
    ``logistic-models [2/4] fitting the interaction model (12 s)``. ``MORIE_NO_PROGRESS=1``
    silences it, like the download bars.
    """

    def __init__(self, label: str, total: int, stream: IO[str] | None = None) -> None:
        self.label, self.total, self.n = label, total, 0
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = not _progress_off()
        self.t0 = time.monotonic()

    def step(self, what: str) -> None:
        self.n += 1
        if self.enabled:
            elapsed = time.monotonic() - self.t0
            count = f"[{self.n}/{self.total}] " if self.total else ""
            self.stream.write(f"{self.label} {count}{what} ({elapsed:.0f} s)\n")
            self.stream.flush()
