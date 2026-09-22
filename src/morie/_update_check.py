# SPDX-License-Identifier: AGPL-3.0-or-later
"""New-version detection and the ``morie update`` command.

``import morie`` performs a fail-silent, daily-cached check for a newer
release on PyPI and prints a one-line stderr notice when the installed
version is out of date.  The network request runs in a background
daemon thread, so it never slows ``import morie`` down -- the hot path
only reads a small cache file.

Opt out entirely with the environment variable ``MORIE_NO_UPDATE_CHECK``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

PYPI_JSON_URL = "https://pypi.org/pypi/morie/json"
_CHECK_INTERVAL = 24 * 60 * 60  # seconds between PyPI checks
_NET_TIMEOUT = 3.0
_NOTIFIED = False

__all__ = ["maybe_notify", "check_pypi_latest", "run_update"]


def _cache_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "morie", "update_check.json")


def _parse_version(s: str) -> tuple[int, ...]:
    """Version key: numeric components, then a pre-release marker.

    '0.9.0' -> (0, 9, 0, 1); '2.0.0rc1' -> (2, 0, 0, 0, 1). A pre-release
    (rc/a/b/dev suffix on the last numeric chunk) sorts BELOW the release
    with the same numbers, so a release candidate on PyPI is never
    advertised as a newer stable release than the one installed.
    """
    parts: list[int] = []
    pre: tuple[int, ...] = (1,)
    for chunk in str(s).split("."):
        digits = ""
        rest = ""
        for k, ch in enumerate(chunk):
            if ch.isdigit():
                digits += ch
            else:
                rest = chunk[k:]
                break
        if not digits:
            rest = chunk  # ".post1", ".rc1", ".dev0": the whole chunk is the tag
            if not re.match(r"(a|b|rc|dev|post)\d*$", rest):
                break
        else:
            parts.append(int(digits))
        if rest.startswith("+"):
            break  # a local label ("+unknown", "+g1234") ranks with its release
        if rest:
            m = re.match(r"(a|b|rc|dev|post)?(\d*)", rest)
            tag = (m.group(1) or "") if m else ""
            num = int(m.group(2) or 0) if m else 0
            pre = (2, num) if tag == "post" else (
                0, {"dev": 0, "a": 1, "b": 2, "rc": 3}.get(tag, 0), num)
            break
    return tuple(parts or [0]) + pre


def _read_cache() -> dict:
    try:
        with open(_cache_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(latest: str) -> None:
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"latest": latest, "last_check": time.time()}, fh)
    except OSError:
        pass


def check_pypi_latest(timeout: float = _NET_TIMEOUT) -> str | None:
    """Return morie's latest version on PyPI, or None on any failure."""
    import urllib.request

    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = data.get("info", {}).get("version")
        return latest if isinstance(latest, str) and latest else None
    except Exception:
        return None


def _refresh_cache_async() -> None:
    """Refresh the cached latest-version in a background daemon thread."""
    import threading

    def _worker() -> None:
        latest = check_pypi_latest()
        if latest:
            _write_cache(latest)

    threading.Thread(target=_worker, name="morie-update-check", daemon=True).start()


def maybe_notify(installed_version: str) -> None:
    """Print a one-line stderr notice if a newer morie release exists.

    Uses a daily-cached result, so there is no network call on the
    ``import morie`` hot path; a stale cache triggers a background
    refresh for next time.  Fail-silent, runs at most once per process,
    and is a no-op under ``MORIE_NO_UPDATE_CHECK``.
    """
    global _NOTIFIED
    if _NOTIFIED or os.environ.get("MORIE_NO_UPDATE_CHECK"):
        return
    _NOTIFIED = True

    installed = _parse_version(installed_version)
    if installed[:3] <= (0, 0, 0):  # dev / unknown install -- never nag
        return

    cache = _read_cache()
    latest = cache.get("latest")
    fresh = (time.time() - cache.get("last_check", 0)) < _CHECK_INTERVAL

    if isinstance(latest, str) and _parse_version(latest) > installed:
        sys.stderr.write(
            f"[morie] A newer version is available: {latest} "
            f"(you have {installed_version}).\n"
            f"        Update with `morie update` or `pip install -U morie`. "
            f"Silence with MORIE_NO_UPDATE_CHECK=1.\n"
        )

    if not fresh:
        _refresh_cache_async()


def run_update(yes: bool = False) -> int:
    """``morie update`` -- check PyPI and optionally upgrade in place."""
    try:
        import morie

        installed_version = getattr(morie, "__version__", "0.0.0+unknown")
    except Exception:
        installed_version = "0.0.0+unknown"

    print(f"morie {installed_version} -- checking PyPI for updates ...")
    latest = check_pypi_latest(timeout=10.0)
    if latest is None:
        print("Could not reach PyPI. Check your connection and try again.")
        return 1

    if _parse_version(latest) <= _parse_version(installed_version):
        print(f"morie is up to date (latest on PyPI: {latest}).")
        _write_cache(latest)
        return 0

    print(f"A newer version is available: {latest} (you have {installed_version}).")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "morie"]

    if not yes:
        try:
            reply = input("Update now? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Skipped. To update later, run:\n  " + " ".join(cmd))
            return 0

    import subprocess

    print("Running: " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        _write_cache(latest)
        print(f'Updated to morie {latest}. The R package updates via install.packages("morie").')
    return result.returncode
