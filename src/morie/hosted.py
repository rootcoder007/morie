"""The hosted MORIE inference tier at llm.rmorie.com.

An authenticated, rate-limited OpenAI-compatible endpoint run by the
project (LiteLLM in front of an Ollama server). It is the second stop in
the provider chain, after a local Ollama and before any cloud API key,
and it only ever speaks when the user has logged in: ``morie login`` runs
the GitHub device flow, receives a per-user key, and stores it with mode
0600 under the XDG config directory. Nothing is sent anywhere without
that key, and no anonymous endpoint is contacted.

Environment
-----------
MORIE_HOSTED_BASE_URL
    Override the endpoint (``https://llm.rmorie.com/v1``). An empty string
    disables the tier.
MORIE_HOSTED_KEY
    Override the stored key (CI, containers).
MORIE_HOSTED_MODEL
    Model name; the gateway only serves the cloud models of the server.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import time
import webbrowser
from pathlib import Path

import httpx

DEFAULT_HOSTED_BASE_URL = "https://llm.rmorie.com/v1"
DEFAULT_HOSTED_AUTH_URL = "https://llm.rmorie.com/auth"
DEFAULT_HOSTED_MODEL = "minimax-m3:cloud"
_PROBE_TIMEOUT = 2.0


def hosted_base_url() -> str | None:
    """Return the hosted endpoint, or None when disabled via an empty override."""
    if "MORIE_HOSTED_BASE_URL" in os.environ:
        url = os.environ["MORIE_HOSTED_BASE_URL"].strip()
        return url.rstrip("/") or None
    return DEFAULT_HOSTED_BASE_URL


def hosted_auth_url() -> str:
    return os.environ.get("MORIE_HOSTED_AUTH_URL", DEFAULT_HOSTED_AUTH_URL).strip().rstrip("/")


def hosted_model() -> str:
    return os.environ.get("MORIE_HOSTED_MODEL", DEFAULT_HOSTED_MODEL).strip() or DEFAULT_HOSTED_MODEL


def credentials_path() -> Path:
    """``$XDG_CONFIG_HOME/morie/credentials.json`` (shared with the R package)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "morie" / "credentials.json"


def read_credentials() -> dict:
    try:
        with open(credentials_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_credentials(data: dict) -> Path:
    """Write the credentials file with owner-only permissions."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def hosted_key() -> str | None:
    """The user's key: MORIE_HOSTED_KEY, else the stored credential."""
    env = os.environ.get("MORIE_HOSTED_KEY", "").strip()
    if env:
        return env
    key = read_credentials().get("hosted_key")
    return key.strip() if isinstance(key, str) and key.strip() else None


_hosted_cached: bool | None = None


def reset_probe_cache() -> None:
    global _hosted_cached
    _hosted_cached = None


def probe_hosted(timeout: float = _PROBE_TIMEOUT) -> bool:
    """True when the user is logged in and the gateway accepts the key.

    Cached for the process lifetime like the Ollama probe; never called
    without a key, so a fresh install makes no network request here.
    """
    global _hosted_cached
    if _hosted_cached is not None:
        return _hosted_cached
    base, key = hosted_base_url(), hosted_key()
    if not base or not key:
        _hosted_cached = False
        return False
    try:
        resp = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        _hosted_cached = resp.status_code < 400
    except Exception:
        _hosted_cached = False
    return _hosted_cached


def device_login(open_browser: bool = True, poll_max_seconds: float = 600.0, echo=print) -> str:
    """Run the GitHub device flow against the gateway's auth service.

    1. ``POST {auth}/device/code`` returns ``user_code``, ``verification_uri``,
       ``device_code`` and ``interval``.
    2. The user enters the code on GitHub (the browser is opened when
       possible); the CLI polls ``POST {auth}/device/token`` until the
       service has verified the sign-in and minted a key.
    3. The key is stored 0600 in :func:`credentials_path` and returned.
    """
    auth = hosted_auth_url()
    start = httpx.post(f"{auth}/device/code", timeout=15.0)
    if start.status_code >= 400:
        raise RuntimeError(f"the sign-in service answered {start.status_code}")
    info = start.json()
    user_code, uri = info["user_code"], info["verification_uri"]
    echo(f"Sign in at {uri} and enter the code: {user_code}")
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(uri)
    interval = float(info.get("interval", 5))
    deadline = time.monotonic() + poll_max_seconds
    while time.monotonic() < deadline:
        time.sleep(interval)
        poll = httpx.post(f"{auth}/device/token", json={"device_code": info["device_code"]}, timeout=15.0)
        if poll.status_code == 200:
            body = poll.json()
            if body.get("api_key"):
                data = read_credentials()
                data.update(
                    {
                        "hosted_key": body["api_key"],
                        "hosted_user": body.get("user", ""),
                        "hosted_base_url": hosted_base_url(),
                    }
                )
                path = write_credentials(data)
                reset_probe_cache()
                echo(f"Logged in as {body.get('user', 'user')}; key stored in {path}")
                return body["api_key"]
        elif poll.status_code == 428:  # authorization_pending / slow_down
            interval = float(poll.json().get("interval", interval))
            continue
        else:
            raise RuntimeError(f"the sign-in service answered {poll.status_code}: {poll.text[:120]}")
    raise TimeoutError("the sign-in was not completed in time; run `morie login` again")


def logout(echo=print) -> bool:
    """Forget the stored key. Returns True when a key was removed."""
    data = read_credentials()
    had = bool(data.pop("hosted_key", None))
    data.pop("hosted_user", None)
    if data:
        write_credentials(data)
    else:
        with contextlib.suppress(FileNotFoundError):
            credentials_path().unlink()
    reset_probe_cache()
    echo("Hosted key removed." if had else "No hosted key was stored.")
    return had


def status() -> dict:
    """What `morie doctor` reports: endpoint, whether a key is present, whether it works."""
    key = hosted_key()
    return {
        "base_url": hosted_base_url(),
        "logged_in": bool(key),
        "user": read_credentials().get("hosted_user", ""),
        "reachable": probe_hosted() if key else False,
    }
