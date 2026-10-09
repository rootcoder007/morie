"""The hosted MORIE inference tier at llm.rmorie.com.

An authenticated, rate-limited OpenAI-compatible endpoint run by the
project (LiteLLM in front of an Ollama server). It is the last resort in
the provider chain, after a local Ollama and after every cloud API key of
the user's own, and it only ever speaks when the user has a key: issued on
request at https://rmorie.com/access and stored with ``morie login --token``,
or minted by ``morie login`` (GitHub device flow) or ``morie login --email``,
with mode 0600 under the XDG config directory. Its address comes from the
signed services document (``morie.services``), not from a constant here. Nothing is sent anywhere without
that key, and no anonymous endpoint is contacted.

Environment
-----------
MORIE_HOSTED_BASE_URL
    Override the endpoint (``https://llm.rmorie.com``; the request layer adds
    ``/v1/chat/completions``). An empty string
    disables the tier.
MORIE_HOSTED_KEY
    Override the stored key (CI, containers).
MORIE_HOSTED_MODEL
    Model name; the gateway only serves the cloud models of the server.

The address and model can also be saved with ``morie config set hosted.url|hosted.model``
(``$XDG_CONFIG_HOME/morie/llm.json``); a variable that is set wins over the saved value.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

DEFAULT_HOSTED_BASE_URL = "https://llm.rmorie.com"
DEFAULT_HOSTED_AUTH_URL = "https://llm.rmorie.com/auth"
DEFAULT_HOSTED_MODEL = "minimax-m3:cloud"
_PROBE_TIMEOUT = 2.0


def hosted_base_url() -> str | None:
    """The hosted endpoint, or None when disabled.

    ``MORIE_HOSTED_BASE_URL`` overrides ("" or "off" disables); otherwise the signed
    services document decides (``morie.services``), and a document with the tier
    switched off disables it here too. The tier is a last resort behind a local model
    or your own API key; keys are issued on request at https://rmorie.com/access.
    """
    if "MORIE_HOSTED_BASE_URL" in os.environ:
        url = os.environ["MORIE_HOSTED_BASE_URL"].strip()
    else:
        from .llm_config import saved_value

        url = saved_value("hosted.url")  # `morie config set hosted.url ...`
    if url is not None:
        if url.lower() in ("off", "none", "disabled"):
            return None
        return url.rstrip("/") or None
    from . import services

    llm = services.llm()
    if llm.get("mode") != "key" or not llm.get("base_url"):
        return None
    return str(llm["base_url"]).rstrip("/")


def hosted_auth_url() -> str:
    """Base URL of the hosted-tier login service (``MORIE_HOSTED_AUTH_URL``, else the services document), without a trailing slash."""
    env = os.environ.get("MORIE_HOSTED_AUTH_URL", "").strip()
    if env:
        return env.rstrip("/")
    from . import services

    return (str(services.llm().get("auth_url") or "") or DEFAULT_HOSTED_AUTH_URL).rstrip("/")


def hosted_model() -> str:
    """Default model on the hosted tier (``MORIE_HOSTED_MODEL``, else the services document, else the built-in default)."""
    from .llm_config import value

    chosen = value("hosted.model")  # MORIE_HOSTED_MODEL, else `morie config set hosted.model ...`
    if chosen:
        return chosen
    from . import services

    return str(services.llm().get("default_model") or "") or DEFAULT_HOSTED_MODEL


def access_hint() -> str:
    """One line on how to get a hosted key."""
    from . import services

    return services.access_hint()


def credentials_path() -> Path:
    """``$XDG_CONFIG_HOME/morie/credentials.json`` (shared with the R package)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "morie" / "credentials.json"


def read_credentials() -> dict:
    """The stored credentials as a dict; ``{}`` when the file is missing, unreadable or not a JSON object."""
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
_hosted_models: list[str] | None = None  # what the gateway listed for this key, once probed
_hosted_failure: str | None = None


def reset_probe_cache() -> None:
    """Forget the cached hosted-tier probe (reachability, model list, last failure) so the next call probes again."""
    global _hosted_cached, _hosted_models, _hosted_failure
    _hosted_cached = None
    _hosted_models = None
    _hosted_failure = None


def probe_hosted(timeout: float = _PROBE_TIMEOUT) -> bool:
    """True when the user is logged in and the gateway accepts the key.

    Cached for the process lifetime like the Ollama probe; never called
    without a key, so a fresh install makes no network request here.
    """
    global _hosted_cached, _hosted_models, _hosted_failure
    if _hosted_cached is not None:
        return _hosted_cached
    base, key = hosted_base_url(), hosted_key()
    if not base or not key:
        _hosted_cached = False
        return False
    try:
        resp = httpx.get(f"{base}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        _hosted_cached = resp.status_code < 400
        if _hosted_cached:
            _hosted_failure = None
            with contextlib.suppress(Exception):
                ids = [m.get("id") for m in resp.json().get("data", [])]
                _hosted_models = [m for m in ids if isinstance(m, str) and m] or None
        else:
            _hosted_failure = "rejected" if resp.status_code in (401, 403) else f"http {resp.status_code}"
    except Exception:
        _hosted_cached = False
        _hosted_failure = "network"
    return _hosted_cached


def hosted_failure() -> str | None:
    """Why the last probe failed: "rejected" (401/403: the key is no longer
    valid at the gateway), "network", "http NNN", or None when it succeeded."""
    probe_hosted()
    return _hosted_failure


def hosted_problem_line(cli: str = "morie") -> str:
    """One sentence for a logged-in user whose probe failed, with the remedy."""
    why = hosted_failure()
    if why == "rejected":
        return (
            f"logged in, but the gateway rejected the key (it was reset or replaced "
            f"by a newer sign-in): run `{cli} login` again"
        )
    if why in (None, "network"):
        return "logged in, gateway not reachable from this machine (network)"
    return f"logged in, gateway answered {why}: try again, or `{cli} login`"


def hosted_models() -> list[str] | None:
    """The models the gateway lists for this key (None until a successful probe)."""
    probe_hosted()
    return _hosted_models


def hosted_model_available() -> str:
    """The configured model, or the gateway's first model when the configured one is not offered.

    Cloud models get retired upstream; a stale default must not turn every
    hosted request into a 400. Nothing is fetched beyond the probe already made.
    """
    wanted = hosted_model()
    listed = hosted_models()
    if listed and wanted not in listed:
        return listed[0]
    return wanted


def _say(msg: str) -> None:
    print(msg, flush=True)  # the code must reach a redirected stdout before polling starts


def _can_open_browser() -> bool:
    """False over SSH and on a Linux/BSD machine without a desktop session."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd")):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _open_browser(uri: str) -> None:
    """Open ``uri`` without ever waiting on the browser.

    With ``$BROWSER`` set, :func:`webbrowser.open` runs that command and waits
    for it; a browser that stays open (Brave, Firefox) then blocked the sign-in
    before its first poll. Each ``$BROWSER`` entry is started detached instead.
    Without a desktop session nothing is opened (a console browser such as
    lynx would take over the terminal): the printed URL works on any device.
    """
    if not _can_open_browser():
        return
    for entry in filter(None, os.environ.get("BROWSER", "").split(os.pathsep)):
        try:
            cmd = shlex.split(entry)
        except ValueError:
            continue
        cmd = [c.replace("%s", uri) for c in cmd] if any("%s" in c for c in cmd) else [*cmd, uri]
        try:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        except OSError:
            continue
    with contextlib.suppress(Exception):
        browser = webbrowser.get()
        # a plain GenericBrowser is a console browser webbrowser waits on; the GUI launchers return at once
        if type(browser) is not webbrowser.GenericBrowser:
            browser.open(uri)


def _browsable(uri: object) -> bool:
    """Only an https address of a public host is handed to the browser: the sign-in service
    names the page, and bricklayer's fourth review found that name taken on trust."""
    import ipaddress
    from urllib.parse import urlsplit

    if not isinstance(uri, str) or not uri.startswith("https://"):
        return False
    host = (urlsplit(uri).hostname or "").lower()
    if not host or "." not in host or host.endswith((".local", ".internal", ".localhost", ".lan", ".home", ".corp")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True


def device_login(open_browser: bool = True, poll_max_seconds: float = 600.0, echo=_say) -> str:
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
    echo(f"Sign in at {uri} and enter the code: {user_code}  (no GitHub? run: morie login --email you@example.com)")
    if open_browser and _browsable(uri):
        _open_browser(uri)
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


def store_token(token: str, echo=_say) -> str:
    """Store a key obtained elsewhere (the landing page, or one emailed by the gateway).

    The key is trimmed, written to the shared credentials file and probed once;
    a key the gateway rejects is still stored but the user is told.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("an empty token cannot be stored")
    data = read_credentials()
    previous = dict(data)
    data.update({"hosted_key": token, "hosted_base_url": hosted_base_url()})
    path = write_credentials(data)
    reset_probe_cache()
    if probe_hosted():
        echo(f"Token stored in {path}; the gateway accepts it.")
        return token
    why = hosted_failure()
    write_credentials(previous)  # nothing is kept that the gateway refused
    reset_probe_cache()
    if why == "rejected":
        raise ValueError("the gateway rejected that key; nothing stored (check it, or run `morie login` again)")
    raise ValueError(f"the gateway could not be reached to check that key ({why}); nothing stored, try again")


def email_login(email: str, code: str | None = None, ask=input, echo=_say, to_email: bool = False) -> str:
    """Sign in with an emailed one-time code instead of GitHub.

    ``POST {auth}/email/code`` sends a 6-digit code (10 minutes, single use);
    the user types it (or passes ``code``), ``POST {auth}/email/verify``
    returns the key, stored like the device-flow one. With ``to_email`` the
    gateway emails the key instead of returning it; nothing is stored and the
    user pastes it later with ``morie login --token``.
    """
    auth = hosted_auth_url()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("an email address is required")
    if code is None:
        r = httpx.post(f"{auth}/email/code", json={"email": email}, timeout=20.0)
        if r.status_code != 200:
            raise RuntimeError(r.json().get("error", f"the sign-in service answered {r.status_code}"))
        echo(f"A 6-digit code was sent to {email} (valid for 10 minutes).")
        code = ask("Enter the code: ").strip()
    payload = {"email": email, "code": code}
    if to_email:
        payload["deliver"] = "email"
    r = httpx.post(f"{auth}/email/verify", json=payload, timeout=20.0)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("error", f"the sign-in service answered {r.status_code}"))
    body = r.json()
    if to_email:
        if not body.get("sent"):
            raise RuntimeError("the sign-in service did not confirm the email")
        echo(f"Your key was emailed to {email}. Store it with: morie login --token")
        return ""
    data = read_credentials()
    data.update(
        {"hosted_key": body["api_key"], "hosted_user": body.get("user", ""), "hosted_base_url": hosted_base_url()}
    )
    path = write_credentials(data)
    reset_probe_cache()
    echo(f"Logged in; key stored in {path}")
    return body["api_key"]


def logout(echo=_say) -> bool:
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


def models_lines() -> list[str]:
    """What `morie models` prints for the hosted tier: every model this key may ask, default marked."""
    s = status()
    if not s["base_url"]:
        return ["Hosted tier: disabled (MORIE_HOSTED_BASE_URL is empty)"]
    if not s["logged_in"]:
        return [
            f"Hosted tier ({DEFAULT_HOSTED_BASE_URL}): not logged in -- run `morie login` (GitHub) or `morie login --email you@example.com`"
        ]
    if not s["reachable"]:
        return [f"Hosted tier ({s['base_url']}): {hosted_problem_line()}"]
    default = hosted_model_available()
    who = f", logged in as {s['user']}" if s.get("user") else ""
    lines = [f"Hosted tier ({s['base_url']}){who}; default marked *:"]
    lines += [f"  {'*' if m == default else ' '} {m}" for m in hosted_models() or []]
    return lines


def status() -> dict:
    """What `morie doctor` reports: endpoint, whether a key is present, whether it works."""
    key = hosted_key()
    return {
        "base_url": hosted_base_url(),
        "logged_in": bool(key),
        "user": read_credentials().get("hosted_user", ""),
        "reachable": probe_hosted() if key else False,
        "failure": hosted_failure() if key else None,
    }


# ---------------------------------------------------------------------------
# Your own endpoint: `morie provider set|show|unset`
# ---------------------------------------------------------------------------

_PROVIDER_KEYS = ("api_base_url", "api_key", "api_model")


def provider_set(base_url: str, key: str, model: str | None = None, echo=_say) -> dict:
    """Attach an OpenAI-compatible endpoint (chat completions at BASE_URL/chat/completions).

    Works for OpenAI, Anthropic's compatibility endpoint (https://api.anthropic.com/v1),
    OpenRouter, Mistral, Groq, a local LM Studio / vLLM / llama.cpp server, or any
    other server that speaks that API. Environment variables LLM_API_BASE_URL,
    LLM_API_KEY and MORIE_API_MODEL still take precedence when set.
    """
    base_url = base_url.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    if not key.strip():
        raise ValueError("the key is empty")
    data = read_credentials()
    data["api_base_url"] = base_url
    data["api_key"] = key.strip()
    if model and model.strip():
        data["api_model"] = model.strip()
    else:
        data.pop("api_model", None)
    path = write_credentials(data)
    echo(
        f"Endpoint attached: {base_url}"
        + (f" (model {data.get('api_model')})" if data.get("api_model") else "")
        + f"; stored in {path}"
    )
    return {k: data.get(k) for k in _PROVIDER_KEYS}


def provider_show(echo=_say) -> dict:
    """Describe the attached OpenAI-compatible endpoint without printing its key.

    Args:
        echo: where the description goes (default: stdout).

    Returns:
        The stored ``api_base_url``, ``api_model`` and ``api_key`` entries (``None`` where unset).
    """
    data = read_credentials()
    base, model = data.get("api_base_url"), data.get("api_model")
    if not base:
        echo("No endpoint attached. Attach one with: morie provider set --base-url URL --key KEY [--model NAME]")
    else:
        key = str(data.get("api_key", ""))
        echo(
            f"Endpoint: {base}\nModel:    {model or 'server default'}\nKey:      stored ({len(key)} characters; never printed)"
        )
    return {k: data.get(k) for k in _PROVIDER_KEYS}


def provider_unset(echo=_say) -> bool:
    """Detach the OpenAI-compatible endpoint from the stored credentials.

    Other stored credentials are kept; the file is removed only when
    nothing else is left in it, and nothing is written when no endpoint
    was attached.

    Args:
        echo: where the outcome line goes (default: stdout).

    Returns:
        True if an endpoint was attached.
    """
    data = read_credentials()
    had = any(k in data for k in _PROVIDER_KEYS)
    for k in _PROVIDER_KEYS:
        data.pop(k, None)
    if data:
        write_credentials(data)
    else:
        with contextlib.suppress(FileNotFoundError):
            credentials_path().unlink()
    echo("Endpoint detached." if had else "No endpoint was attached.")
    return had


def provider_line() -> str | None:
    """One line for `morie models` when an endpoint is attached or set in the environment."""
    from .llm import _api_base_url, _api_key, _api_model

    base = _api_base_url()
    if not base or not _api_key():
        return None
    return f"Your endpoint ({base}): model {_api_model()}"
