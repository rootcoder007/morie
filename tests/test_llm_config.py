# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saved LLM settings (`morie config`, morie.llm.config) and route selection.

The bug these guard against: a user logged in to the hosted tier, with an Ollama
server running but nothing pulled, got "no model" from `morie ask` because the
empty Ollama counted as a route and the hosted tier was never reached. No test
touches the network or $HOME: XDG_CONFIG_HOME is a temp dir and httpx is mocked.
"""

from __future__ import annotations

import json
import os
import stat

import httpx
import pytest

from morie import cli_help, hosted, llm, llm_config, services

HOSTED = "https://llm.example.org"
OLLAMA = "http://localhost:11434"
_ENV = (
    "MORIE_LLM_ROUTE",
    "MORIE_LLM_BASE_URL",
    "MORIE_LLM_API_KEY",
    "MORIE_LLM_MODEL",
    "LLM_API_BASE_URL",
    "LLM_API_KEY",
    "MORIE_API_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "MORIE_OLLAMA_MODEL",
    "OLLAMA_API_KEY",
    "MORIE_HOSTED_BASE_URL",
    "MORIE_HOSTED_MODEL",
    "MORIE_HOSTED_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "MORIE_OPENAI_MODEL",
)


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    # the services document would be read from a cache or the bundled copy; pin it
    monkeypatch.setattr(
        services, "llm", lambda: {"mode": "key", "base_url": HOSTED, "default_model": "minimax-m3:cloud"}
    )
    llm._reset_route_cache()
    hosted.reset_probe_cache()
    yield tmp_path
    llm._reset_route_cache()
    hosted.reset_probe_cache()


def _fake_get(ollama_models, hosted_models=("minimax-m3:cloud", "gpt-oss-120b:cf"), seen=None):
    """httpx.get for Ollama /api/tags (None: not running) and the hosted /v1/models."""

    def get(url, headers=None, timeout=None, **kw):
        if seen is not None:
            seen.append((url, (headers or {}).get("Authorization")))
        req = httpx.Request("GET", url)
        if url.endswith("/api/tags"):
            if ollama_models is None:
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(200, json={"models": [{"name": m, "size": 1} for m in ollama_models]}, request=req)
        if url.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in hosted_models]}, request=req)
        raise AssertionError(f"unexpected GET {url}")

    return get


def _login(xdg):
    hosted.write_credentials({"hosted_key": "sk-hosted-123456"})


# ---------------------------------------------------------------------------
# the reported bug: an empty Ollama must not hide the hosted tier
# ---------------------------------------------------------------------------


def test_empty_ollama_is_skipped_for_the_hosted_tier(xdg, monkeypatch):
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    assert llm._ollama_tags() == []
    assert llm._probe_ollama() is False
    assert llm.detect_available_provider() == "hosted"
    assert "hosted MORIE tier" in llm.route_summary()


def test_ollama_with_a_model_still_comes_first(xdg, monkeypatch):
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    assert llm.detect_available_provider() == "ollama"
    assert llm._ollama_model() == "gemma4:e2b"


def test_named_ollama_model_or_forced_route_uses_an_empty_server(xdg, monkeypatch):
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    assert llm.detect_available_provider(route="ollama") == "ollama"
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    llm._reset_route_cache()
    assert llm.detect_available_provider() == "ollama"
    assert llm._ollama_model() == "qwen3:8b"


def _sse(*chunks):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    return "\n\n".join([*lines, "data: [DONE]"]) + "\n\n"


def test_streamed_ask_reaches_the_hosted_tier_past_an_empty_ollama(xdg, monkeypatch):
    """`morie ask` streams by default: the end-to-end path the user hit."""
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    monkeypatch.setattr(llm, "_retrieve_relevant_source", lambda q: "")
    posted = []

    def fake_stream(method, url, json=None, headers=None, timeout=None):
        posted.append((url, json["model"], headers.get("Authorization")))
        req = httpx.Request(method, url)
        if not url.startswith(HOSTED):
            return httpx.Response(404, json={"error": "model not found"}, request=req)
        return httpx.Response(200, text=_sse("Hello", " there"), request=req)

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _Ctx(fake_stream(*a, **k)))
    from morie.perseus import ask_percy

    payload = ask_percy("hi", stream=True, use_agent=False)
    assert "".join(payload["output_stream"]) == "Hello there"
    assert payload["mode"] == "live_api"
    assert posted == [(f"{HOSTED}/v1/chat/completions", "minimax-m3:cloud", "Bearer sk-hosted-123456")]


class _Ctx:
    """httpx.stream() stand-in: a context manager around a canned response."""

    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self.resp

    def __exit__(self, *exc):
        return False


def test_a_streamed_failure_falls_through_to_the_next_provider(xdg, monkeypatch):
    """Before: the first streamed attempt was returned unstarted, so its failure never reached the chain."""
    _login(xdg)
    monkeypatch.setenv("OLLAMA_MODEL", "gone:latest")
    monkeypatch.setattr(httpx, "get", _fake_get(["other:1b"]))
    monkeypatch.setattr(llm, "_retrieve_relevant_source", lambda q: "")
    calls = []

    def fake_stream(base_url, model, messages, *, api_key=None, timeout=0):
        calls.append(base_url)
        if base_url == OLLAMA:
            req = httpx.Request("POST", base_url)
            raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))
        yield "from hosted"

    monkeypatch.setattr(llm, "_stream_completion", fake_stream)
    out = llm.ask("q", stream=True)
    assert list(out) == ["from hosted"]
    assert calls == [OLLAMA, HOSTED]


def test_a_forced_route_never_falls_back(xdg, monkeypatch):
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    llm_config.config(route="hosted")
    assert llm.detect_available_provider() == "hosted"
    assert llm._attempts_for("hosted", None, "hosted") == [(HOSTED, "minimax-m3:cloud", "sk-hosted-123456")]
    # route = ollama keeps to Ollama: the hosted tier is not tried behind it
    attempts = llm._attempts_for("ollama", None, "ollama")
    assert [a[0] for a in attempts] == [OLLAMA]
    # auto keeps the fallback chain
    assert [a[0] for a in llm._attempts_for("ollama", None, "auto")] == [OLLAMA, HOSTED]


def test_forced_hosted_without_a_key_is_local(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    assert llm.detect_available_provider(route="hosted") == "local"
    with pytest.raises(ValueError):
        llm.detect_available_provider(route="cloud")


# ---------------------------------------------------------------------------
# settings: file, precedence, validation
# ---------------------------------------------------------------------------


def test_reading_the_settings_writes_nothing(xdg):
    tab = llm.config()
    assert [r["key"] for r in tab] == list(llm_config.KEYS)
    assert {r["source"] for r in tab} == {"default"}
    assert not (xdg / "morie" / "llm.json").exists()
    assert "route" in str(tab)


def test_saved_settings_are_private_and_apply(xdg, monkeypatch):
    seen = []
    monkeypatch.setattr(httpx, "get", _fake_get([], seen=seen))
    tab = llm.config(
        route="ollama",
        ollama_url="192.168.1.20:11434/",
        ollama_model="qwen3:8b",
        ollama_key="ok-12345678",
        **{"hosted.model": "gpt-oss-120b:cf"},
    )
    path = xdg / "morie" / "llm.json"
    assert path == llm.config_path()
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(path).st_mode) == stat.S_IRUSR | stat.S_IWUSR
    data = json.loads(path.read_text())
    assert data["route"] == "ollama" and data["ollama.url"] == "192.168.1.20:11434"
    rows = {r["key"]: r for r in tab}
    assert rows["route"]["source"] == "saved" and rows["ollama.key"]["value"] == "set"
    assert llm._ollama_base_url() == "http://192.168.1.20:11434"
    assert llm._ollama_model() == "qwen3:8b"
    assert hosted.hosted_model() == "gpt-oss-120b:cf"
    llm._ollama_tags()
    assert seen == [("http://192.168.1.20:11434/api/tags", "Bearer ok-12345678")]
    assert llm.config_get("ollama.key") == "set"  # no character of a key is ever shown
    llm.config_unset("route", "ollama.url", "ollama.model", "ollama.key", "hosted.model")
    assert not path.exists()
    assert llm.config_get("route") == "auto"


def test_an_environment_variable_wins_and_warns(xdg, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    with pytest.warns(UserWarning, match="OLLAMA_MODEL"):
        llm.config(ollama_model="saved-model")
    assert llm._ollama_model() == "env-model"
    assert llm_config.source("ollama.model") == "environment"
    monkeypatch.delenv("OLLAMA_MODEL")
    assert llm._ollama_model() == "saved-model"
    # the older names morie read are still accepted
    monkeypatch.setenv("MORIE_OLLAMA_MODEL", "old-name")
    assert llm._ollama_model() == "old-name"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434/")
    assert llm._ollama_base_url() == "http://box:11434"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert llm._ollama_base_url() == "http://0.0.0.0:11434"
    monkeypatch.setenv("MORIE_LLM_ROUTE", "hosted")
    assert llm_config.route() == "hosted"


def test_ollama_off_skips_it(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(a))
    llm.config(ollama_url="off")
    assert llm._ollama_base_url() is None
    assert llm.detect_available_provider() == "local"
    assert calls == []


def test_bad_values_are_refused(xdg):
    with pytest.raises(ValueError, match="route"):
        llm.config(route="cloud")
    with pytest.raises(ValueError, match="own.url"):
        llm.config(own_url="localhost:1234")
    with pytest.raises(ValueError, match="https"):
        llm.config(hosted_url="http://llm.example.org")
    with pytest.raises(KeyError, match="unknown setting"):
        llm.config(colour="blue")
    assert not llm.config_path().exists()


def test_own_endpoint_settings(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(None))
    llm.config(own_url="http://localhost:1234/v1", own_model="my-model")
    assert llm.detect_available_provider() == "api"  # a local server needs no key
    assert llm._api_model() == "my-model"
    attempts = llm._provider_attempts("api", None)
    assert attempts[0] == ("http://localhost:1234/v1", "my-model", None)
    assert llm._chat_url("http://localhost:1234/v1") == "http://localhost:1234/v1/chat/completions"
    # the endpoint attached with `morie provider set` still counts, behind a saved own.url
    hosted.write_credentials({"api_base_url": "https://attached.example/v1", "api_key": "k"})
    assert llm._api_base_url() == "http://localhost:1234/v1"
    llm.config_unset("own.url")
    assert llm._api_base_url() == "https://attached.example/v1"


def test_chat_url():
    assert llm._chat_url("http://localhost:11434") == "http://localhost:11434/v1/chat/completions"
    assert llm._chat_url("https://llm.rmorie.com/") == "https://llm.rmorie.com/v1/chat/completions"
    assert llm._chat_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/chat/completions"
    assert llm._chat_url(llm.GEMINI_BASE_URL) == llm.GEMINI_BASE_URL + "/chat/completions"


def test_hosted_key_goes_through_the_login_path(xdg, monkeypatch):
    stored = []
    monkeypatch.setattr(hosted, "store_token", lambda tok, echo=None: stored.append(tok) or tok)
    llm.config(hosted_key="sk-new-key-1234")
    assert stored == ["sk-new-key-1234"]
    assert not llm.config_path().exists()  # the key never lands in llm.json
    hosted.write_credentials({"hosted_key": "sk-new-key-1234"})
    assert llm_config.source("hosted.key") == "saved"
    llm.config(hosted_key=None)
    assert hosted.hosted_key() is None


def test_saved_hosted_url(xdg, monkeypatch):
    llm.config(hosted_url="https://gw.example.org/")
    assert hosted.hosted_base_url() == "https://gw.example.org"
    llm.config(hosted_url="off")
    assert hosted.hosted_base_url() is None
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "https://env.example.org")
    assert hosted.hosted_base_url() == "https://env.example.org"


# ---------------------------------------------------------------------------
# CLI: morie config ..., morie help ..., morie ask --route
# ---------------------------------------------------------------------------


def test_config_cli(xdg, monkeypatch):
    out = []
    assert llm_config.cli("set", ["route", "hosted"], out=out.append) == 0
    assert out[-1] == "route = hosted"
    assert llm_config.cli("get", ["route"], out=out.append) == 0 and out[-1] == "hosted"
    assert llm_config.cli("set", ["route", "nope"], out=out.append) == 1
    assert "route must be one of" in out[-1]
    assert llm_config.cli("set", ["ollama_model"], out=out.append) == 2  # a non-secret needs a value
    assert llm_config.cli("set", ["own.key"], out=out.append, secret_ask=lambda p: "sk-own-12345678") == 0
    assert json.loads(llm.config_path().read_text())["own.key"] == "sk-own-12345678"
    assert "sk-own-12345678" not in "\n".join(out)
    assert llm_config.cli("unset", ["route"], out=out.append) == 0 and out[-1] == "route = auto"
    assert llm_config.cli("get", ["colour"], out=out.append) == 2
    assert llm_config.cli("path", [], out=out.append) == 0 and out[-1] == str(llm.config_path())
    assert llm_config.cli("help", [], out=out.append) == 0 and "MORIE_LLM_ROUTE" in out[-1]
    assert llm_config.cli("show", [], out=out.append) == 0 and "own.key" in out[-2]


def test_config_setup_wizard(xdg, monkeypatch):
    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    answers = iter(["hosted", "gpt-oss-120b:cf"])
    out = []
    assert llm_config.setup(out=out.append, ask=lambda p: next(answers), secret_ask=lambda p: "") == 0
    data = json.loads(llm.config_path().read_text())
    assert data == {"route": "hosted", "hosted.model": "gpt-oss-120b:cf"}
    assert any("ask uses: hosted MORIE tier, model gpt-oss-120b:cf" in line for line in out)


def test_runner_parses_config_help_and_route():
    from morie.runner import build_parser

    p = build_parser()
    a = p.parse_args(["config", "set", "ollama.url", "http://box:11434"])
    assert (a.command, a.action, a.rest) == ("config", "set", ["ollama.url", "http://box:11434"])
    assert p.parse_args(["config"]).action == "show"
    assert p.parse_args(["ask", "--route", "hosted", "hi"]).route == "hosted"
    assert p.parse_args(["help", "llm"]).topic == "llm"
    with pytest.raises(SystemExit):
        p.parse_args(["ask", "--route", "cloud", "hi"])


def test_runner_config_and_help_commands(xdg, monkeypatch, capsys):
    from morie import runner

    monkeypatch.setattr("sys.argv", ["morie", "config", "set", "route", "own"])
    assert runner._main_impl() == 0
    assert "route = own" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["morie", "help", "config"])
    assert runner._main_impl() == 0
    assert "morie config setup" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["morie", "help", "nope"])
    assert runner._main_impl() == 2


def test_ask_route_flag_reaches_the_provider_choice(xdg, monkeypatch, capsys):
    from morie import perseus, runner

    seen = {}

    def fake_ask_percy(question, **kw):
        seen.update(kw)
        return {"mode": "live_api", "model": "m", "output_text": "ok"}

    monkeypatch.setattr(runner, "ask_percy", fake_ask_percy)
    monkeypatch.setattr("sys.argv", ["morie", "ask", "--no-stream", "--route", "hosted", "hi"])
    assert runner._main_impl() == 0
    assert seen["route"] == "hosted"
    assert perseus.ask_percy.__kwdefaults__["route"] is None


def test_help_topics():
    for t in ("getting-started", "llm", "config"):
        assert cli_help.show(t)
    assert "getting-started" in cli_help.show(None)
    assert cli_help.show("ask") == cli_help.show("llm")
    assert cli_help.show("nope") is None


def test_doctor_reports_the_route(xdg, monkeypatch):
    from morie import doctor

    _login(xdg)
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    ok, detail = doctor._check_ollama()
    assert not ok and "no model pulled" in detail
    ok, detail = doctor._check_route()
    assert ok and detail.startswith("ask uses: hosted MORIE tier, model minimax-m3:cloud")
