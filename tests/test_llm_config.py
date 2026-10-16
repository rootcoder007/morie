# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saved LLM settings (`morie config`, morie.llm.config) and route selection: the model is the
user's own (an endpoint, a provider's key, or Ollama); nothing is hosted. No test touches the
network or $HOME: XDG_CONFIG_HOME is a temp dir and httpx is mocked.
"""

from __future__ import annotations

import json
import os
import stat

import httpx
import pytest

from morie import cli_help, hosted, llm, llm_config, providers

OLLAMA = "http://localhost:11434"
GROQ = providers.preset("groq").base
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
    "MORIE_HOSTED_KEY",
    *(e for p in providers.PRESETS for e in (p.key_env, p.alt_env, providers.model_env(p.id)) if e),
)


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    llm._reset_route_cache()
    hosted.reset_probe_cache()
    yield tmp_path
    llm._reset_route_cache()
    hosted.reset_probe_cache()


def _fake_get(ollama_models, seen=None):
    """httpx.get for Ollama /api/tags (None: not running)."""

    def get(url, headers=None, timeout=None, **kw):
        if seen is not None:
            seen.append((url, (headers or {}).get("Authorization")))
        req = httpx.Request("GET", url)
        if url.endswith("/api/tags"):
            if ollama_models is None:
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(200, json={"models": [{"name": m, "size": 1} for m in ollama_models]}, request=req)
        raise AssertionError(f"unexpected GET {url}")

    return get


# ---------------------------------------------------------------------------
# the automatic order: own endpoint, each provider with a key, then Ollama
# ---------------------------------------------------------------------------


def test_an_empty_ollama_is_skipped(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    assert llm._ollama_tags() == []
    assert llm._probe_ollama() is False
    assert llm.detect_available_provider() == "local"
    assert "ask has no route yet" in llm.route_summary()
    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    assert llm.detect_available_provider() == "groq"
    assert llm.route_summary().startswith("ask uses: Groq")


def test_a_provider_key_comes_before_ollama_and_own_before_both(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    assert llm.detect_available_provider() == "ollama"
    assert llm._ollama_model() == "gemma4:e2b"
    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    assert llm.detect_available_provider() == "groq"
    llm.config(own_url="http://localhost:1234/v1")
    assert llm.detect_available_provider() == "own"
    rows = {r["route"]: r for r in llm.route_status()}
    assert rows["own endpoint"]["status"] == "configured"
    assert rows["Groq"]["status"] == "key set"
    assert rows["local Ollama"]["status"] == "available"


def test_named_ollama_model_or_forced_route_uses_an_empty_server(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    assert llm.detect_available_provider(route="ollama") == "ollama"
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    llm._reset_route_cache()
    assert llm.detect_available_provider() == "ollama"
    assert llm._ollama_model() == "qwen3:8b"


def _sse(*chunks):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    return "\n\n".join([*lines, "data: [DONE]"]) + "\n\n"


def test_streamed_ask_reaches_the_provider_past_an_empty_ollama(xdg, monkeypatch):
    """`morie ask` streams by default: the end-to-end path through percy."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-123456")
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    monkeypatch.setattr(llm, "_retrieve_relevant_source", lambda q: "")
    posted = []

    def fake_stream(method, url, json=None, headers=None, timeout=None):
        posted.append((url, json["model"], headers.get("Authorization")))
        return httpx.Response(200, text=_sse("Hello", " there"), request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _Ctx(fake_stream(*a, **k)))
    monkeypatch.setenv("MORIE_GROQ_MODEL", "llama-x")
    from morie.perseus import ask_percy

    payload = ask_percy("hi", stream=True, use_agent=False)
    assert "".join(payload["output_stream"]) == "Hello there"
    assert payload["mode"] == "live_api"
    assert posted == [(f"{GROQ}/chat/completions", "llama-x", "Bearer gsk-123456")]


class _Ctx:
    """httpx.stream() stand-in: a context manager around a canned response."""

    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self.resp

    def __exit__(self, *exc):
        return False


def test_a_streamed_failure_falls_through_to_the_next_route(xdg, monkeypatch):
    """The first streamed attempt is primed, so its failure reaches the chain."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    monkeypatch.setenv("MORIE_GROQ_MODEL", "llama-x")
    monkeypatch.setattr(httpx, "get", _fake_get(["other:1b"]))
    monkeypatch.setattr(llm, "_retrieve_relevant_source", lambda q: "")
    calls = []

    def fake_stream(base_url, model, messages, *, api_key=None, timeout=0):
        calls.append(base_url)
        if base_url == GROQ:
            req = httpx.Request("POST", base_url)
            raise httpx.HTTPStatusError("401", request=req, response=httpx.Response(401, request=req))
        yield "from ollama"

    monkeypatch.setattr(llm, "_stream_completion", fake_stream)
    assert list(llm.ask("q", stream=True)) == ["from ollama"]
    assert calls == [GROQ, OLLAMA]


def test_anthropic_and_gemini_are_asked_in_their_own_format(xdg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")
    monkeypatch.setattr(httpx, "get", _fake_get(None))
    monkeypatch.setattr(llm, "_retrieve_relevant_source", lambda q: "")
    seen = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append((url, json, headers))
        req = httpx.Request("POST", url)
        if "anthropic" in url:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "from claude"}]}, request=req)
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "from gemini"}]}}]}, request=req
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm.ask("q") == "from claude"
    url, body, hdr = seen[-1]
    assert url == "https://api.anthropic.com/v1/messages"
    assert hdr["x-api-key"] == "sk-ant-1" and hdr["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-opus-5-5" and body["messages"][-1]["role"] == "user"
    assert isinstance(body.get("system"), (str, list))
    assert list(llm.ask("q", stream=True)) == ["from claude"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "g-1")
    monkeypatch.setenv("MORIE_GEMINI_MODEL", "gemini-x")
    assert llm.ask("q") == "from gemini"
    url, body, hdr = seen[-1]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert hdr == {"x-goog-api-key": "g-1"} and "systemInstruction" in body


def test_a_provider_without_a_model_says_how_to_name_one(xdg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    monkeypatch.setattr(httpx, "get", _fake_get(None))
    with pytest.raises(llm.NoModelError, match="groq.model"):
        llm._attempts_for("groq", None, "auto")
    assert llm._attempts_for("groq", "named-now", "groq")[0][2] == "named-now"


def test_a_forced_route_never_falls_back(xdg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    monkeypatch.setenv("MORIE_GROQ_MODEL", "llama-x")
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    llm_config.config(route="groq")
    assert llm.detect_available_provider() == "groq"
    assert [a[1] for a in llm._attempts_for("groq", None, "groq")] == [GROQ]
    assert [a[1] for a in llm._attempts_for("ollama", None, "ollama")] == [OLLAMA]
    # auto keeps the fallback chain, from the chosen route on
    assert [a[1] for a in llm._attempts_for("groq", None, "auto")] == [GROQ, OLLAMA]
    # a model named for the call belongs to the first route only
    att = llm._attempts_for("groq", "custom-model", "auto")
    assert att[0][2] == "custom-model" and att[1][2] == "gemma4:e2b"


def test_hosted_is_retired_and_an_unset_provider_is_local(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(["gemma4:e2b"]))
    with pytest.raises(ValueError, match="retired"):
        llm.detect_available_provider(route="hosted")
    with pytest.raises(ValueError, match="retired"):
        llm.config(route="hosted")
    assert llm.detect_available_provider(route="anthropic") == "local"  # no key: nothing to ask
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
        **{"anthropic.model": "claude-x"},
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
    assert providers.model("anthropic") == "claude-x"
    llm._ollama_tags()
    assert seen == [("http://192.168.1.20:11434/api/tags", "Bearer ok-12345678")]
    assert llm.config_get("ollama.key") == "set"  # no character of a key is ever shown
    llm.config_unset("route", "ollama.url", "ollama.model", "ollama.key", "anthropic.model")
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
    monkeypatch.setenv("MORIE_LLM_ROUTE", "mistral")
    assert llm_config.route() == "mistral"
    monkeypatch.setenv("MORIE_LLM_ROUTE", "hosted")
    assert llm_config.route() == "auto"  # a retired or unknown route in the environment is not followed


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
    with pytest.raises(KeyError, match="unknown setting"):
        llm.config(hosted_url="https://llm.example.org")
    with pytest.raises(KeyError, match="unknown setting"):
        llm.config(colour="blue")
    assert not llm.config_path().exists()


def test_own_endpoint_settings(xdg, monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(None))
    llm.config(own_url="http://localhost:1234/v1", own_model="my-model")
    assert llm.detect_available_provider() == "own"  # a local server needs no key
    assert llm._api_model() == "my-model"
    attempts = llm._provider_attempts("own", None)
    assert attempts[0] == ("openai", "http://localhost:1234/v1", "my-model", None)
    assert llm._chat_url("http://localhost:1234/v1") == "http://localhost:1234/v1/chat/completions"
    # the endpoint attached with `morie provider set` still counts, behind a saved own.url
    hosted.write_credentials({"api_base_url": "https://attached.example/v1", "api_key": "k"})
    assert llm._api_base_url() == "http://localhost:1234/v1"
    llm.config_unset("own.url")
    assert llm._api_base_url() == "https://attached.example/v1"


def test_chat_url():
    assert llm._chat_url("http://localhost:11434") == "http://localhost:11434/v1/chat/completions"
    assert llm._chat_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    # a provider's documented base, as it is: Perplexity's has no version
    assert llm._chat_url("https://api.perplexity.ai") == "https://api.perplexity.ai/chat/completions"
    assert llm._chat_url("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1/chat/completions"


def test_a_provider_key_is_saved_privately_and_never_shown(xdg, monkeypatch):
    llm.config(anthropic_key="sk-ant-secret-123")
    assert providers.key("anthropic") == "sk-ant-secret-123"
    assert llm.config_get("anthropic.key") == "set"
    assert "sk-ant-secret" not in str(llm.config())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert providers.key("anthropic") == "sk-env"  # the environment wins
    assert llm_config.source("anthropic.key") == "environment"


# ---------------------------------------------------------------------------
# CLI: morie config ..., morie help ..., morie ask --route
# ---------------------------------------------------------------------------


def test_config_cli(xdg, monkeypatch):
    out = []
    assert llm_config.cli("set", ["route", "anthropic"], out=out.append) == 0
    assert out[-1] == "route = anthropic"
    assert llm_config.cli("get", ["route"], out=out.append) == 0 and out[-1] == "anthropic"
    assert llm_config.cli("set", ["route", "hosted"], out=out.append) == 1 and "retired" in out[-1]
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
    monkeypatch.setattr(httpx, "get", _fake_get(None))
    monkeypatch.setattr(providers, "list_models", lambda pid, timeout=10.0: ["llama-a", "llama-b"])
    answers = iter(["groq", "llama-b"])
    out = []
    assert llm_config.setup(out=out.append, ask=lambda p: next(answers), secret_ask=lambda p: "gsk-wizard-1") == 0
    data = json.loads(llm.config_path().read_text())
    assert data == {"route": "groq", "groq.key": "gsk-wizard-1", "groq.model": "llama-b"}
    assert any("Models: llama-a, llama-b" in line for line in out)
    assert any("ask uses: Groq, model llama-b" in line for line in out)
    assert not any("gsk-wizard-1" in line for line in out)


def test_runner_parses_config_help_and_route():
    from morie.runner import build_parser

    p = build_parser()
    a = p.parse_args(["config", "set", "ollama.url", "http://box:11434"])
    assert (a.command, a.action, a.rest) == ("config", "set", ["ollama.url", "http://box:11434"])
    assert p.parse_args(["config"]).action == "show"
    assert p.parse_args(["ask", "--route", "groq", "hi"]).route == "groq"
    with pytest.raises(SystemExit):
        p.parse_args(["ask", "--route", "hosted", "hi"])
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
    monkeypatch.setattr("sys.argv", ["morie", "ask", "--no-stream", "--route", "anthropic", "hi"])
    assert runner._main_impl() == 0
    assert seen["route"] == "anthropic"
    assert perseus.ask_percy.__kwdefaults__["route"] is None


def test_help_topics():
    for t in ("getting-started", "llm", "config"):
        assert cli_help.show(t)
    assert "getting-started" in cli_help.show(None)
    assert cli_help.show("ask") == cli_help.show("llm")
    assert cli_help.show("nope") is None


def test_doctor_reports_the_route(xdg, monkeypatch):
    from morie import doctor

    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    monkeypatch.setenv("MORIE_GROQ_MODEL", "llama-x")
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    ok, detail = doctor._check_ollama()
    assert not ok and "no model pulled" in detail
    ok, detail = doctor._check_route()
    assert ok and detail.startswith("ask uses: Groq, model llama-x")
    ok, detail = doctor._check_providers()
    assert ok and "groq (model llama-x)" in detail
    ok, detail = doctor._check_data_key()
    assert not ok and "morie login" in detail
