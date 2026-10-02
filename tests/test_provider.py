# SPDX-License-Identifier: AGPL-3.0-or-later
"""`morie provider`: attach your own OpenAI-compatible endpoint; the chain reads it."""

import json

import pytest

from morie import hosted, llm
from morie.runner import build_parser


@pytest.fixture
def creds(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(hosted, "credentials_path", lambda: path)
    for var in ("LLM_API_BASE_URL", "LLM_API_KEY", "MORIE_API_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return path


def test_provider_set_show_unset(creds, capsys):
    out = hosted.provider_set("https://api.example.org/v1/", "sk-test-1234567", "demo-model")
    assert out == {"api_base_url": "https://api.example.org/v1", "api_key": "sk-test-1234567", "api_model": "demo-model"}
    assert json.loads(creds.read_text())["api_base_url"] == "https://api.example.org/v1"
    assert llm._api_base_url() == "https://api.example.org/v1"
    assert llm._api_key() == "sk-test-1234567"
    assert llm._api_model() == "demo-model"
    assert hosted.provider_line() == "Your endpoint (https://api.example.org/v1): model demo-model"
    hosted.provider_show()
    shown = capsys.readouterr().out
    assert "never printed" in shown and "sk-test-1234567" not in shown and "sk-t" not in shown
    assert hosted.provider_unset() is True
    assert llm._api_base_url() is None and hosted.provider_line() is None
    assert hosted.provider_unset() is False


def test_env_wins_over_stored(creds, monkeypatch):
    hosted.provider_set("https://stored.example.org/v1", "stored-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://env.example.org/v1/")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("MORIE_API_MODEL", "env-model")
    assert llm._api_base_url() == "https://env.example.org/v1"
    assert llm._api_key() == "env-key"
    assert llm._api_model() == "env-model"


def test_provider_set_rejects_bad_input(creds):
    with pytest.raises(ValueError):
        hosted.provider_set("api.example.org/v1", "k")
    with pytest.raises(ValueError):
        hosted.provider_set("https://api.example.org/v1", "   ")


def test_provider_verb_parses():
    p = build_parser()
    a = p.parse_args(["provider", "set", "--base-url", "https://x/v1", "--key", "k", "--model", "m"])
    assert (a.command, a.provider_cmd, a.base_url, a.key, a.model) == ("provider", "set", "https://x/v1", "k", "m")
    assert p.parse_args(["provider", "show"]).provider_cmd == "show"
    assert p.parse_args(["provider", "unset"]).provider_cmd == "unset"


def test_positional_paths_for_profile_and_sample():
    p = build_parser()
    assert p.parse_args(["profile-dataset", "data.csv"]).path == "data.csv"
    assert p.parse_args(["sample", "data.csv", "--n", "5"]).path == "data.csv"
    assert p.parse_args(["sample", "--csv", "data.csv", "--n", "5"]).csv == "data.csv"


def test_hosted_failure_wording(monkeypatch):
    monkeypatch.setattr(hosted, "hosted_failure", lambda: "rejected")
    assert "run `morie login` again" in hosted.hosted_problem_line()
    monkeypatch.setattr(hosted, "hosted_failure", lambda: "network")
    assert "not reachable" in hosted.hosted_problem_line()
