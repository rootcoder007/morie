"""The hosted tier: credentials file, probe, provider order, device-flow login."""

import json
import os
import stat

import httpx
import pytest

from morie import hosted, llm


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MORIE_HOSTED_KEY", raising=False)
    monkeypatch.delenv("MORIE_HOSTED_BASE_URL", raising=False)
    hosted.reset_probe_cache()
    llm._ollama_cached = False
    yield tmp_path
    hosted.reset_probe_cache()
    llm._ollama_cached = None


def test_credentials_are_written_owner_only(isolated_home):
    path = hosted.write_credentials({"hosted_key": "sk-test"})
    assert path == isolated_home / "morie" / "credentials.json"
    if os.name != "nt":  # NTFS has no POSIX mode bits; the file is still user-private there
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR
    assert hosted.hosted_key() == "sk-test"
    assert hosted.logout(echo=lambda *_: None) is True
    assert hosted.hosted_key() is None and not path.exists()


def test_env_key_overrides_file(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "file-key"})
    monkeypatch.setenv("MORIE_HOSTED_KEY", "env-key")
    assert hosted.hosted_key() == "env-key"


def test_probe_never_runs_without_a_key(isolated_home, monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(a) or None)
    assert hosted.probe_hosted() is False
    assert calls == []


def test_probe_uses_bearer_key_and_caches(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "sk-abc"})
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append((url, headers["Authorization"]))
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert hosted.probe_hosted() is True
    assert hosted.probe_hosted() is True
    assert seen == [(hosted.DEFAULT_HOSTED_BASE_URL + "/v1/models", "Bearer sk-abc")]


def test_provider_order_puts_hosted_after_local_ollama(isolated_home, monkeypatch):
    for k in ("GEMINI_API_KEY", "LLM_API_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert llm.detect_available_provider() == "local"
    hosted.write_credentials({"hosted_key": "sk-abc"})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": []}))
    hosted.reset_probe_cache()
    assert llm.detect_available_provider() == "hosted"
    llm._ollama_cached = True
    assert llm.detect_available_provider() == "ollama"


def test_empty_base_url_disables_the_tier(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "sk-abc"})
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "")
    assert hosted.hosted_base_url() is None
    assert hosted.probe_hosted() is False


def test_device_login_polls_until_key_and_stores_it(isolated_home, monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append(url)
        if url.endswith("/device/code"):
            return httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 0,
                },
            )
        if len(posts) < 4:
            return httpx.Response(428, json={"error": "authorization_pending", "interval": 0})
        return httpx.Response(200, json={"api_key": "sk-new", "user": "octocat"})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(hosted.time, "sleep", lambda s: None)
    lines = []
    key = hosted.device_login(open_browser=False, echo=lines.append)
    assert key == "sk-new"
    assert "ABCD-1234" in lines[0]
    data = json.loads((isolated_home / "morie" / "credentials.json").read_text())
    assert data["hosted_key"] == "sk-new" and data["hosted_user"] == "octocat"
    assert posts[0].endswith("/device/code") and posts[-1].endswith("/device/token")
