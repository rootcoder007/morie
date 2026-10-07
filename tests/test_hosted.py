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


def test_provider_order_puts_hosted_after_local_ollama_and_after_every_cloud_key(isolated_home, monkeypatch):
    for k in ("GEMINI_API_KEY", "LLM_API_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert llm.detect_available_provider() == "local"
    hosted.write_credentials({"hosted_key": "sk-abc"})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": []}))
    hosted.reset_probe_cache()
    assert llm.detect_available_provider() == "hosted"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert llm.detect_available_provider() == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert llm.detect_available_provider() == "openai"
    # and in the fallback chain the hosted tier is the last attempt
    attempts = llm._provider_attempts("openai", None)
    assert [a[0] for a in attempts][-1] == hosted.hosted_base_url()
    monkeypatch.delenv("OPENAI_API_KEY")
    llm._ollama_cached = True
    assert llm.detect_available_provider() == "ollama"
    assert "rmorie.com/access" in hosted.access_hint()


def test_empty_base_url_disables_the_tier(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "sk-abc"})
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "")
    assert hosted.hosted_base_url() is None
    assert hosted.probe_hosted() is False
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "OFF")  # the spelling that works on Windows too
    assert hosted.hosted_base_url() is None


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


def test_email_login_sends_code_then_stores_key(isolated_home, monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/email/code"):
            return httpx.Response(200, json={"sent": True, "expires_in": 600})
        assert json == {"email": "vee@example.com", "code": "123456"}
        return httpx.Response(200, json={"api_key": "sk-mail", "user": "mail:abc"})

    monkeypatch.setattr(httpx, "post", fake_post)
    lines = []
    key = hosted.email_login("Vee@Example.com ", ask=lambda _: " 123456 ", echo=lines.append)
    assert key == "sk-mail"
    assert posts[0][1] == {"email": "vee@example.com"}
    assert hosted.hosted_key() == "sk-mail"
    assert any("6-digit" in line for line in lines)


def test_email_login_reports_service_errors(isolated_home, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json=None, timeout=None: httpx.Response(
            429, json={"error": "too many codes requested; try again later"}
        ),
    )
    with pytest.raises(RuntimeError, match="too many codes"):
        hosted.email_login("vee@example.com", ask=lambda _: "000000")
    with pytest.raises(ValueError):
        hosted.email_login("not-an-address")


def test_store_token_writes_the_key_and_probes_it(isolated_home, monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["auth"] = headers["Authorization"]
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(httpx, "get", fake_get)
    lines = []
    assert hosted.store_token("  sk-pasted  ", echo=lines.append) == "sk-pasted"
    assert hosted.read_credentials()["hosted_key"] == "sk-pasted"
    assert seen["auth"] == "Bearer sk-pasted"
    assert "accepts it" in lines[0]
    with pytest.raises(ValueError):
        hosted.store_token("   ")


def test_email_login_can_have_the_key_mailed(isolated_home, monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/email/code"):
            return httpx.Response(200, json={"sent": True})
        return httpx.Response(200, json={"sent": True, "user": "mail:abc"})

    monkeypatch.setattr(httpx, "post", fake_post)
    lines = []
    out = hosted.email_login("vee@example.com", code="123456", echo=lines.append, to_email=True)
    assert out == ""
    assert posts[-1][1] == {"email": "vee@example.com", "code": "123456", "deliver": "email"}
    assert "hosted_key" not in hosted.read_credentials()
    assert "morie login --token" in lines[0]


def test_hosted_model_falls_back_to_what_the_gateway_lists(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "sk-abc"})
    listed = {"data": [{"id": "gemma4:31b-cloud"}, {"id": "minimax-m3:cloud"}]}
    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: httpx.Response(200, json=listed))
    monkeypatch.setenv("MORIE_HOSTED_MODEL", "qwen3.5:397b-cloud")  # retired upstream
    assert hosted.hosted_model() == "qwen3.5:397b-cloud"
    assert hosted.hosted_model_available() == "gemma4:31b-cloud"
    monkeypatch.setenv("MORIE_HOSTED_MODEL", "minimax-m3:cloud")
    assert hosted.hosted_model_available() == "minimax-m3:cloud"
    hosted.reset_probe_cache()
    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: httpx.Response(500))
    monkeypatch.setenv("MORIE_HOSTED_MODEL", "anything:cloud")
    assert hosted.hosted_model_available() == "anything:cloud"  # no list known: keep the configured name


def test_models_lines_cover_every_hosted_state(monkeypatch):
    from morie import hosted

    monkeypatch.setattr(
        hosted, "status", lambda: {"base_url": None, "logged_in": False, "user": "", "reachable": False}
    )
    assert hosted.models_lines() == ["Hosted tier: disabled (MORIE_HOSTED_BASE_URL is empty)"]
    monkeypatch.setattr(
        hosted,
        "status",
        lambda: {"base_url": "https://llm.rmorie.com", "logged_in": False, "user": "", "reachable": False},
    )
    assert "not logged in" in hosted.models_lines()[0]
    monkeypatch.setattr(
        hosted,
        "status",
        lambda: {"base_url": "https://llm.rmorie.com", "logged_in": True, "user": "gh:vee", "reachable": False},
    )
    assert hosted.models_lines()[0].startswith("Hosted tier (https://llm.rmorie.com): logged in, gateway not reachable")
    monkeypatch.setattr(
        hosted,
        "status",
        lambda: {"base_url": "https://llm.rmorie.com", "logged_in": True, "user": "gh:vee", "reachable": True},
    )
    monkeypatch.setattr(hosted, "hosted_models", lambda: ["a:cloud", "b:cloud"])
    monkeypatch.setattr(hosted, "hosted_model_available", lambda: "b:cloud")
    assert hosted.models_lines() == [
        "Hosted tier (https://llm.rmorie.com), logged in as gh:vee; default marked *:",
        "    a:cloud",
        "  * b:cloud",
    ]


def test_only_a_public_https_page_is_opened():
    assert hosted._browsable("https://github.com/login/device")
    assert hosted._browsable("https://llm.rmorie.com:8443/auth/x")
    for u in (
        "http://github.com/login/device",
        "https://127.0.0.1/",
        "https://localhost/",
        "https://10.0.0.5/x",
        "https://192.168.1.9/",
        "https://169.254.169.254/latest",
        "https://100.64.0.1/",
        "https://[::1]/",
        "https://metadata.google.internal/",
        "https://box.lan/",
        "https://metadata/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        None,
        "",
        3,
    ):
        assert not hosted._browsable(u), u
