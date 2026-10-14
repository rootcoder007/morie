"""The data key: credentials file, key check, sign-in; and that it never makes a model route."""

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
        return httpx.Response(200, json={"user": "gh:vee"})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert hosted.probe_hosted() is True
    assert hosted.probe_hosted() is True
    assert seen == [(hosted.hosted_base_url() + "/v1/whoami", "Bearer sk-abc")]
    assert hosted.hosted_base_url() == hosted.DEFAULT_DATA_BASE_URL


def test_probe_falls_back_to_the_manifest_and_names_a_refusal(isolated_home, monkeypatch):
    hosted.write_credentials({"hosted_key": "sk-abc"})
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(url.rsplit("/", 1)[-1])
        return httpx.Response(404) if url.endswith("/v1/whoami") else httpx.Response(401)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert hosted.probe_hosted() is False
    assert seen == ["whoami", "manifest.json"]
    assert hosted.hosted_failure() == "rejected"
    assert "login" in hosted.hosted_problem_line()


def test_the_data_key_is_not_a_model_route(isolated_home, monkeypatch):
    from morie import providers

    for p in providers.PRESETS:
        for e in (p.key_env, p.alt_env):
            if e:
                monkeypatch.delenv(e, raising=False)
    for k in ("LLM_API_BASE_URL", "MORIE_LLM_BASE_URL", "MORIE_LLM_ROUTE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "off")
    hosted.write_credentials({"hosted_key": "sk-abc"})
    assert llm.detect_available_provider() == "local"  # a data key opens tables, never a model
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert llm.detect_available_provider() == "groq"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert llm.detect_available_provider() == "anthropic"  # the table's order
    monkeypatch.setenv("MORIE_LLM_BASE_URL", "http://localhost:1234/v1")
    assert llm.detect_available_provider() == "own"  # your own endpoint first
    with pytest.raises(ValueError, match="retired"):
        llm.detect_available_provider("hosted")
    assert "morie login" in hosted.access_hint()


def test_a_data_service_switched_off_sends_no_key(isolated_home, monkeypatch):
    from morie import services

    hosted.write_credentials({"hosted_key": "sk-abc"})
    monkeypatch.setattr(services, "data", lambda: {"mode": "off", "base_url": ""})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("no request when the service is off"))
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
        return httpx.Response(200, json={"user": "gh:vee"})

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


def test_only_a_public_https_page_is_opened():
    assert hosted._browsable("https://github.com/login/device")
    assert hosted._browsable("https://data.rmorie.com:8443/auth/x")
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
