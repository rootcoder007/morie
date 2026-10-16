"""The terms and profile step before a new data key: the terms are accepted ([y/N] or
--accept-terms), a name and a reason are given, and Schedule B needs a supervisor, chair, director
or coordinator. Nothing that could issue a key is sent before that."""

import sys

import httpx
import pytest

from morie import hosted, runner

TERMS = {"version": "2026-10-16", "url": "https://data.example/terms"}


def answers(*values):
    it = iter(values)
    return lambda prompt: next(it, "")


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("MORIE_HOSTED_AUTH_URL", "https://data.example/auth")


def test_terms_info_reads_version_and_address(auth, monkeypatch):
    seen = []

    def fake_get(url, timeout=None):
        seen.append(url)
        return httpx.Response(200, json=TERMS)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert hosted.terms_info() == TERMS
    assert seen == ["https://data.example/terms.json"]
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: httpx.Response(200, json={"version": "2026-10-16"}))
    assert hosted.terms_info()["url"] == "https://data.example/terms"


def test_terms_info_unreadable_stops_the_sign_in(auth, monkeypatch):
    monkeypatch.setenv("MORIE_HOSTED_AUTH_URL", "https://data.example")  # an address without /auth
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: httpx.Response(503, json={}))
    with pytest.raises(RuntimeError, match="could not read the data access terms"):
        hosted.terms_info()
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: httpx.Response(200, json={"version": "latest"}))
    with pytest.raises(RuntimeError, match="could not read"):
        hosted.terms_info()

    def boom(url, timeout=None):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(RuntimeError, match="https://data.example/terms.json"):
        hosted.terms_info()


def test_terms_declined_at_the_prompt_issue_no_key(monkeypatch):
    monkeypatch.setattr(hosted, "terms_info", lambda: TERMS)
    lines = []
    with pytest.raises(ValueError, match="were not accepted, so no key was issued.*--accept-terms"):
        hosted.signin_profile(ask=answers("n"), echo=lines.append)
    assert "Data access terms, version 2026-10-16" in lines[0]


def test_accepted_sign_in_asks_name_organisation_reason_and_schedule_b(monkeypatch):
    monkeypatch.setattr(hosted, "terms_info", lambda: TERMS)
    p = hosted.signin_profile(
        ask=answers("YES", "Ada Lovelace", "", "Course project on court delays in Ontario.", "n"), echo=lambda _: None
    )
    assert p["accept_terms"] == "2026-10-16" and p["name"] == "Ada Lovelace"
    assert "affiliation" not in p and p["accept_restricted"] is False
    assert p["purpose"] == "Course project on court delays in Ontario."
    assert p["client"].startswith("morie/") and "supervisor_name" not in p


def test_name_and_reason_are_required(monkeypatch):
    monkeypatch.setattr(hosted, "terms_info", lambda: TERMS)
    with pytest.raises(ValueError, match="your name is needed for a key: pass --name"):
        hosted.signin_profile(accept_terms=True, ask=answers(""))
    with pytest.raises(ValueError, match="the reason you need the data.*--reason"):
        hosted.signin_profile(accept_terms=True, name="Ada", purpose="too short", ask=answers(""))


def test_schedule_b_needs_a_supervisor_with_role_and_email(monkeypatch):
    monkeypatch.setattr(hosted, "terms_info", lambda: TERMS)
    base = dict(accept_terms=True, name="Ada", affiliation="Uni", purpose="Thesis chapter on court delays in Ontario.")
    with pytest.raises(ValueError, match="supervisor's, chair's, director's or coordinator's name"):
        hosted.signin_profile(restricted=True, ask=answers(""), **base)
    with pytest.raises(ValueError, match="their role, one of supervisor, chair"):
        hosted.signin_profile(restricted=True, supervisor_name="Dr B", supervisor_role="boss", ask=answers(""), **base)
    with pytest.raises(ValueError, match="their email address"):
        hosted.signin_profile(
            restricted=True, supervisor_name="Dr B", supervisor_role="Chair", supervisor_email="nope", **base
        )
    p = hosted.signin_profile(
        restricted=True, supervisor_name="Dr B", supervisor_role="Chair", supervisor_email="B@Uni.example", **base
    )
    assert p["accept_restricted"] is True and p["affiliation"] == "Uni"
    assert (p["supervisor_name"], p["supervisor_role"], p["supervisor_email"]) == ("Dr B", "chair", "b@uni.example")


def test_schedule_b_chosen_at_the_prompt_asks_for_the_supervisor(monkeypatch):
    monkeypatch.setattr(hosted, "terms_info", lambda: TERMS)
    p = hosted.signin_profile(
        accept_terms=True,
        ask=answers(
            "Ada", "Uni", "Thesis chapter on court delays in Ontario.", "y", "Dr B", "director", "b@uni.example"
        ),
    )
    assert p["accept_restricted"] is True and p["supervisor_role"] == "director"


def test_the_profile_goes_with_the_sign_in_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        return httpx.Response(503, json={"error": "down"})

    monkeypatch.setattr(httpx, "post", fake_post)
    profile = {"accept_terms": "2026-10-16", "name": "Ada"}
    with pytest.raises(RuntimeError):
        hosted.device_login(open_browser=False, echo=lambda _: None, profile=profile)
    assert posts[-1] == (posts[-1][0], profile) and posts[-1][0].endswith("/device/code")
    with pytest.raises(RuntimeError, match="down"):
        hosted.email_login("ada@example.org", ask=lambda _: "1", echo=lambda _: None, profile=profile)
    assert posts[-1][1] == {"email": "ada@example.org", **profile}


def test_morie_login_passes_the_flags_and_the_profile(monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(hosted, "read_credentials", lambda: {})
    monkeypatch.setattr(hosted, "signin_profile", lambda **k: got.update(k) or {"accept_terms": "v", "name": "Ada"})
    monkeypatch.setattr(hosted, "device_login", lambda **k: got.update(device=k))
    monkeypatch.setattr(hosted, "email_login", lambda email, **k: got.update(email=(email, k)))
    argv = [
        "morie",
        "login",
        "--no-browser",
        "--accept-terms",
        "--name",
        "Ada Lovelace",
        "--affiliation",
        "Uni",
        "--reason",
        "Thesis chapter on court delays.",
        "--restricted",
        "--supervisor-name",
        "Dr B",
        "--supervisor-role",
        "chair",
        "--supervisor-email",
        "b@uni.example",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    assert got["accept_terms"] is True and got["restricted"] is True
    assert (got["name"], got["affiliation"], got["purpose"]) == (
        "Ada Lovelace",
        "Uni",
        "Thesis chapter on court delays.",
    )
    assert (got["supervisor_name"], got["supervisor_role"], got["supervisor_email"]) == (
        "Dr B",
        "chair",
        "b@uni.example",
    )
    assert got["device"]["profile"] == {"accept_terms": "v", "name": "Ada"}
    got.clear()
    monkeypatch.setattr(sys, "argv", ["morie", "login", "--email", "ada@example.org"])
    assert runner.main() == 0
    assert got["accept_terms"] is False and got["restricted"] is None
    assert got["email"] == ("ada@example.org", {"to_email": False, "profile": {"accept_terms": "v", "name": "Ada"}})
    # a malformed address stops before the terms are asked
    got.clear()
    monkeypatch.setattr(sys, "argv", ["morie", "login", "--email", "not-an-address"])
    assert runner.main() == 1 and "accept_terms" not in got
    assert "an email address is required" in capsys.readouterr().out
