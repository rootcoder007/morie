"""The data client against the audit of 2026-11: tables that are not served, where the key goes,
what an error may repeat, the sign-in poll interval, and downloads that cannot be checked."""

import base64
import gzip
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request

import pytest

from morie import data, datahub, hosted
from morie._redact import key_forms, redact

KEY = "sk-" + "a1B2c3D4" * 6


class _Resp:
    def __init__(self, body):
        self._b = body

    def read(self, n=-1):
        b, self._b = (self._b, b"") if n < 0 else (self._b[:n], self._b[n:])
        return b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    headers = {}
    status = 200


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_user_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(hosted, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.delenv("MORIE_HOSTED_KEY", raising=False)
    monkeypatch.delenv("MORIE_OFFLINE", raising=False)
    monkeypatch.setenv("MORIE_DATA_URL", "https://data.example.test")
    monkeypatch.setattr(hosted, "hosted_base_url", lambda: "https://data.example.test")
    return tmp_path


MANIFEST = {
    "datasets": [
        {"key": "open/t", "rows": 1},
        {"key": "icpsr_99999/fake", "rows": 1},
        {"key": "ncvs_38604/fake", "rows": 1},
        {"key": "other/marked", "rows": 1, "access": "restricted", "source": "Its Owner"},
        {"key": "other/flag", "rows": 1, "restricted": True},
        {"key": "other/meta", "rows": 1, "meta": {"access": "terms"}},
        {"key": "other/public", "rows": 1, "access": "Public"},
    ]
}


def test_unserved_tables_are_not_listed():
    keys = [e["key"] for e in datahub.hosted_entries(MANIFEST)]
    assert keys == ["open/t", "other/public"]
    assert datahub.unserved("open/t") is None
    assert "ICPSR" in datahub.unserved("icpsr_99999/fake")
    assert "ICPSR" in datahub.unserved("ncvs_38604/fake")
    assert datahub.unserved("ncvs_386040/x") is None  # another study's number
    assert "Its Owner" in datahub.unserved("other/marked", MANIFEST["datasets"][3])


@pytest.mark.parametrize("key", ["icpsr_99999/fake", "ncvs_38604/fake", "other/marked", "other/flag", "other/meta"])
def test_an_unserved_table_is_refused_before_any_request_or_cache(hub, monkeypatch, key):
    (hub / "data_rmorie_manifest.json").write_text(json.dumps(MANIFEST))
    asked = []
    monkeypatch.setattr(datahub, "urlopen", lambda req, timeout=0: asked.append(req.full_url))
    monkeypatch.setattr(data, "cache_load", lambda *a, **k: asked.append("cache"))
    with pytest.raises(datahub.DataHubAuthError, match="not served by data.rmorie.com"):
        datahub.load_hosted_dataset(key)
    with pytest.raises(datahub.DataHubAuthError, match="not served by data.rmorie.com"):
        datahub.hosted_to_csv(key, hub / "out.csv")
    with pytest.raises(KeyError, match="not served"):
        data.dataset_info(key)
    assert asked == []


def test_a_direct_path_to_an_icpsr_table_makes_no_request(hub, monkeypatch):
    asked = []
    monkeypatch.setattr(datahub, "urlopen", lambda req, timeout=0: asked.append(req.full_url))
    with pytest.raises(datahub.DataHubAuthError, match="download it from ICPSR"):
        datahub._get("/icpsr_99999/fake.csv.gz")
    with pytest.raises(datahub.DataHubAuthError, match="download it from ICPSR"):
        datahub._get_to_file("/ncvs_38604/fake.csv.gz", hub / "x.gz", "x")
    assert asked == []


def test_a_410_for_another_table_is_not_worded_as_icpsr(hub, monkeypatch):
    def gone(req, timeout=0):
        raise HTTPError(req.full_url, 410, "Gone", {}, None)

    monkeypatch.setattr(datahub, "urlopen", gone)
    with pytest.raises(datahub.DataHubAuthError) as e:
        datahub._get("/other/marked.csv.gz")
    assert "ICPSR" not in str(e.value) and "its source" in str(e.value)


def test_refusals_no_longer_send_anyone_to_sign_in(hub, monkeypatch):
    for code in (401, 403):
        said = datahub._refusal_message(code, "/open/t.csv.gz")
        assert "need no sign-in" in said and "morie login" not in said and "terms" not in said
    monkeypatch.setenv("MORIE_HOSTED_KEY", KEY)
    assert "morie logout" in datahub._refusal_message(401, "/open/t.csv.gz")


def test_the_key_does_not_follow_a_redirect():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.setdefault(self.server.server_port, []).append(self.headers.get("Authorization"))
            if self.path == "/go":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{other.server_port}/landing")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    first, other = HTTPServer(("127.0.0.1", 0), H), HTTPServer(("127.0.0.1", 0), H)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (first, other)]
    for t in threads:
        t.start()
    try:
        req = Request(f"http://127.0.0.1:{first.server_port}/go", headers={"Authorization": f"Bearer {KEY}"})
        with datahub.urlopen(req, timeout=10) as r:
            assert r.read() == b"ok"
    finally:
        for s in (first, other):
            s.shutdown()
            s.server_close()
    assert seen[first.server_port] == [f"Bearer {KEY}"]
    assert seen[other.server_port] == [None]


def _echoes(key):
    raw = key.encode()
    out = [key, quote(key, safe=""), raw.hex(), raw.hex().upper(), key.replace("-", ""), "-".join(key)]
    for pad in (b"", b"x", b"xy"):
        out.append(base64.b64encode(pad + raw + b"tail").decode())
        out.append(base64.urlsafe_b64encode(pad + raw + b"?>~").decode())
    return out


@pytest.mark.parametrize("i", range(12))
def test_no_form_of_the_key_survives_redaction(i):
    echo = _echoes(KEY)[i]
    said = redact(f"Internal echo {echo} end", KEY)
    for f in key_forms(KEY):
        assert f not in said
    assert KEY[3:20] not in said and "[redacted]" in said
    assert redact("nothing here", KEY) == "nothing here"
    assert redact("short secrets are left", "abc") == "short secrets are left"


def test_an_error_from_the_data_service_does_not_repeat_the_key(hub, monkeypatch):
    monkeypatch.setenv("MORIE_HOSTED_KEY", KEY)

    def boom(req, timeout=0):
        raise HTTPError(req.full_url, 500, f"Internal echo {KEY} / {base64.b64encode(KEY.encode()).decode()}", {}, None)

    monkeypatch.setattr(datahub, "urlopen", boom)
    with pytest.raises(HTTPError) as e:
        datahub._get("/open/t.csv.gz")
    text = (
        str(e.value) + repr(e.value.__cause__) + repr(e.value.__context__ if not e.value.__suppress_context__ else "")
    )
    assert KEY not in text and "[redacted]" in str(e.value) and e.value.code == 500


def test_the_poll_interval_is_bounded():
    f = hosted._poll_interval
    assert [f(v) for v in (1, 5, 60, "7")] == [1.0, 5.0, 60.0, 7.0]
    assert [f(v) for v in (0, -1, 0.5, 61, 1e6, float("nan"), float("inf"), "abc", None, [])] == [5.0] * 10
    assert f(1e9, 12.0) == 12.0


def test_the_device_poll_never_sleeps_past_its_deadline(hub, monkeypatch):
    slept = []

    class R:
        def __init__(self, code, body):
            self.status_code, self._b, self.text = code, body, json.dumps(body)

        def json(self):
            return self._b

    def post(url, json=None, timeout=0):
        if url.endswith("/device/code"):
            return R(
                200,
                {
                    "user_code": "U",
                    "verification_uri": "https://x.example",
                    "device_code": "dc-" + "b" * 30,
                    "interval": 1e6,
                },
            )
        return R(428, {"interval": 0})

    monkeypatch.setattr(hosted.httpx, "post", post)
    monkeypatch.setattr(hosted, "hosted_auth_url", lambda: "https://auth.example.test")
    monkeypatch.setattr(hosted.time, "sleep", lambda s: slept.append(s))
    clock = iter(range(0, 10_000, 2))
    monkeypatch.setattr(hosted.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        hosted.device_login(echo=lambda *a: None, open_browser=False, poll_max_seconds=20)
    assert slept and max(slept) <= 5.0 and min(slept) >= 0.0


def test_a_sign_in_error_does_not_repeat_the_device_code(hub, monkeypatch):
    code = "dc-" + "b" * 30

    class R:
        def __init__(self, status, body):
            self.status_code, self._b, self.text = status, body, json.dumps(body)

        def json(self):
            return self._b

    def post(url, json=None, timeout=0):
        if url.endswith("/device/code"):
            return R(200, {"user_code": "U", "verification_uri": "https://x.example", "device_code": code})
        return R(400, {"error": f"bad device_code {code}"})

    monkeypatch.setattr(hosted.httpx, "post", post)
    monkeypatch.setattr(hosted, "hosted_auth_url", lambda: "https://auth.example.test")
    monkeypatch.setattr(hosted.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as e:
        hosted.device_login(echo=lambda *a: None, open_browser=False)
    assert code not in str(e.value) and "answered 400" in str(e.value)


def _gz(text):
    return gzip.compress(text.encode())


def test_a_table_listed_without_a_digest_warns_and_one_no_manifest_lists_is_not_loaded(hub, monkeypatch):
    body = _gz("a,b\n1,2\n")
    f = hub / "t.csv.gz"
    f.write_bytes(body)
    monkeypatch.setattr(datahub, "hosted_manifest", lambda refresh=False: {"datasets": [{"key": "open/t"}]})
    with pytest.warns(UserWarning, match="lists no SHA-256"):
        datahub._check_digest(f, "open/t")
    assert f.exists()

    def none(refresh=False):
        raise OSError("no network")

    monkeypatch.setattr(datahub, "hosted_manifest", none)
    with pytest.raises(ValueError, match="cannot be checked"):
        datahub._check_digest(f, "open/t")
    assert not f.exists()


def test_a_mismatch_reads_the_manifest_again_once(hub, monkeypatch):
    body = _gz("a,b\n1,2\n")
    good = hashlib.sha256(body).hexdigest()
    f = hub / "t.csv.gz"
    f.write_bytes(body)
    calls = []

    def manifest(refresh=False):
        calls.append(refresh)
        return {"datasets": [{"key": "open/t", "sha256_gz": good if refresh or True in calls else "1" * 64}]}

    monkeypatch.setattr(datahub, "hosted_manifest", manifest)
    datahub._check_digest(f, "open/t")
    assert f.exists() and True in calls
    # a real mismatch still stops, and removes the file
    monkeypatch.setattr(
        datahub, "hosted_manifest", lambda refresh=False: {"datasets": [{"key": "open/t", "sha256": "2" * 64}]}
    )
    with pytest.raises(ValueError, match="does not match"):
        datahub._check_digest(f, "open/t")
    assert not f.exists()
