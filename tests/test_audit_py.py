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
    monkeypatch.setenv("MORIE_HOSTED_KEY", KEY)  # not read: the wording is the same
    assert "morie log" not in datahub._refusal_message(401, "/open/t.csv.gz")


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


# ---- the rest of the audit ----


def test_the_data_url_override_is_https_or_this_machine(monkeypatch):
    for ok in ("https://data.example.test/", "http://127.0.0.1:8080", "http://localhost:9"):
        monkeypatch.setenv("MORIE_DATA_URL", ok)
        assert datahub.data_url() == ok.rstrip("/")
    for bad in ("http://data.example.test", "ftp://x.example", "data.example.test", "http://10.0.0.5"):
        monkeypatch.setenv("MORIE_DATA_URL", bad)
        with pytest.raises(ValueError, match="must be an https address"):
            datahub.data_url()


def test_an_r_object_is_checked_against_the_recorded_digest(tmp_path, monkeypatch):
    body = b"not the recorded file"
    monkeypatch.setattr(data, "_data_dir_candidates", lambda: [tmp_path])
    monkeypatch.setattr(datahub, "_get_to_file", lambda path, dest, label, timeout=600: dest.write_bytes(body))
    entry = {"local_path": "data/cache/x.rds", "hosted_file": "otis/x.rds", "hosted_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="does not match the SHA-256 this package records"):
        data._load_hosted_file(entry, "x")
    assert not any(tmp_path.rglob("x.rds*"))
    # the three hosted R objects each record one
    hosted_files = [e for e in data.DATASET_CATALOG.values() if e.get("hosted_file")]
    assert hosted_files and all(len(e.get("hosted_sha256", "")) == 64 for e in hosted_files)
    # the right bytes are kept under the final name (what follows is the reader's business)
    entry["hosted_sha256"] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(data, "_r_object_frame", lambda p: "frame")
    assert data._load_hosted_file(entry, "x") == "frame"
    assert (tmp_path / "cache" / "x.rds").read_bytes() == body


def test_gamma_and_zeta_at_infinity():
    import math

    from morie.fn import _rrng_core, _stats_core, _tail1core

    assert _stats_core._gammainc_p(2.0, math.inf) == 1.0
    assert _stats_core._gammainc_q(2.0, math.inf) == 0.0
    assert _tail1core._gammap(2.0, math.inf) == 1.0
    assert _tail1core.pgamma(math.inf, 2.0) == 1.0
    assert _rrng_core.pgamma(math.inf, 2.0) == 1.0
    assert _rrng_core.pgamma(math.inf, 2.0, lower_tail=False) == 0.0
    # the upper tail is computed directly: 1 - p gave 0 here
    q = _rrng_core.pgamma(60.0, 2.0, lower_tail=False)
    assert q == pytest.approx(61.0 * math.exp(-60.0), rel=1e-10) and q > 0
    assert _rrng_core.pgamma(1.5, 2.0, lower_tail=False) == pytest.approx(2.5 * math.exp(-1.5), rel=1e-12)
    assert _stats_core._Zipf._zeta(math.inf) == 1.0
    assert _stats_core._Zipf._zeta(2.0) == pytest.approx(math.pi**2 / 6, rel=1e-9)


def test_a_density_of_huge_values_fails_with_words():
    from morie.fn._stats_core import gaussian_kde

    for bad in ([1e300, -1e300, 5e299], [1.0, float("inf"), 2.0], [1.0, float("nan"), 2.0]):
        with pytest.raises(ValueError, match="finite and below 1e150"):
            gaussian_kde(bad)
    assert gaussian_kde([1.0, 2.0, 4.0, 7.0]).n == 4


def test_a_line_break_in_a_markdown_cell_stays_in_its_row():
    from morie import reporting
    from morie.fn import _frame_core as pd

    md = reporting._df_markdown(pd.DataFrame({"a": ["line1\nline2", "x|y"], "b": ["p\r\nq", "z"]}))
    lines = md.split("\n")
    assert len(lines) == 4 and "line1<br>line2" in lines[2] and "p<br>q" in lines[2] and "x\\|y" in lines[3]


def test_a_bundle_lists_only_files_inside_the_capsule(tmp_path):
    from morie.capsule import _sign

    cps = tmp_path / "cps"
    cps.mkdir()
    (cps / "a.txt").write_text("a")
    (tmp_path / "secret.txt").write_text("s")
    key = _sign.keygen() if hasattr(_sign, "keygen") else None
    if key is None:
        pytest.skip("no key generator at this name")
    for bad in (["../secret.txt"], [str(tmp_path / "secret.txt")], ["a.txt", "../secret.txt"]):
        with pytest.raises(ValueError, match="not files inside"):
            _sign.bundle(str(cps), {"m": 1}, key, files=bad)


def test_an_ingest_error_does_not_repeat_the_key(monkeypatch):
    import httpx

    from morie.ingest import canlii

    def handler(request):
        return httpx.Response(500, text=f"upstream said api_key={KEY} is bad")

    with pytest.raises(canlii.CanLIIError) as e:
        canlii._get_json(
            "caseBrowse/en/", {}, api_key=KEY, timeout=5, user_agent="t", transport=httpx.MockTransport(handler)
        )
    assert KEY not in str(e.value) and "[redacted]" in str(e.value)
