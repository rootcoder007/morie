# SPDX-License-Identifier: AGPL-3.0-or-later
"""`morie pull` takes any catalog key; CKAN downloads fall back to the Wayback Machine."""

import io
import json
from urllib.error import URLError

import pytest

from morie import data
from morie.runner import build_parser


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _snapshot_body():
    return json.dumps(
        {
            "archived_snapshots": {
                "closest": {"available": True, "url": "http://web.archive.org/web/2026/https://x.ca/f.csv"}
            }
        }
    ).encode()


def test_pull_parses_catalog_keys_and_all():
    p = build_parser()
    assert p.parse_args(["pull", "ocp21"]).dataset == "ocp21"
    a = p.parse_args(["pull", "--all", "--out", "d"])
    assert a.all and str(a.out) == "d" and a.dataset is None


def test_wayback_snapshot_url(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=0):
        seen["url"] = url
        return _Resp(_snapshot_body())

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    assert data.wayback_snapshot_url("https://x.ca/f.csv") == "https://web.archive.org/web/2026/https://x.ca/f.csv"
    assert "archive.org/wayback/available?url=https%3A%2F%2Fx.ca%2Ff.csv" in seen["url"]
    monkeypatch.setattr(data, "urlopen", lambda url, timeout=0: _Resp(b'{"archived_snapshots": {}}'))
    assert data.wayback_snapshot_url("https://x.ca/f.csv") is None


def test_download_file_uses_snapshot_when_live_fails(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(req, timeout=0):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if url == "https://x.ca/f.csv":
            raise URLError("site down")
        if "wayback/available" in url:
            return _Resp(_snapshot_body())
        return _Resp(b"a,b\n1,2\n")

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    dest = tmp_path / "f.csv"
    src = data._download_file("https://x.ca/f.csv", dest)
    assert src.startswith("https://web.archive.org/")
    assert dest.read_bytes() == b"a,b\n1,2\n"
    assert calls[0] == "https://x.ca/f.csv"


def test_download_file_raises_when_no_snapshot(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=0):
        url = req if isinstance(req, str) else req.full_url
        if "wayback/available" in url:
            return _Resp(b'{"archived_snapshots": {}}')
        raise URLError("site down")

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    with pytest.raises(URLError):
        data._download_file("https://x.ca/f.csv", tmp_path / "f.csv")


def test_ckan_resource_file_reads_the_download(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=0):
        url = req if isinstance(req, str) else req.full_url
        if "resource_show" in url:
            return _Resp(json.dumps({"result": {"url": "https://x.ca/f.csv"}}).encode())
        if url == "https://x.ca/f.csv":
            return _Resp(b"a,b\n1,2\n3,4\n")
        raise URLError("unexpected " + url)

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    monkeypatch.setattr(data, "_user_cache_dir", lambda: tmp_path)
    df = data._ckan_resource_file("abc", "demo")
    assert list(df.columns) == ["a", "b"] and len(df) == 2
    assert (tmp_path / "ckan" / "abc" / "f.csv").exists()


def test_download_bootstrap_maps_surveys_to_catalog_keys(monkeypatch, capsys):
    from morie import data as d
    from morie.runner import _main_impl

    seen = []
    monkeypatch.setattr(
        d,
        "fetch_ckan_to_cache",
        lambda key, limit=0, **_kw: (
            seen.append(key) or __import__("morie.fn._frame_core", fromlist=["DataFrame"]).DataFrame({"w": [1, 2]})
        ),
    )
    monkeypatch.setattr("sys.argv", ["morie", "download-bootstrap", "--survey", "csus_2023"])
    assert _main_impl() == 0
    assert seen == ["cu23bt"]
    with pytest.raises(SystemExit):  # argparse rejects an unknown survey
        build_parser().parse_args(["download-bootstrap", "--survey", "zzz_1999"])


def test_pull_routes_hosted_keys_to_load_dataset(monkeypatch, tmp_path):
    from morie import runner
    from morie.fn import _frame_core as pd

    monkeypatch.setattr(runner, "load_dataset", lambda key, **kw: pd.DataFrame({"k": [key]}), raising=False)
    monkeypatch.setattr("morie.data.load_dataset", lambda key, **kw: pd.DataFrame({"k": [key]}))
    monkeypatch.setattr(runner, "_hub_rows", lambda key: 10)  # a small table: the frame route (big ones stream)
    out = tmp_path / "o.csv"
    monkeypatch.setattr("sys.argv", ["morie", "pull", "chicago_crime/incidents", "--out", str(out)])
    assert runner._main_impl() == 0
    assert pd.read_csv(out)["k"].tolist() == ["chicago_crime/incidents"]
