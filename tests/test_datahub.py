# SPDX-License-Identifier: AGPL-3.0-or-later
"""data.rmorie.com: key handling, manifest cache, download + dataset-store cache, list_datasets rows."""

import gzip
import io
import json
from urllib.error import HTTPError

import pytest

from morie import data, datahub, hosted


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


MANIFEST = {
    "generated_utc": "2026-10-01T00:00:00Z",
    "datasets": [
        {
            "db": "chicago_crime",
            "table": "incidents",
            "key": "chicago_crime/incidents",
            "rows": 3,
            "columns": ["id", "type"],
            "bytes_gz": 10,
            "sha256": "x",
            "source": "bigquery-public-data.chicago_crime.crime",
            "meta": {"description": "Chicago Police incidents"},
        }
    ],
}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_user_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(hosted, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.delenv("MORIE_HOSTED_KEY", raising=False)
    monkeypatch.setenv("MORIE_DATA_URL", "https://data.example.test")
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append((req.full_url, req.headers.get("Authorization")))
        if req.headers.get("Authorization") != "Bearer sk-good":
            raise HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        if req.full_url.endswith("/manifest.json"):
            return _Resp(json.dumps(MANIFEST).encode())
        if req.full_url.endswith("/chicago_crime/incidents.csv.gz"):
            return _Resp(gzip.compress(b"id,type\n1,THEFT\n2,BATTERY\n3,THEFT\n"))
        raise HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(datahub, "urlopen", fake_urlopen)
    return calls


def test_no_key_is_a_clear_error(hub):
    with pytest.raises(datahub.DataHubAuthError, match="morie login"):
        datahub.hosted_manifest()


def test_rejected_key_says_login_again(hub, monkeypatch):
    monkeypatch.setenv("MORIE_HOSTED_KEY", "sk-stale")
    with pytest.raises(datahub.DataHubAuthError, match="login` again"):
        datahub.hosted_manifest()


def test_manifest_is_cached_and_download_is_cached_in_the_store(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("MORIE_HOSTED_KEY", "sk-good")
    m = datahub.hosted_manifest()
    assert m["datasets"][0]["key"] == "chicago_crime/incidents"
    assert datahub.hosted_manifest()["datasets"] == m["datasets"]  # second call: cache file, no request
    assert sum(u.endswith("/manifest.json") for u, _ in hub) == 1
    assert all(a == "Bearer sk-good" for _, a in hub)
    db = tmp_path / "cache.db"
    df = data.load_dataset("chicago_crime/incidents", db_path=db)
    assert list(df.columns) == ["id", "type"] and len(df) == 3
    again = data.load_dataset("chicago_crime/incidents", db_path=db)
    assert len(again) == 3
    assert sum(u.endswith(".csv.gz") for u, _ in hub) == 1  # served from the dataset store the second time
    rows = {d["key"]: d for d in data.list_datasets(db_path=db)}
    hubrow = rows["chicago_crime/incidents"]
    assert hubrow["route"].startswith("data.rmorie.com") and hubrow["cached"] and hubrow["rows"] == 3
    assert "ocp21" in rows


def test_unknown_plain_key_still_raises(hub):
    with pytest.raises(KeyError, match="data.rmorie.com"):
        data.load_dataset("no-such-dataset")


def test_pull_parses_hosted_keys():
    from morie.runner import build_parser

    assert build_parser().parse_args(["pull", "chicago_crime/incidents"]).dataset == "chicago_crime/incidents"


def test_empty_cached_table_is_refetched(hub, monkeypatch, tmp_path):
    from morie.fn import _frame_core as pd

    monkeypatch.setenv("MORIE_HOSTED_KEY", "sk-good")
    db = tmp_path / "cache.db"
    data.cache_store(pd.DataFrame({"id": [], "type": []}), datahub.hosted_table_name("chicago_crime/incidents"), db)
    df = datahub.load_hosted_dataset("chicago_crime/incidents", db_path=db)
    assert len(df) == 3  # the empty cached copy was ignored and the edge copy fetched
