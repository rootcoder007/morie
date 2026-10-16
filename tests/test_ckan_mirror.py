"""The CKAN survey route when open.canada.ca is down (2026-10-16): the pinned file URL, then the
data.rmorie.com mirror, checked against the SHA-256 its index records. Mirrors rmorie's
tests/testthat/test-ckan-mirror.R."""

import gzip
import hashlib
import json

import pytest

pd = pytest.importorskip("pandas")

from morie import data, datahub  # noqa: E402

CSV = b"SEQID,age\n1,20\n2,31\n"


def _index(sha=None, rid="rid-1", name="ocp21"):
    return json.dumps(
        {
            "files": {
                name: {
                    "file": f"{name}.csv.gz",
                    "original_name": "cpads.csv",
                    "sha256": sha or hashlib.sha256(CSV).hexdigest(),
                    "ckan_resource_id": rid,
                }
            }
        }
    ).encode()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_user_cache_dir", lambda: tmp_path)
    return tmp_path


def _serve(monkeypatch, index, calls=None):
    monkeypatch.setattr(datahub, "_get", lambda path, timeout=60: index)

    def get_to_file(path, dest, label, timeout=600):
        if calls is not None:
            calls.append(path)
        dest.write_bytes(gzip.compress(CSV))
        return len(CSV)

    monkeypatch.setattr(datahub, "_get_to_file", get_to_file)


def test_mirror_read_and_checked(cache, monkeypatch):
    calls = []
    _serve(monkeypatch, _index(), calls)
    df = data._ckan_mirror("rid-1", "cpads")
    assert list(df["age"]) == [20, 31]
    # the checked file already there is reused
    df2 = data._ckan_mirror("rid-1", "cpads")
    assert len(df2) == 2 and len(calls) == 1


def test_mirror_sha_mismatch_refused(cache, monkeypatch):
    _serve(monkeypatch, _index(sha="0" * 64, name="bad"))
    with pytest.raises(RuntimeError, match="does not match the SHA-256"):
        data._ckan_mirror("rid-1", "cpads")
    assert not (cache / "ckan-mirror" / "bad.csv").exists()


def test_mirror_without_entry(cache, monkeypatch):
    _serve(monkeypatch, _index(rid="other"))
    with pytest.raises(RuntimeError, match="holds no copy of csads"):
        data._ckan_mirror("rid-404", "csads")


def test_route_live_then_mirror_then_both(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("open.canada.ca could not be reached (504)")

    monkeypatch.setattr(data, "_ckan_resource_file_live", boom)
    monkeypatch.setattr(data, "_ckan_mirror", lambda rid, key: pd.DataFrame({"SEQID": [1, 2, 3]}))
    assert len(data._ckan_resource_file("rid-1", "cpads")) == 3

    def no_key(*a, **k):
        raise RuntimeError("the curated tables need your MORIE data key")

    monkeypatch.setattr(data, "_ckan_mirror", no_key)
    with pytest.raises(RuntimeError, match=r"\(504\); the data.rmorie.com mirror: the curated tables need"):
        data._ckan_resource_file("rid-1", "cpads")

    monkeypatch.setattr(data, "_ckan_resource_file_live", lambda *a, **k: pd.DataFrame({"a": [1]}))
    assert list(data._ckan_resource_file("rid-1", "cpads")["a"]) == [1]


def test_pinned_url_when_metadata_down(cache, monkeypatch):
    rid = next(iter(data.CKAN_FILES))
    seen = []

    def down(*a, **k):
        raise OSError("timed out")

    monkeypatch.setattr(data, "urlopen", down)

    def fake_dl(url, dest, timeout=60, label=None):
        seen.append(url)
        dest.write_bytes(b"SEQID\n7\n")
        return url

    monkeypatch.setattr(data, "_download_file", fake_dl)
    df = data._ckan_resource_file_live(rid, "cpads")
    assert seen == [data.CKAN_FILES[rid]]
    assert list(df["SEQID"]) == [7]
    with pytest.raises(RuntimeError, match=r"could not be reached \(timed out\); nothing cached for x"):
        data._ckan_resource_file_live("not-pinned", "x")
