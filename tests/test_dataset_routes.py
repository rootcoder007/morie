# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every catalog key has a route a user can take: portal download, the data.rmorie.com copy, or a named own file."""

import io
import zipfile

import pytest

from morie import data
from morie.fn import _frame_core as pd


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_user_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setenv("MORIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(data, "_builtin_db_connect", lambda: None)
    return tmp_path


def test_routes_name_a_source_for_the_health_infobase_and_otis_keys():
    cat = data.DATASET_CATALOG
    assert data.dataset_route(cat["hibua"]).startswith("health-infobase.canada.ca (or data.rmorie.com)")
    assert data.dataset_route(cat["hibp"]) == "data.rmorie.com (your MORIE key)"
    assert data.dataset_route(cat["otisfin"]).startswith("data.rmorie.com file")
    assert data.dataset_route(cat["otisloc"]) == "data.ontario.ca"
    assert data.dataset_route(cat["mapq"]).startswith("own file")
    own = [k for k, e in cat.items() if data.dataset_route(e).startswith("own file")]
    assert not any(k.startswith("hib") or k.startswith("otis") for k in own)


def _zip_with(member: str, csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, csv)
    return buf.getvalue()


def test_health_infobase_table_comes_from_the_portal_zip(isolated, monkeypatch):
    def fake_download(url, dest, timeout=60, label=None):
        assert "health-infobase.canada.ca" in url
        dest.write_bytes(_zip_with("Alcohol.csv", "question,label,colpercent\nLifetime,Yes,55.1\nLifetime,No,44.9\n"))
        return url

    monkeypatch.setattr(data, "_download_file", fake_download)
    df = data.load_dataset("hibua", db_path=isolated / "c.db")
    assert list(df.columns) == ["question", "label", "colpercent"] and len(df) == 2


def test_health_infobase_falls_back_to_the_hosted_copy_and_asks_for_a_login(isolated, monkeypatch):
    def failing(url, dest, timeout=60, label=None):
        raise OSError("portal down")

    monkeypatch.setattr(data, "_download_file", failing)
    from morie import datahub, hosted

    monkeypatch.setattr(hosted, "hosted_key", lambda: None)
    with pytest.raises(RuntimeError, match="morie login"):
        data.load_dataset("hibub", db_path=isolated / "c.db")
    monkeypatch.setattr(hosted, "hosted_key", lambda: "sk-test")
    monkeypatch.setattr(
        datahub, "load_hosted_dataset", lambda key, db_path=None, refresh=False: pd.DataFrame({"k": [key]})
    )
    df = data.load_dataset("hibub", db_path=isolated / "c.db")
    assert list(df["k"]) == ["hib/csus_cannabis"]
    # a key with only a hosted copy (no portal file) goes straight there
    df = data.load_dataset("hibp", db_path=isolated / "c2.db")
    assert list(df["k"]) == ["hib/cpads_cpads"]


def test_an_r_object_is_saved_and_the_user_is_told_where(isolated, monkeypatch):
    from morie import datahub

    def fake_get(path, dest, label, timeout=600):
        assert path == "/files/otis/dt_expanded.rds"
        dest.write_bytes(b"RDS")
        return 3

    monkeypatch.setattr(datahub, "_get_to_file", fake_get)
    with pytest.raises(NotImplementedError, match="rmorie::morie_load_dataset\\('otisexp'\\)"):
        data.load_dataset("otisexp", db_path=isolated / "c.db")
    assert (isolated / "data" / "cache" / "dt_expanded.rds").read_bytes() == b"RDS"
