"""1.3.4: every catalog route reachable from a clean install, without network."""

import gzip
import io
import json
import tarfile
import zipfile
from urllib.error import HTTPError

import pytest

from morie import data as mdata
from morie import earth, siu_fetch


class _Reply:
    """A fake HTTP response: whole-body and chunked reads, headers, context manager."""

    headers: dict = {}

    def __init__(self, body: bytes):
        self._body = body
        self._pos = 0

    def read(self, n: int = -1):
        if n is None or n < 0:
            out, self._pos = self._body[self._pos :], len(self._body)
        else:
            out, self._pos = self._body[self._pos : self._pos + n], self._pos + n
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(mdata, "_builtin_db_connect", lambda: None)
    monkeypatch.setattr(mdata, "_project_root", lambda: tmp_path / "src")
    monkeypatch.setattr(mdata, "_user_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setenv("MORIE_DATA_DIR", str(tmp_path / "mydata"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "cache.db"


def test_own_files_resolve_through_morie_data_dir(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    f = tmp_path / "mydata" / "datasets" / "hib" / "CSUS" / "Alcohol.csv"
    f.parent.mkdir(parents=True)
    f.write_text("year,pct\n2023,12.5\n2024,11.0\n")
    df = mdata.load_dataset("hibua", db_path=db)
    assert len(df) == 2 and list(df.columns) == ["year", "pct"]
    assert mdata._find_local_file("data/datasets/hib/CSUS/Alcohol.csv") == f
    assert mdata._find_local_file("data/datasets/hib/CSUS/Nope.csv") is None


def test_otis_keys_use_the_ontario_downloader(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    seen = []

    def fake_load(dataset_id):
        seen.append(dataset_id)
        return mdata.pd.DataFrame.from_records([{"institution": "X", "n": 1}])

    monkeypatch.setattr("morie.otis_datasets.load_otis_dataset", fake_load)
    df = mdata.load_dataset("otisb09", db_path=db)
    assert seen == ["b09"] and len(df) == 1
    # cached under the catalog table name for the next call
    monkeypatch.setattr(
        "morie.otis_datasets.load_otis_dataset", lambda *_: (_ for _ in ()).throw(AssertionError("net"))
    )
    assert len(mdata.load_dataset("otisb09", db_path=db)) == 1


def _fake_rmoriedata_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:

        def add(name, payload: bytes):
            info = tarfile.TarInfo(f"rmoriedata/inst/extdata/{name}")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        add(
            "_catalog.csv",
            b'"slug","source_path","kind","n_rows","n_cols","legacy"\n'
            b'"siu_directors_reports","siu_directors_reports.csv.gz","table",2,3,\n'
            b'"tps_dictionary","tps_dictionary.json","dictionary",,,\n',
        )
        add(
            "_schema.csv",
            b'"slug","position","name","class"\n'
            b'"siu_directors_reports",1,"case_number","character"\n'
            b'"siu_directors_reports",2,"_language","character"\n'
            b'"siu_directors_reports",3,"officer_count","integer"\n',
        )
        add(
            "siu_directors_reports.csv.gz",
            gzip.compress(b"case_number,X_language,officer_count\n16-OFI-019,en,2\n20-OFD-082,fr,3\n"),
        )
        add("tps_dictionary.json", b"{}")
    return buf.getvalue()


def test_rmoriedata_tables_are_read_from_the_cran_tarball(monkeypatch, tmp_path):
    db = _isolate(monkeypatch, tmp_path)
    calls = []

    def fake_urlopen(url, timeout=30):
        calls.append(url)
        assert url == mdata.RMORIEDATA_TARBALL
        return _Reply(_fake_rmoriedata_tarball())

    monkeypatch.setattr("morie.data.urlopen", fake_urlopen)
    rows = mdata.list_rmoriedata()
    assert [r["slug"] for r in rows] == ["siu_directors_reports", "tps_dictionary"]
    df = mdata.load_dataset("siu", db_path=db)
    assert len(df) == 2
    assert list(df.columns) == ["case_number", "_language", "officer_count"]  # schema names, not the mangled header
    assert calls == [mdata.RMORIEDATA_TARBALL]  # fetched once, then served from the extracted copy
    with pytest.raises(ValueError, match="dictionary"):
        mdata.load_rmoriedata("tps_dictionary")
    with pytest.raises(KeyError, match="siu_directors_reports"):
        mdata.load_rmoriedata("nope")


def test_cihi_indicator_library_is_downloaded_once(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    calls = []
    xlsx = tmp_path / "fake.xlsx"
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["indicator", "value"])
    ws.append(["a", 1.5])
    wb.save(xlsx)
    monkeypatch.setattr(
        "morie.data.urlopen", lambda req, timeout=30: (calls.append(req.full_url), _Reply(xlsx.read_bytes()))[1]
    )
    df = mdata.fetch_cihi_indicator_library()
    assert list(df.columns) == ["indicator", "value"] and len(df) == 1
    assert calls == [mdata.CIHI_INDICATOR_LIBRARY_URL]
    mdata.fetch_cihi_indicator_library()
    assert len(calls) == 1


def test_dataset_route_names_every_kind():
    routes = {k: mdata.dataset_route(e) for k, e in mdata.DATASET_CATALOG.items()}
    assert routes["ocp21"] == "open.canada.ca"
    assert routes["otisb09"] == "data.ontario.ca"
    assert routes["siu"] == "rmoriedata (CRAN)"
    assert routes["cihidt"] == "CIHI"
    assert routes["naps-no2-on-2023"] == "ECCC NAPS"
    assert routes["mapq"].startswith("own file: ")
    assert all("route" in d for d in mdata.list_datasets(db_path=":memory:"))


def test_siu_index_links_use_the_current_report_url():
    html = (
        '<a href="/en/directors_report_details.php?drid=5289"><span>26-OCI-270</span></a>'
        '<a href="case_summary_details.php?id=7">16-OFI-019</a>'
        '<a href="/en/case_status.php">Case status</a>'
    )
    links = siu_fetch._extract_case_links(html)
    assert [c for c, _ in links] == ["26-OCI-270", "16-OFI-019"]
    assert links[0][1] == "https://www.siu.on.ca/en/directors_report_details.php?drid=5289"


NAPS_SAMPLE = (
    "﻿File generated on // Fichier créé sur, 2026-06-16,,,,\n"
    ",,,,\n"
    "Pollutant // Polluant:, NO2,,,,\n"
    "Units // Unités, ppb,,,,\n"
    "Note // Note,Data is hour ending local standard time,,,,\n"
    ",,,,\n"
    "Pollutant//Polluant,NAPS ID//Identifiant SNPA,City//Ville,Province/Territory//Province/Territoire,"
    "Latitude//Latitude,Longitude//Longitude,Date//Date,H01//H01,H02//H02,H03//H03\n"
    "NO2,10102,St. John's,NL,47.56038,-52.71147,2023-01-01,7,-999,1\n"
    "NO2,60101,Toronto,ON,43.7,-79.4,2023-01-01,12,13,\n"
)


def test_naps_hourly_csv_parses_to_long_rows_and_fetch_filters_province(monkeypatch, tmp_path):
    rows = earth.parse_naps_hourly_csv(NAPS_SAMPLE, "no2")
    assert [(r["station_id"], r["datetime_local"], r["value"]) for r in rows] == [
        ("10102", "2023-01-01 01:00", 7.0),
        ("10102", "2023-01-01 03:00", 1.0),
        ("60101", "2023-01-01 01:00", 12.0),
        ("60101", "2023-01-01 02:00", 13.0),
    ]
    assert rows[0]["unit"] == "ppb" and rows[0]["province"] == "NL"
    assert earth.naps_hourly_file_path(2023, "pm25").endswith(
        "/2023/ContinuousData-DonneesContinu/HourlyData-DonneesHoraires/PM25_2023.csv"
    )

    class FakeResp:
        def __init__(self, status, content=b"", payload=None):
            self.status_code, self.content, self._payload = status, content, payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, **kw):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            if url.endswith("/path_contents"):
                return FakeResp(200, payload={"path_contents": [{"name": "NO2_2023.csv"}]})
            assert "NO2_2023.csv" in url
            return FakeResp(200, content=NAPS_SAMPLE.encode("utf-8"))

    monkeypatch.setattr(earth, "earth_cache_dir", lambda: tmp_path)
    import morie.earth as e2

    monkeypatch.setattr(e2, "httpx", type("H", (), {"Client": FakeClient}), raising=False)
    rows = earth._naps_download_rows(2023, "no2", client_factory=FakeClient)
    assert len(rows) == 4


@pytest.mark.parametrize("code", [404, 500])
def test_ckan_resource_without_a_datastore_is_downloaded_as_a_file(monkeypatch, tmp_path, code):
    db = _isolate(monkeypatch, tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CSV/Data_donnees/CADS.csv", "PUMF_ID,AGE\n1,30\n2,41\n")
        zf.writestr("CSV/Data_donnees/CADS_bsw.csv", "PUMF_ID,BWGT1\n1,0.5\n2,1.5\n3,2.5\n")
        zf.writestr("CSV/Documentation/Guide/CADS2019_UgE.pdf", "%PDF")
    zip_bytes = buf.getvalue()
    calls = []

    def fake_urlopen(req, timeout=30):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if "datastore_search" in url:
            raise HTTPError(url, code, "datastore unavailable", {}, None)
        if "resource_show" in url:
            return _Reply(
                json.dumps(
                    {"result": {"url": "https://www150.statcan.gc.ca/n1/pub/13-25-0005/2021001/CSV.zip"}}
                ).encode()
            )
        assert url.endswith("CSV.zip")

        class R(_Reply):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                out, self._body = self._body[:n] if n > 0 else self._body, self._body[n:] if n > 0 else b""
                return out

        return R(zip_bytes)

    monkeypatch.setattr("morie.data.urlopen", fake_urlopen)
    df = mdata.load_dataset("cu20mf", db_path=db)
    assert list(df.columns) == ["PUMF_ID", "AGE"] and len(df) == 2
    bt = mdata.load_dataset("cu20bt", db_path=db)
    assert list(bt.columns) == ["PUMF_ID", "BWGT1"] and len(bt) == 3
    assert sum(u.endswith("CSV.zip") for u in calls) == 1  # one download serves both keys
    assert len(mdata.load_dataset("cu20mf", db_path=db)) == 2  # cached under the table name
