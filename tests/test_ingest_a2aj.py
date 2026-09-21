# SPDX-License-Identifier: AGPL-3.0-or-later
"""A2AJ Canadian Legal Data client: pure helpers and a mocked API (no network)."""

import datetime as dt
import json

import httpx
import pytest

from morie.ingest import a2aj


def _transport(handler):
    return httpx.MockTransport(handler)


def _json(payload, status=200):
    return httpx.Response(status, content=json.dumps(payload).encode(),
                          headers={"content-type": "application/json"})


def test_parquet_url_shapes():
    assert a2aj.parquet_url("SCC") == (
        "https://huggingface.co/datasets/a2aj/canadian-case-law/resolve/main/SCC/train.parquet")
    assert a2aj.parquet_url("REGULATIONS-ON", "laws").endswith(
        "canadian-laws/resolve/main/REGULATIONS-ON/train.parquet")
    with pytest.raises(ValueError):
        a2aj.parquet_url("scc")
    with pytest.raises(ValueError):
        a2aj.parquet_url("SCC", "statutes")


def test_records_df_unions_keys_and_keeps_lists():
    df = a2aj._records_df([
        {"citation_en": "2020 SCC 1", "cases_cited_en": ["2019 SCC 9"]},
        {"citation_en": "2021 ONCA 2", "score": "0.5"},
    ])
    assert list(df.columns) == ["citation_en", "cases_cited_en", "score"]
    assert list(df["cases_cited_en"]) == [["2019 SCC 9"], None]
    assert list(df["score"]) == [None, "0.5"]
    assert len(a2aj._records_df([])) == 0


def test_coverage_parses_dates_and_counts():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _json({"results": [
            {"dataset": "SCC", "description_en": "Supreme Court of Canada",
             "description_fr": "Cour supreme du Canada",
             "earliest_document_date": "1877-01-15",
             "latest_document_date": "2026-09-18", "number_of_documents": 10893},
        ]})

    df = a2aj.coverage("cases", transport=_transport(handler))
    assert seen["url"] == "https://api.a2aj.ca/coverage?doc_type=cases"
    assert df["earliest_document_date"][0] == dt.date(1877, 1, 15)
    assert df["number_of_documents"][0] == 10893
    with pytest.raises(ValueError):
        a2aj.coverage("statutes")


def test_search_builds_query_and_validates():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return _json({"results": [
            {"dataset": "SCC", "citation_en": "[1959] SCR 121",
             "name_en": "Roncarelli v. Duplessis", "score": "0.0043"},
        ]})

    df = a2aj.search("roncarelli", search_type="name", dataset=["SCC", "ONCA"],
                     size=5, start_date="1900-01-01", transport=_transport(handler))
    assert seen["params"] == {
        "query": "roncarelli", "search_type": "name", "doc_type": "cases",
        "size": "5", "search_language": "en", "sort_results": "default",
        "dataset": "SCC,ONCA", "start_date": "1900-01-01",
    }
    assert df["score"][0] == pytest.approx(0.0043)
    assert df["name_en"][0] == "Roncarelli v. Duplessis"
    for bad in (dict(size=0), dict(size=51), dict(search_type="title"),
                dict(sort_results="random"), dict(search_language="de")):
        with pytest.raises(ValueError):
            a2aj.search("x", transport=_transport(handler), **bad)
    with pytest.raises(ValueError):
        a2aj.search("", transport=_transport(handler))


def test_fetch_returns_dict_or_none():
    def handler(request):
        params = dict(request.url.params)
        if params["citation"] == "2007 BCSC 1700":
            return _json({})
        assert params["include_citations"] == "true"
        assert params["citations_limit"] == "50"
        return _json({"results": [
            {"dataset": "SCC", "citation_en": "2023 SCC 17",
             "unofficial_text_en": "Canadian Council for Refugees v. Canada",
             "cases_cited_en": ["2020 SCC 5", "2019 FCA 1"],
             "citing_cases_count": "59"},
        ]})

    doc = a2aj.fetch("2023 SCC 17", include_citations=True, citations_limit=50,
                     transport=_transport(handler))
    assert doc["citing_cases_count"] == 59
    assert doc["cases_cited_en"] == ["2020 SCC 5", "2019 FCA 1"]
    assert a2aj.fetch("2007 BCSC 1700", transport=_transport(handler)) is None
    with pytest.raises(ValueError):
        a2aj.fetch("2023 SCC 17", output_language="de", transport=_transport(handler))


def test_http_error_and_non_json_raise():
    def bad_status(request):
        return httpx.Response(422, content=b'{"detail": "size too large"}')

    def not_json(request):
        return httpx.Response(200, content=b"<html>maintenance</html>")

    with pytest.raises(a2aj.A2AJError, match="HTTP 422"):
        a2aj.coverage(transport=_transport(bad_status))
    with pytest.raises(a2aj.A2AJError, match="non-JSON"):
        a2aj.coverage(transport=_transport(not_json))


def test_download_streams_to_cache(tmp_path):
    body = b"PAR1" + b"\x00" * 16 + b"PAR1"
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=body)

    p = a2aj.download("SCT", cache_dir=tmp_path, transport=_transport(handler))
    assert p == tmp_path / "cases" / "SCT.parquet"
    assert p.read_bytes() == body
    assert calls == [a2aj.parquet_url("SCT")]
    a2aj.download("SCT", cache_dir=tmp_path, transport=_transport(handler))
    assert len(calls) == 1, "cached file is reused"
    a2aj.download("SCT", cache_dir=tmp_path, refresh=True, transport=_transport(handler))
    assert len(calls) == 2


def test_citation_edges_from_frame_and_dict():
    from morie.fn import _frame_core as pd
    df = pd.DataFrame({
        "citation_en": ["2020 SCC 1", "2021 ONCA 2", "2022 FC 3"],
        "cases_cited_en": [["2019 SCC 9", "2018 FCA 3"], [], None],
    })
    edges = a2aj.citation_edges(df)
    assert list(edges["from"]) == ["2020 SCC 1", "2020 SCC 1"]
    assert list(edges["to"]) == ["2019 SCC 9", "2018 FCA 3"]
    one = a2aj.citation_edges({"citation_en": "2023 SCC 17",
                               "cases_cited_en": ["2020 SCC 5"]})
    assert list(one["to"]) == ["2020 SCC 5"]
    with pytest.raises(KeyError):
        a2aj.citation_edges(pd.DataFrame({"citation_en": ["x"]}))
    with pytest.raises(ValueError):
        a2aj.citation_edges(df, language="de")


def test_gaps_names_the_open_issues():
    g = a2aj.gaps()
    assert len(g) == 24
    by_code = dict(zip(g["code"], g["canlii_database_id"]))
    assert by_code["HRTO"] == "onhrt"
    assert by_code["BCHRT"] == "bchrt"
    assert by_code["ONSC"] == "onsc"
    assert by_code["QCCS"] == "qccs"
    issues = dict(zip(g["code"], g["issue"]))
    assert issues["HRTO"] == 4 and issues["ABKB"] == 3 and issues["QCCA"] == 1
    assert len(set(g["canlii_database_id"])) == len(g)


def test_cli_search_writes_csv(capsys):
    def handler(request):
        return _json({"results": [{"dataset": "SCC", "citation_en": "2023 SCC 17",
                                   "name_en": "X v. Y"}]})

    rc = a2aj.cli(["search", "refugee", "--dataset", "SCC", "--size", "1"],
                  transport=_transport(handler))
    out = capsys.readouterr().out
    assert rc == 0
    assert "2023 SCC 17" in out and out.startswith("dataset,")
