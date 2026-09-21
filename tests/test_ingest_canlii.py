# SPDX-License-Identifier: AGPL-3.0-or-later
"""CanLII API client: id derivation, parameter validation and a mocked API (no network)."""

import json

import httpx
import pytest

from morie.ingest import canlii


def _transport(handler):
    return httpx.MockTransport(handler)


def _json(payload, status=200):
    return httpx.Response(status, content=json.dumps(payload).encode(),
                          headers={"content-type": "application/json"})


def test_case_id_from_neutral_citation():
    assert canlii.case_id("2007 BCSC 1700") == {
        "citation": "2007 BCSC 1700", "database_id": "bcsc", "case_id": "2007bcsc1700"}
    assert canlii.case_id("2008 SCC 9")["database_id"] == "csc-scc"
    assert canlii.case_id("2008 SCC 9")["case_id"] == "2008scc9"
    assert canlii.case_id("[1959] SCR 121") == {
        "citation": "[1959] SCR 121", "database_id": None, "case_id": None}


def test_key_resolution(monkeypatch):
    monkeypatch.delenv("CANLII_API_KEY", raising=False)
    with pytest.raises(canlii.CanLIIError, match="CANLII_API_KEY"):
        canlii.databases(transport=_transport(lambda r: _json({})))
    monkeypatch.setenv("CANLII_API_KEY", "envkey")
    seen = {}

    def handler(request):
        seen["key"] = request.url.params["api_key"]
        return _json({"caseDatabases": []})

    canlii.databases(transport=_transport(handler))
    assert seen["key"] == "envkey"
    canlii.databases(api_key="argkey", transport=_transport(handler))
    assert seen["key"] == "argkey"


def test_cases_query_and_language_keyed_ids():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url).split("?")[0]
        seen["params"] = dict(request.url.params)
        return _json({"cases": [
            {"databaseId": "onhrt", "caseId": {"en": "2024hrto1"},
             "title": "A v. B", "citation": "2024 HRTO 1"},
        ]})

    df = canlii.cases("onhrt", result_count=200, decision_date_after="2024-01-01",
                      api_key="k", transport=_transport(handler))
    assert seen["url"] == "https://api.canlii.org/v1/caseBrowse/en/onhrt/"
    assert seen["params"] == {"api_key": "k", "offset": "0", "resultCount": "200",
                              "decisionDateAfter": "2024-01-01"}
    assert list(df["caseId"]) == ["2024hrto1"]
    for bad in (dict(result_count=0), dict(result_count=10001),
                dict(decision_date_after="2024/01/01"), dict(language="de")):
        with pytest.raises(ValueError):
            canlii.cases("onhrt", api_key="k", transport=_transport(handler), **bad)
    with pytest.raises(ValueError):
        canlii.cases("ON HRT", api_key="k", transport=_transport(handler))


def test_case_and_citator_paths():
    seen = []

    def handler(request):
        seen.append(str(request.url).split("?")[0])
        if "caseCitator" in seen[-1]:
            return _json({"citingCases": [
                {"databaseId": "csc-scc", "caseId": "2010scc1", "title": "T", "citation": "2010 SCC 1"}]})
        return _json({"databaseId": "bcsc", "caseId": "2007bcsc1700", "url": "https://canlii.ca/t/x",
                      "title": "Doe v. Roe", "citation": "2007 BCSC 1700",
                      "decisionDate": "2007-11-20", "keywords": "contract"})

    meta = canlii.case("bcsc", "2007bcsc1700", api_key="k", transport=_transport(handler))
    assert meta["title"] == "Doe v. Roe" and meta["decisionDate"] == "2007-11-20"
    cit = canlii.citator("csc-scc", "2008scc9", "citingCases", api_key="k",
                         transport=_transport(handler))
    assert list(cit["citation"]) == ["2010 SCC 1"]
    assert seen == ["https://api.canlii.org/v1/caseBrowse/en/bcsc/2007bcsc1700/",
                    "https://api.canlii.org/v1/caseCitator/en/csc-scc/2008scc9/citingCases"]
    with pytest.raises(ValueError):
        canlii.citator("bcsc", "x", "citedThings", api_key="k", transport=_transport(handler))


def test_legislation_paths():
    seen = []

    def handler(request):
        seen.append(str(request.url).split("?")[0])
        if seen[-1].endswith("/legislationBrowse/en/"):
            return _json({"legislationDatabases": [
                {"databaseId": "cas", "type": "STATUTE", "jurisdiction": "ca", "name": "Federal"}]})
        if seen[-1].endswith("/cas/"):
            return _json({"legislations": [
                {"databaseId": "cas", "legislationId": "rsc-1985-c-c-46",
                 "title": "Criminal Code", "citation": "RSC 1985, c C-46", "type": "STATUTE"}]})
        return _json({"legislationId": "rsc-1985-c-c-46", "title": "Criminal Code",
                      "repealed": "NO", "content": []})

    assert list(canlii.legislation_databases(api_key="k", transport=_transport(handler))["databaseId"]) == ["cas"]
    assert list(canlii.legislations("cas", api_key="k", transport=_transport(handler))["title"]) == ["Criminal Code"]
    one = canlii.legislation("cas", "rsc-1985-c-c-46", api_key="k", transport=_transport(handler))
    assert one["repealed"] == "NO"


def test_http_errors_carry_the_api_message():
    def handler(request):
        return httpx.Response(403, content=b'{"message": "invalid api key"}')

    with pytest.raises(canlii.CanLIIError, match="HTTP 403: invalid api key"):
        canlii.databases(api_key="bad", transport=_transport(handler))
