"""CanLII REST API client.

The Canadian Legal Information Institute (https://www.canlii.org)
publishes every Canadian court and most tribunals, which makes it the
route to the decisions the A2AJ corpus does not yet carry (see
:func:`morie.ingest.a2aj.gaps`): the Human Rights Tribunal of Ontario,
the BC Human Rights Tribunal, the Ontario Superior Court, Alberta,
Quebec and the remaining provinces. Its API (https://api.canlii.org/v1)
returns metadata and the citation network, not full text; every record
carries the canlii.org URL of the decision. A key is free on request
from CanLII and is read from the ``CANLII_API_KEY`` environment
variable when not passed explicitly.

Quick usage
-----------

  >>> from morie.ingest import canlii
  >>> canlii.databases()                       # every court and tribunal
  >>> canlii.cases("onhrt", result_count=200, decision_date_after="2024-01-01")
  >>> ids = canlii.case_id("2007 BCSC 1700")   # {'database_id': 'bcsc', 'case_id': '2007bcsc1700'}
  >>> canlii.case(ids["database_id"], ids["case_id"])
  >>> canlii.citator("csc-scc", "2008scc9", "citingCases")

The R arm exposes the same surface as ``morie_ingest_canlii_*``.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from morie.fn import _frame_core as pd

API_URL = "https://api.canlii.org/v1"
DEFAULT_USER_AGENT = "morie/python (+https://github.com/rootcoder007/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_RESULT_COUNT = 10000

_ID_RE = re.compile(r"^[a-z0-9-]+$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_NEUTRAL_RE = re.compile(r"^\s*([0-9]{4})\s+([A-Za-z]+)\s+([0-9]+)\s*$")


class CanLIIError(RuntimeError):
    """A CanLII API call returned an HTTP error or a non-JSON body."""


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("CANLII_API_KEY", "")
    if not key:
        raise CanLIIError(
            "A CanLII API key is required: pass api_key= or set the CANLII_API_KEY "
            "environment variable. Keys are free on request from "
            "https://www.canlii.org/en/feedback/feedback.html")
    return key


def _check_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f'{what} must be one lower-case CanLII identifier such as "bcsc" or "2007bcsc1700"')
    return value


def _check_lang(language: str) -> str:
    if language not in ("en", "fr"):
        raise ValueError("language must be 'en' or 'fr'")
    return language


def _date_params(**kwargs: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        v = str(v)
        if not _DATE_RE.match(v):
            raise ValueError(f'{k} must be one "YYYY-MM-DD" string')
        out[k] = v
    return out


def _get_json(path: str, params: dict[str, Any] | None = None, *, api_key: str | None,
              timeout: float, user_agent: str,
              transport: httpx.BaseTransport | None = None) -> Any:
    query = {"api_key": _resolve_key(api_key)}
    query.update({k: v for k, v in (params or {}).items() if v is not None})
    with httpx.Client(timeout=timeout, headers={"User-Agent": user_agent},
                      follow_redirects=True, transport=transport) as client:
        r = client.get(f"{API_URL}/{path}", params=query)
    try:
        parsed = r.json()
    except ValueError:
        parsed = None
    if r.status_code >= 400:
        detail = r.text[:200]
        if isinstance(parsed, dict):
            detail = parsed.get("message") or parsed.get("error") or detail
        raise CanLIIError(f"{path} -> HTTP {r.status_code}: {detail}")
    if parsed is None:
        raise CanLIIError(f"{path} returned non-JSON: {r.text[:120]}")
    return parsed


def _records_df(records: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Language-keyed ids (``{"en": "2008scc9"}``) collapse to the string."""
    if not records:
        return pd.DataFrame({})
    rows = []
    for r in records:
        r = dict(r)
        for k in ("caseId", "legislationId", "databaseId"):
            v = r.get(k)
            if isinstance(v, dict) and v:
                r[k] = str(next(iter(v.values())))
        rows.append(r)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return pd.DataFrame({k: [r.get(k) for r in rows] for k in keys})


def databases(language: str = "en", *, api_key: str | None = None,
              timeout: float = DEFAULT_TIMEOUT_SECONDS, user_agent: str = DEFAULT_USER_AGENT,
              transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Every court and tribunal: ``databaseId``, ``jurisdiction``, ``name``."""
    res = _get_json(f"caseBrowse/{_check_lang(language)}/", api_key=api_key,
                    timeout=timeout, user_agent=user_agent, transport=transport)
    return _records_df(res.get("caseDatabases"))


def cases(database_id: str, *, offset: int = 0, result_count: int = 100,
          published_before: str | None = None, published_after: str | None = None,
          modified_before: str | None = None, modified_after: str | None = None,
          changed_before: str | None = None, changed_after: str | None = None,
          decision_date_before: str | None = None, decision_date_after: str | None = None,
          language: str = "en", api_key: str | None = None,
          timeout: float = DEFAULT_TIMEOUT_SECONDS, user_agent: str = DEFAULT_USER_AGENT,
          transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Decisions of one database, newest first; at most 10,000 per call.

    Returns ``databaseId``, ``caseId``, ``title`` and ``citation``.
    """
    _check_id(database_id, "database_id")
    result_count = int(result_count)
    if not 1 <= result_count <= MAX_RESULT_COUNT:
        raise ValueError(f"result_count must be between 1 and {MAX_RESULT_COUNT}")
    params: dict[str, Any] = {"offset": int(offset), "resultCount": result_count}
    params.update(_date_params(
        publishedBefore=published_before, publishedAfter=published_after,
        modifiedBefore=modified_before, modifiedAfter=modified_after,
        changedBefore=changed_before, changedAfter=changed_after,
        decisionDateBefore=decision_date_before, decisionDateAfter=decision_date_after))
    res = _get_json(f"caseBrowse/{_check_lang(language)}/{database_id}/", params,
                    api_key=api_key, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    return _records_df(res.get("cases"))


def case(database_id: str, case_id: str, *, language: str = "en",
         api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
         user_agent: str = DEFAULT_USER_AGENT,
         transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Metadata of one decision (``url``, ``title``, ``citation``, ``decisionDate``, ...)."""
    _check_id(database_id, "database_id")
    _check_id(case_id, "case_id")
    res = _get_json(f"caseBrowse/{_check_lang(language)}/{database_id}/{case_id}/",
                    api_key=api_key, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    df = _records_df([res])
    return {k: df[k][0] for k in df.columns}


def citator(database_id: str, case_id: str, metadata_type: str = "citedCases", *,
            api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
            user_agent: str = DEFAULT_USER_AGENT,
            transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Cited cases, citing cases or cited legislation of one decision (English only)."""
    if metadata_type not in ("citedCases", "citingCases", "citedLegislations"):
        raise ValueError("metadata_type must be 'citedCases', 'citingCases' or 'citedLegislations'")
    _check_id(database_id, "database_id")
    _check_id(case_id, "case_id")
    res = _get_json(f"caseCitator/en/{database_id}/{case_id}/{metadata_type}",
                    api_key=api_key, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    return _records_df(res.get(metadata_type))


def legislation_databases(language: str = "en", *, api_key: str | None = None,
                          timeout: float = DEFAULT_TIMEOUT_SECONDS,
                          user_agent: str = DEFAULT_USER_AGENT,
                          transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Legislation databases: ``databaseId``, ``type``, ``jurisdiction``, ``name``."""
    res = _get_json(f"legislationBrowse/{_check_lang(language)}/", api_key=api_key,
                    timeout=timeout, user_agent=user_agent, transport=transport)
    return _records_df(res.get("legislationDatabases"))


def legislations(database_id: str, *, language: str = "en", api_key: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, user_agent: str = DEFAULT_USER_AGENT,
                 transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Statutes or regulations of one legislation database."""
    _check_id(database_id, "database_id")
    res = _get_json(f"legislationBrowse/{_check_lang(language)}/{database_id}/",
                    api_key=api_key, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    return _records_df(res.get("legislations"))


def legislation(database_id: str, legislation_id: str, *, language: str = "en",
                api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                user_agent: str = DEFAULT_USER_AGENT,
                transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Metadata of one statute or regulation."""
    _check_id(database_id, "database_id")
    _check_id(legislation_id, "legislation_id")
    res = _get_json(
        f"legislationBrowse/{_check_lang(language)}/{database_id}/{legislation_id}/",
        api_key=api_key, timeout=timeout, user_agent=user_agent, transport=transport)
    df = _records_df([res])
    return {k: df[k][0] for k in df.columns}


def case_id(citation: str) -> dict[str, str | None]:
    """CanLII ``database_id`` and ``case_id`` from a neutral citation.

    ``"2007 BCSC 1700"`` names its court (``BCSC``) and CanLII's case id
    is the citation in lower case without spaces (``"2007bcsc1700"``).
    The Supreme Court of Canada is the one court whose database id
    differs from its citation code (``"csc-scc"``). A reporter-style
    citation (``"[1959] SCR 121"``) has no derivable id: both are None.
    """
    m = _NEUTRAL_RE.match(str(citation))
    if not m:
        return {"citation": citation, "database_id": None, "case_id": None}
    year, court, number = m.groups()
    db = court.lower()
    if db == "scc":
        db = "csc-scc"
    return {"citation": citation, "database_id": db,
            "case_id": f"{year}{court.lower()}{number}"}
