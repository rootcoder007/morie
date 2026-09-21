"""A2AJ Canadian Legal Data client.

Access to Algorithmic Justice (https://a2aj.ca) publishes Canadian
court and tribunal decisions plus federal and provincial statutes and
regulations through three open channels, documented at
https://github.com/a2aj-ca/canadian-legal-data:

* a REST API at https://api.a2aj.ca (coverage / search / fetch);
* Hugging Face datasets ``a2aj/canadian-case-law`` and
  ``a2aj/canadian-laws``, one Parquet file per court or law type;
* the same Parquet files by direct download.

Every text is an unofficial copy; each record carries an
``upstream_license`` field that may restrict commercial use.

Quick usage
-----------

  >>> from morie.ingest import a2aj
  >>> a2aj.coverage("cases")[["dataset", "number_of_documents"]]
  >>> hits = a2aj.search("roncarelli", search_type="name", dataset="SCC")
  >>> doc = a2aj.fetch("2023 SCC 17", end_char=400)
  >>> sct = a2aj.load("SCT", columns=["citation_en", "cases_cited_en"])
  >>> edges = a2aj.citation_edges(sct)

The R arm exposes the same surface as ``morie_ingest_a2aj_*``.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import httpx

from morie.fn import _frame_core as pd

API_URL = "https://api.a2aj.ca"
HF_URL = "https://huggingface.co/datasets/a2aj"
DEFAULT_USER_AGENT = "morie/python (+https://github.com/rootcoder007/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0

_ISSUE_HRTO = 4
_ISSUE_ALBERTA = 3
_ISSUE_NATIONAL = 1


class A2AJError(RuntimeError):
    """An A2AJ API call returned an HTTP error or a non-JSON body."""


def _get_json(endpoint: str, params: dict[str, Any] | None, *, timeout: float,
              user_agent: str, transport: httpx.BaseTransport | None = None) -> Any:
    clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    with httpx.Client(timeout=timeout, headers={"User-Agent": user_agent},
                      follow_redirects=True, transport=transport) as client:
        r = client.get(f"{API_URL}/{endpoint}", params=clean)
    if r.status_code >= 400:
        raise A2AJError(f"{endpoint} -> HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as exc:
        raise A2AJError(f"{endpoint} returned non-JSON: {r.text[:120]}") from exc


def _records_df(records: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Union of keys across records; list-valued fields stay lists."""
    if not records:
        return pd.DataFrame({})
    keys: list[str] = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    return pd.DataFrame({k: [r.get(k) for r in records] for k in keys})


def _as_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    return _dt.date.fromisoformat(str(value)[:10])


def coverage(doc_type: str = "cases", *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
             user_agent: str = DEFAULT_USER_AGENT,
             transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Every court, tribunal or law collection in the corpus.

    Returns a DataFrame with ``dataset``, ``description_en``,
    ``description_fr``, ``earliest_document_date``,
    ``latest_document_date`` (both ``datetime.date``) and
    ``number_of_documents``.
    """
    if doc_type not in ("cases", "laws"):
        raise ValueError("doc_type must be 'cases' or 'laws'")
    res = _get_json("coverage", {"doc_type": doc_type}, timeout=timeout,
                    user_agent=user_agent, transport=transport)
    rows = res.get("results") or []
    for r in rows:
        for k in ("earliest_document_date", "latest_document_date"):
            if k in r:
                r[k] = _as_date(r[k])
        if "number_of_documents" in r and r["number_of_documents"] is not None:
            r["number_of_documents"] = int(r["number_of_documents"])
    return _records_df(rows)


def search(query: str, *, search_type: str = "full_text", doc_type: str = "cases",
           size: int = 10, search_language: str = "en", sort_results: str = "default",
           dataset: str | list[str] | None = None, start_date: str | None = None,
           end_date: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
           user_agent: str = DEFAULT_USER_AGENT,
           transport: httpx.BaseTransport | None = None) -> pd.DataFrame:
    """Full-text or title search; at most 50 hits, no paging.

    ``dataset`` restricts to one or more dataset codes (``"SCC"`` or
    ``["SCC", "ONCA"]``); see :func:`coverage`.
    """
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if search_type not in ("full_text", "name"):
        raise ValueError("search_type must be 'full_text' or 'name'")
    if doc_type not in ("cases", "laws"):
        raise ValueError("doc_type must be 'cases' or 'laws'")
    if search_language not in ("en", "fr"):
        raise ValueError("search_language must be 'en' or 'fr'")
    if sort_results not in ("default", "newest_first", "oldest_first"):
        raise ValueError("sort_results must be 'default', 'newest_first' or 'oldest_first'")
    size = int(size)
    if not 1 <= size <= 50:
        raise ValueError("size must be between 1 and 50")
    if isinstance(dataset, (list, tuple)):
        dataset = ",".join(dataset)
    params = {
        "query": query, "search_type": search_type, "doc_type": doc_type,
        "size": size, "search_language": search_language,
        "sort_results": sort_results, "dataset": dataset,
        "start_date": start_date, "end_date": end_date,
    }
    res = _get_json("search", params, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    rows = res.get("results") or []
    for r in rows:
        if "score" in r and r["score"] is not None:
            r["score"] = float(r["score"])
    return _records_df(rows)


def fetch(citation: str, *, doc_type: str = "cases", output_language: str = "en",
          section: str | None = None, start_char: int = 0, end_char: int = -1,
          include_citations: bool = False, citations_limit: int = 100,
          citations_offset: int = 0, timeout: float = DEFAULT_TIMEOUT_SECONDS,
          user_agent: str = DEFAULT_USER_AGENT,
          transport: httpx.BaseTransport | None = None) -> dict[str, Any] | None:
    """One document by citation, as a dict; ``None`` when not in the corpus.

    Text is under ``unofficial_text_en`` / ``unofficial_text_fr``. With
    ``include_citations=True`` (cases only) ``cases_cited_*`` and
    ``cases_citing_*`` are lists. For a court the corpus does not cover
    see :func:`gaps` and :mod:`morie.ingest.canlii`.
    """
    if not isinstance(citation, str) or not citation:
        raise ValueError("citation must be a non-empty string")
    if doc_type not in ("cases", "laws"):
        raise ValueError("doc_type must be 'cases' or 'laws'")
    if output_language not in ("en", "fr", "both"):
        raise ValueError("output_language must be 'en', 'fr' or 'both'")
    params: dict[str, Any] = {
        "citation": citation, "doc_type": doc_type,
        "output_language": output_language,
        "section": section if doc_type == "laws" else None,
        "start_char": int(start_char), "end_char": int(end_char),
    }
    if doc_type == "cases" and include_citations:
        params.update({"include_citations": "true",
                       "citations_limit": int(citations_limit),
                       "citations_offset": int(citations_offset)})
    res = _get_json("fetch", params, timeout=timeout, user_agent=user_agent,
                    transport=transport)
    rows = res.get("results") or []
    if not rows:
        return None
    doc = dict(rows[0])
    if doc.get("citing_cases_count") is not None:
        doc["citing_cases_count"] = int(doc["citing_cases_count"])
    return doc


def parquet_url(dataset: str, doc_type: str = "cases") -> str:
    """Hugging Face download URL of one dataset's ``train.parquet``."""
    if doc_type not in ("cases", "laws"):
        raise ValueError("doc_type must be 'cases' or 'laws'")
    if not isinstance(dataset, str) or not dataset or any(
            c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in dataset):
        raise ValueError('dataset must be one upper-case code such as "SCC" or "LEGISLATION-FED"')
    repo = "canadian-case-law" if doc_type == "cases" else "canadian-laws"
    return f"{HF_URL}/{repo}/resolve/main/{dataset}/train.parquet"


def _default_cache_dir() -> Path:
    override = os.environ.get("MORIE_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "a2aj"
    from morie.data import _user_cache_dir
    return _user_cache_dir() / "a2aj"


def download(dataset: str, doc_type: str = "cases", *, cache_dir: str | Path | None = None,
             refresh: bool = False, timeout: float = 600.0,
             user_agent: str = DEFAULT_USER_AGENT,
             transport: httpx.BaseTransport | None = None) -> Path:
    """Fetch one dataset's Parquet file into the cache and return its path.

    Files range from a few megabytes (small tribunals) to several
    hundred (the Supreme Court of Canada, the BC Supreme Court).
    """
    url = parquet_url(dataset, doc_type)
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    dest = root / doc_type / f"{dataset}.parquet"
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".parquet.part")
    with httpx.Client(timeout=timeout, headers={"User-Agent": user_agent},
                      follow_redirects=True, transport=transport) as client, client.stream("GET", url) as r:
        if r.status_code >= 400:
            raise A2AJError(f"download of {url} failed: HTTP {r.status_code}")
        with open(part, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    os.replace(part, dest)
    return dest


def load(dataset: str, doc_type: str = "cases", *, columns: list[str] | None = None,
         cache_dir: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download (or reuse) one dataset and decode it with the native reader.

    ``unofficial_text_*`` holds the full texts and dominates the file;
    pass ``columns`` to skip it. ``cases_cited_*`` and
    ``cases_citing_*`` come back as lists per row.
    """
    path = download(dataset, doc_type, cache_dir=cache_dir, refresh=refresh)
    return pd.read_parquet(str(path), columns=columns)


def citation_edges(x: pd.DataFrame | dict[str, Any], language: str = "en") -> pd.DataFrame:
    """Edge list (``from``, ``to``) from the ``cases_cited_<language>`` field.

    Accepts a DataFrame from :func:`load` or the dict from :func:`fetch`
    with ``include_citations=True``. Order follows first appearance in
    each decision.
    """
    if language not in ("en", "fr"):
        raise ValueError("language must be 'en' or 'fr'")
    from_key, to_key = f"citation_{language}", f"cases_cited_{language}"
    if isinstance(x, dict):
        froms = [x.get(from_key)]
        cited = [x.get(to_key)]
    else:
        if from_key not in x.columns or to_key not in x.columns:
            raise KeyError(f"x needs columns {from_key} and {to_key}")
        froms = list(x[from_key])
        cited = list(x[to_key])
    src: list[str] = []
    dst: list[str] = []
    for f, tos in zip(froms, cited):
        if not tos:
            continue
        for t in tos:
            if t:
                src.append(f)
                dst.append(t)
    return pd.DataFrame({"from": src, "to": dst})


def cli(args: list[str], *, transport: httpx.BaseTransport | None = None) -> int:
    """``morie ingest a2aj {coverage,search,fetch} ...`` handler.

    Writes CSV to ``--out`` or stdout (``fetch`` writes the text itself).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="morie ingest a2aj",
                                description="A2AJ Canadian Legal Data (api.a2aj.ca)")
    sub = p.add_subparsers(dest="verb", required=True)
    cov = sub.add_parser("coverage", help="courts, tribunals and law collections in the corpus")
    cov.add_argument("--doc-type", choices=["cases", "laws"], default="cases")
    cov.add_argument("--out")
    srch = sub.add_parser("search", help="full-text or title search (at most 50 hits)")
    srch.add_argument("query")
    srch.add_argument("--search-type", choices=["full_text", "name"], default="full_text")
    srch.add_argument("--doc-type", choices=["cases", "laws"], default="cases")
    srch.add_argument("--size", type=int, default=10)
    srch.add_argument("--language", choices=["en", "fr"], default="en")
    srch.add_argument("--sort", choices=["default", "newest_first", "oldest_first"], default="default")
    srch.add_argument("--dataset", help="comma-separated dataset codes, e.g. SCC,ONCA")
    srch.add_argument("--start-date")
    srch.add_argument("--end-date")
    srch.add_argument("--out")
    ftch = sub.add_parser("fetch", help="one document's unofficial text by citation")
    ftch.add_argument("citation")
    ftch.add_argument("--doc-type", choices=["cases", "laws"], default="cases")
    ftch.add_argument("--language", choices=["en", "fr", "both"], default="en")
    ftch.add_argument("--section")
    ftch.add_argument("--start-char", type=int, default=0)
    ftch.add_argument("--end-char", type=int, default=-1)
    ftch.add_argument("--out")
    ns = p.parse_args(args)

    def _emit(df: pd.DataFrame, out: str | None) -> None:
        if out:
            df.to_csv(out, index=False)
            print(f"wrote {out}  ({len(df):,} rows)", file=sys.stderr)
        else:
            sys.stdout.write(df.to_csv(index=False))

    if ns.verb == "coverage":
        _emit(coverage(ns.doc_type, transport=transport), ns.out)
        return 0
    if ns.verb == "search":
        df = search(ns.query, search_type=ns.search_type, doc_type=ns.doc_type,
                    size=ns.size, search_language=ns.language, sort_results=ns.sort,
                    dataset=ns.dataset, start_date=ns.start_date, end_date=ns.end_date,
                    transport=transport)
        _emit(df, ns.out)
        return 0
    doc = fetch(ns.citation, doc_type=ns.doc_type, output_language=ns.language,
                section=ns.section, start_char=ns.start_char, end_char=ns.end_char,
                transport=transport)
    if doc is None:
        print(f"{ns.citation!r} is not in the A2AJ corpus; see morie.ingest.a2aj.gaps()",
              file=sys.stderr)
        return 1
    text = "\n\n".join(str(doc[k]) for k in ("unofficial_text_en", "unofficial_text_fr")
                       if doc.get(k))
    if ns.out:
        Path(ns.out).write_text(text, encoding="utf-8")
        print(f"wrote {ns.out}  ({len(text):,} chars)", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")
    return 0


def gaps() -> pd.DataFrame:
    """Courts and tribunals the A2AJ corpus does not carry.

    Maps each gap named in the project's open issues (HRTO, BCHRT and
    ONSC in #4, Alberta in #3, Quebec and the remaining provinces in
    #1) to the CanLII database id used on canlii.org, for
    :mod:`morie.ingest.canlii`.
    """
    rows = [
        ("HRTO", "Human Rights Tribunal of Ontario", "ON", "onhrt", _ISSUE_HRTO),
        ("BCHRT", "British Columbia Human Rights Tribunal", "BC", "bchrt", _ISSUE_HRTO),
        ("ONSC", "Ontario Superior Court of Justice", "ON", "onsc", _ISSUE_HRTO),
        ("ONCJ", "Ontario Court of Justice", "ON", "oncj", None),
        ("ABCA", "Court of Appeal of Alberta", "AB", "abca", _ISSUE_ALBERTA),
        ("ABKB", "Court of King's Bench of Alberta", "AB", "abkb", _ISSUE_ALBERTA),
        ("ABCJ", "Alberta Court of Justice", "AB", "abcj", _ISSUE_ALBERTA),
        ("QCCA", "Court of Appeal of Quebec", "QC", "qcca", _ISSUE_NATIONAL),
        ("QCCS", "Superior Court of Quebec", "QC", "qccs", _ISSUE_NATIONAL),
        ("QCCQ", "Court of Quebec", "QC", "qccq", _ISSUE_NATIONAL),
        ("MBCA", "Court of Appeal of Manitoba", "MB", "mbca", _ISSUE_NATIONAL),
        ("MBKB", "Court of King's Bench of Manitoba", "MB", "mbkb", _ISSUE_NATIONAL),
        ("SKCA", "Court of Appeal for Saskatchewan", "SK", "skca", _ISSUE_NATIONAL),
        ("SKKB", "Court of King's Bench for Saskatchewan", "SK", "skkb", _ISSUE_NATIONAL),
        ("NBCA", "Court of Appeal of New Brunswick", "NB", "nbca", _ISSUE_NATIONAL),
        ("NBKB", "Court of King's Bench of New Brunswick", "NB", "nbkb", _ISSUE_NATIONAL),
        ("NLCA", "Court of Appeal of Newfoundland and Labrador", "NL", "nlca", _ISSUE_NATIONAL),
        ("NLSC", "Supreme Court of Newfoundland and Labrador", "NL", "nlsc", _ISSUE_NATIONAL),
        ("PECA", "Prince Edward Island Court of Appeal", "PE", "peca", _ISSUE_NATIONAL),
        ("PESC", "Supreme Court of Prince Edward Island", "PE", "pesctd", _ISSUE_NATIONAL),
        ("NTCA", "Court of Appeal for the Northwest Territories", "NT", "ntca", _ISSUE_NATIONAL),
        ("NTSC", "Supreme Court of the Northwest Territories", "NT", "ntsc", _ISSUE_NATIONAL),
        ("NUCJ", "Nunavut Court of Justice", "NU", "nucj", _ISSUE_NATIONAL),
        ("YKSC", "Supreme Court of Yukon", "YT", "yksc", _ISSUE_NATIONAL),
    ]
    return pd.DataFrame({
        "code": [r[0] for r in rows],
        "name": [r[1] for r in rows],
        "jurisdiction": [r[2] for r in rows],
        "canlii_database_id": [r[3] for r in rows],
        "issue": [r[4] for r in rows],
    })
