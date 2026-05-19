# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-demand scraper for the Ontario Special Investigations Unit (SIU).

The SIU (Ontario's police-oversight agency) publishes Director's Reports
at https://www.siu.on.ca/en/directors_reports.php. This module mines
that catalogue on demand and caches the result as a single CSV in the
morie cache directory.

This is the *Ontario* Special Investigations Unit. It is unrelated to
the *federal* Structured Intervention Units (a corrections
segregation-replacement programme; see ``morie.siuiap`` and the
``morie.fn.siu*`` modules) and to OTIS (Ontario carceral placements).

Site mechanics (verified 2026-05): the index page is incremental --
the first ~24 rows are inline and the rest load by AJAX from
``/ssi/get_more_drs.php?lang=en&lastCount=N`` (15 rows per call). Each
row gives a numeric ``drid``, the case number, and the report signing
date; the full report is an HTML page at
``/en/directors_report_details.php?drid=N``. There is no JSON API.

Distribution policy (2026-05): the scraped corpus is NOT shipped with
the package because the legal status of redistributing aggregated
copies of publicly-posted oversight reports is unsettled. Each user
runs the scraper themselves, which is unambiguously fair use.

The scraper is conservative: a 2-second delay between requests and a
descriptive User-Agent. Run ``fetch_siu_cases(years=...)`` to populate
the cache, then ``morie_load_dataset("siu")`` returns a tidy data.frame.
"""

from __future__ import annotations

import csv
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


__all__ = [
    "SIU_INDEX_URL",
    "SIU_AJAX_URL",
    "SIU_DETAIL_URL",
    "fetch_siu_cases",
    "fetch_siu_dataframe",
    "siu_cache_path",
]


SIU_BASE = "https://www.siu.on.ca"
SIU_INDEX_URL = "https://www.siu.on.ca/en/directors_reports.php"
SIU_AJAX_URL = "https://www.siu.on.ca/ssi/get_more_drs.php"
SIU_DETAIL_URL = "https://www.siu.on.ca/en/directors_report_details.php"
USER_AGENT = "morie/0.9.5 (+https://github.com/hadesllm/morie)"
RATE_LIMIT_SECONDS = 2.0
_INDEX_PAGE_SIZE = 15  # rows returned per get_more_drs.php call


def _http_get(url: str, *, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _extract_index_rows(html: str) -> list[dict]:
    """Parse <tr class="dr-item"> fragments into index-row dicts.

    Returns dicts with keys drid, case_number, date_signed, url.
    """
    rows: list[dict] = []
    for block in re.findall(r'<tr class="dr-item".*?</tr>', html, re.S):
        m_drid = re.search(r'id="(\d+)"', block)
        m_case = re.search(r"<nobr>([^<]+)</nobr>", block)
        m_date = re.search(
            r"<nobr>[^<]+</nobr>\s*</td>\s*<td[^>]*>([^<]+)</td>", block)
        m_href = re.search(r'href="([^"]+)"', block)
        if not (m_drid and m_case and m_href):
            continue
        rows.append({
            "drid": int(m_drid.group(1)),
            "case_number": m_case.group(1).strip(),
            "date_signed": m_date.group(1).strip() if m_date else "",
            "url": urllib.parse.urljoin(SIU_BASE, m_href.group(1)),
        })
    return rows


def _iter_index(
    *, lang: str = "en", max_cases: Optional[int] = None,
    progress: bool = False,
) -> list[dict]:
    """Walk the SIU AJAX index endpoint, collecting every report row."""
    collected: list[dict] = []
    last = 0
    while True:
        url = f"{SIU_AJAX_URL}?lang={lang}&lastCount={last}"
        if progress:
            print(f"[siu] index: lastCount={last}")
        try:
            html = _http_get(url)
        except Exception as e:  # noqa: BLE001 - network best-effort
            if progress:
                print(f"[siu] index fetch failed: {e}")
            break
        chunk = _extract_index_rows(html)
        if not chunk:
            break
        collected.extend(chunk)
        last += len(chunk)
        if max_cases is not None and len(collected) >= max_cases:
            break
        if len(chunk) < _INDEX_PAGE_SIZE:  # short page -> end of catalogue
            break
        time.sleep(RATE_LIMIT_SECONDS)
    return collected


def _case_year(case_number: str) -> Optional[int]:
    """Year encoded in a SIU case number, e.g. '26-TCI-052' -> 2026."""
    m = re.match(r"(\d{2})-", case_number.strip())
    return 2000 + int(m.group(1)) if m else None


def _incident_type(case_number: str) -> str:
    """Middle code of a SIU case number, e.g. '26-TCI-052' -> 'TCI'."""
    parts = case_number.split("-")
    return parts[1].upper() if len(parts) >= 2 else ""


_DATE_FIELDS = {
    "incident_iso": re.compile(
        r"(?:Incident|incident occurred on)\s*[:\-]?\s*"
        r"([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
    "notification_iso": re.compile(
        r"(?:Notification|SIU was notified(?: of the incident)? on)\s*"
        r"[:\-]?\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
    "decision_iso": re.compile(
        r"(?:Director'?s? [Dd]ecision)\s*[:\-]?\s*"
        r"([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
}

# Greedy on the leading proper-noun run (bounded) so a name like
# "Niagara Regional Police Service" is captured whole, not truncated
# to its tail "Regional Police Service".
_SERVICE_FIELD = re.compile(
    r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,4} "
    r"Police(?: Service)?)")
_DECISION_FIELD = re.compile(
    r"(no reasonable grounds|reasonable grounds(?: to believe)?|"
    r"charge\(s\)? (?:was|were) (?:laid|withdrawn)|charges? were laid)",
    re.I)


def _parse_case_page(html: str, row: dict) -> dict:
    """Best-effort parsing of a SIU report page into a flat dict.

    The report pages are narrative prose, so the extracted incident /
    notification dates and police service are best-effort and may be
    blank when a report phrases things unusually.
    """
    record = {
        "drid": row["drid"],
        "case_number": row["case_number"],
        "incident_type": _incident_type(row["case_number"]),
        "report_signed_iso": _to_iso(row.get("date_signed", "")),
        "source_url": row["url"],
    }
    for key, pat in _DATE_FIELDS.items():
        m = pat.search(html)
        record[key] = _to_iso(m.group(1)) if m else ""
    m = _SERVICE_FIELD.search(html)
    record["police_service"] = m.group(1).strip() if m else ""
    m = _DECISION_FIELD.search(html)
    record["director_decision_text"] = m.group(0).strip() if m else ""
    return record


_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"],
    start=1)}


def _to_iso(date_str: str) -> str:
    m = re.match(r"([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", date_str.strip())
    if not m:
        return ""
    month = _MONTHS.get(m.group(1))
    if not month:
        return ""
    return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"


def siu_cache_path(cache_dir: str | Path = "~/.cache/morie/siu") -> Path:
    p = Path(cache_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p / "SIU.csv"


_CSV_FIELDS = [
    "drid", "case_number", "incident_type", "police_service",
    "incident_iso", "notification_iso", "decision_iso",
    "report_signed_iso", "director_decision_text", "source_url",
]


def fetch_siu_cases(
    *,
    years: Optional[Iterable[int]] = None,
    cache_dir: str | Path = "~/.cache/morie/siu",
    overwrite: bool = False,
    progress: bool = True,
    max_cases: Optional[int] = None,
) -> Path:
    """Scrape SIU Director's Reports into a single CSV.

    Args:
        years: Iterable of calendar years to keep (matched against the
            year encoded in each case number). ``None`` keeps all years.
        cache_dir: Directory for the cache CSV.
        overwrite: If False and SIU.csv already exists, return it.
        progress: Print scrape progress when True.
        max_cases: Optional cap on the number of reports fetched
            (useful for a quick smoke test).

    Returns:
        Path to the populated SIU.csv.

    Raises:
        RuntimeError if zero cases are scraped (a signal that the SIU
        site layout has changed and this module needs updating).
    """
    out_path = siu_cache_path(cache_dir)
    if out_path.is_file() and not overwrite:
        return out_path

    index_rows = _iter_index(max_cases=max_cases, progress=progress)
    if years is not None:
        wanted = {int(y) for y in years}
        index_rows = [r for r in index_rows
                      if _case_year(r["case_number"]) in wanted]
    if max_cases is not None:
        index_rows = index_rows[:max_cases]

    # Deduplicate on the detail-page URL.
    seen: set[str] = set()
    unique_rows = []
    for r in index_rows:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique_rows.append(r)

    records: list[dict] = []
    for i, row in enumerate(unique_rows, 1):
        if progress and i % 25 == 0:
            print(f"[siu] case {i}/{len(unique_rows)}")
        try:
            html = _http_get(row["url"])
        except Exception:  # noqa: BLE001 - skip an unreachable report
            continue
        records.append(_parse_case_page(html, row))
        time.sleep(RATE_LIMIT_SECONDS)

    if not records:
        raise RuntimeError(
            "Scraped 0 SIU cases. The site layout may have changed; "
            "verify SIU_AJAX_URL / SIU_DETAIL_URL and the regexes in "
            "siu_fetch.py against https://www.siu.on.ca/en/directors_reports.php"
        )

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    if progress:
        print(f"[siu] wrote {len(records)} cases to {out_path}")
    return out_path


def fetch_siu_dataframe(**kwargs):
    """Scrape SIU Director's Reports and return them as a DataFrame.

    Thin wrapper over :func:`fetch_siu_cases` (which returns the CSV
    path); used as a :data:`morie.data.DATASET_CATALOG` ``fetcher``,
    whose dispatch expects a DataFrame.
    """
    import pandas as pd

    return pd.read_csv(fetch_siu_cases(**kwargs), low_memory=False)
