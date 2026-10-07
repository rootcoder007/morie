"""morie.siu -- automated mining of Ontario Special Investigations Unit
director's reports.

Reads from siu.on.ca (publicly published reports + paired news releases),
parses each report HTML into a structured row, writes a unified SIU.csv
keyed on `case_number` (the SIU's canonical case ID, format YY-XXX-NNN).

URL invariant (see SIU_PLAN_20260506.md):
    drid (URL locator)        ≠ nrid (news-release locator)
    case_number               = canonical join key

Public API:
    scrape_drid(drid, *, client=None, cache=True) -> dict
        Fetch + parse a single director's report by drid.
    scrape_range(drid_min, drid_max, **kw) -> Iterator[dict]
        Polite range scrape with concurrency limits.
    parse_html(html, *, drid=None, source_url=None) -> dict
        Pure parser -- no network -- turns one HTML page into a row.
    write_csv(rows, path) -> None
    write_jsonl(rows, path) -> None         # for narrative_full bodies
    SIU_COLUMNS                              # 45-col canonical schema

Examples
--------
>>> from morie.siu import html_to_text, to_iso_date
>>> to_iso_date("2nd February 2018")
'2018-02-02'
>>> html_to_text("<p>Qu&eacute;bec &amp; Ontario</p>").strip()
'Québec & Ontario'
"""

from . import analyze, llm
from ._parser import parse_html, parse_news_html
from ._schema import BLANK_ROW, SIU_COLUMNS
from ._scraper import scrape_drid, scrape_range
from ._writer import write_csv, write_jsonl
from .audit import siu_audit_panel
from .corpus import (
    PANEL_FIELDS,
    resolve_subject_officials,
    siu_panel,
    siu_reports,
    siu_resolve_so,
    strip_boilerplate,
)
from .native import html_to_text, parse_report_html, parse_report_text, to_iso_date

__all__ = [
    "SIU_COLUMNS",
    "BLANK_ROW",
    "scrape_drid",
    "scrape_range",
    "parse_html",
    "parse_news_html",
    "write_csv",
    "write_jsonl",
    "analyze",
    "PANEL_FIELDS",
    "resolve_subject_officials",
    "siu_panel",
    "siu_reports",
    "siu_resolve_so",
    "strip_boilerplate",
    "html_to_text",
    "llm",
    "parse_report_html",
    "parse_report_text",
    "siu_audit_panel",
    "to_iso_date",
]
