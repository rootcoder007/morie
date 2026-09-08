# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verified-corpus-first SIU surface -- the Python twin of rmorie's.

Three layers, mirroring the R ecosystem exactly:

* :func:`siu_reports` -- the panel-reviewed corpus shipped by the
  'rmoriedata' R package (5,157 reports x 65 columns; 2,182 English
  entries read and cross-audited by a multi-agent review panel),
  downloaded once with a pinned SHA-256 and cached. Reviewed reports are
  NEVER re-derived.
* :func:`siu_resolve_so` -- subject-official counts: the verified corpus
  answers first; unreviewed text falls back to the zero-wrong rule
  engine (a faithful port of rmoriebricklayer's ``siu_resolve.cpp``,
  which scores 2,117 correct / 65 declined / 0 wrong on the corpus).
* :func:`siu_panel` -- the Mixture-of-Agents reading panel for NEW
  reports, on the user's own Ollama-compatible server (``OLLAMA_HOST`` /
  ``OLLAMA_MODEL``; no hardcoded default model).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

__all__ = [
    "PANEL_FIELDS",
    "resolve_subject_officials",
    "siu_panel",
    "siu_reports",
    "siu_resolve_so",
    "strip_boilerplate",
]

# ---------------------------------------------------------------- corpus

_CORPUS_URL = (
    "https://raw.githubusercontent.com/rootcoder007/rmoriedata/main/"
    "inst/extdata/siu_directors_reports.csv.gz"
)
# SHA-256 of the corpus snapshot this build trusts (bricklayer-style pin).
_CORPUS_SHA256 = "9fc08bb8030723e50e14b29dc6f3b07cb495761f4028906ea6e053f18e5509a9"


def _cache_dir() -> Path:
    d = Path(os.environ.get("MORIE_SIU_CACHE", "~/.cache/morie/siu"))
    d = d.expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def siu_reports(update: bool = False, cache_dir: str | Path | None = None):
    """Return the panel-reviewed SIU corpus as a pandas DataFrame.

    The verified corpus is the single source of truth: reviewed rows carry
    ``panel_reviewed == True`` and their ``number_of_subject_officials``
    values were verified by the multi-agent panel. ``update=True`` only
    re-downloads the snapshot (same SHA-256 pin); it never re-derives
    reviewed values.
    """
    from morie.fn import _frame_core as pd

    cache = (Path(cache_dir).expanduser() if cache_dir else _cache_dir())
    dest = cache / "siu_directors_reports.csv.gz"
    if update or not dest.exists():
        import httpx

        r = httpx.get(_CORPUS_URL, timeout=120, follow_redirects=True)
        r.raise_for_status()
        digest = hashlib.sha256(r.content).hexdigest()
        if digest != _CORPUS_SHA256:
            raise RuntimeError(
                "SIU corpus integrity check failed: expected sha256 "
                f"{_CORPUS_SHA256[:16]}..., got {digest[:16]}... -- refusing "
                "to cache an unverified snapshot."
            )
        dest.write_bytes(r.content)
    with gzip.open(dest, "rt", encoding="utf-8") as fh:
        return pd.read_csv(fh, low_memory=False)


# ------------------------------------------------- zero-wrong rule engine
# Faithful port of rmoriebricklayer src/siu_resolve.cpp (the canonical
# home). Edit the C++ first, then mirror here -- the regression tests pin
# every failure class from the 2,182-report zero-wrong pass.

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_BOILER = re.compile(
    r"this information may include[\s\S]*?(?:affected person|evidence)\.?",
    re.IGNORECASE,
)
_GLOSSARY = re.compile(
    r"who,?\s+in the (?:opinion of the SIU Director|"
    r"SIU Director(?:'|’)?s opinion),?"
    r"[\s\S]{0,120}?not a subject offic(?:er|ial)[^.]*\.?",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Normalise NBSPs and remove the privacy + glossary boilerplate."""
    norm = text.replace(" ", " ")
    out = _BOILER.sub(" ", norm)
    return _GLOSSARY.sub(" ", out)


# "SO" is case-strict and "#" is REQUIRED (an icase optional-# variant
# matched "...also 59..." in the wild and blew counts up).
_ORD_SO = re.compile(r"\bSO\s*#\s*(\d{1,2})\b")
_ORD_SPELLED = re.compile(
    r"subject offic(?:er|ial)\s*#\s*(\d{1,2})\b", re.IGNORECASE)
_SECTION = re.compile(r"Subject Offic(?:er|ial)s\b")
_NEXT_SECTION = re.compile(
    r"\n\s{0,3}(?:Witness Offic(?:er|ial)s|Civilian Witness(?:es)?|"
    r"Service Employee Witness|Incident Narrative|Materials [Oo]btained|"
    r"The Scene|Evidence\n|Nature of Injur)")
_ENTRY = re.compile(
    r"\bSO\s*(?:#\s*\d{1,2})?\s{0,3}"
    r"(?:Interviewed|Declined|Did not consent|Not interviewed)")
_ANCHOR1 = re.compile(r"\bSO\s*#\s*1\b|subject offic(?:er|ial)\s*#\s*1\b")
_PLURAL = re.compile(
    r"\bthe\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+subject offic(?:er|ial)s\b", re.IGNORECASE)
_THE_SO = re.compile(r"\b[Tt]he SO\b")
_THE_SUBJ = re.compile(r"\bthe subject offic(?:er|ial)\b", re.IGNORECASE)
_ANY_PLURAL = re.compile(
    r"\bthe SOs\b|the subject offic(?:er|ial)s\b", re.IGNORECASE)
_ZERO = (
    re.compile(r"no subject offic(?:er|ial)s?\b", re.IGNORECASE),
    re.compile(
        r"(?:did not|not|never)\s+designate[d]?\s+(?:a\s+|any\s+)?"
        r"subject offic", re.IGNORECASE),
)


def _max_ordinal(s: str) -> int:
    vals = [int(m.group(1)) for m in _ORD_SO.finditer(s)]
    vals += [int(m.group(1)) for m in _ORD_SPELLED.finditer(s)]
    return max(vals, default=0)


def resolve_subject_officials(report_text: str) -> tuple[int | None, str]:
    """Rule-engine SO count for one report's text. ``(count|None, reason)``.

    ``None`` means "needs a human/panel read" -- never a guess.
    """
    body = strip_boilerplate(report_text)

    # 0. The Team block under "Subject Officials/Officers" is authoritative.
    sec = _SECTION.search(body)
    if sec:
        window = body[sec.end():sec.end() + 2500]
        nxt = _NEXT_SECTION.search(window)
        if nxt:
            window = window[: nxt.start()]
        sec_ord = _max_ordinal(window)
        entries = len(_ENTRY.findall(window))
        sec_n = max(sec_ord, entries)
        if sec_n > 0:
            return sec_n, f"section: max(ordinal {sec_ord}, entries {entries})"

    # 1. Document-wide highest ordinal, but only with the "#1" roster anchor
    # (a lone "SO #7 of YRP" is another force's shorthand).
    max_ord = _max_ordinal(body)
    if max_ord > 0 and _ANCHOR1.search(body):
        return max_ord, f"max ordinal SO #{max_ord}"

    # 2. Spelled-out / numeric plural: "the two subject officials".
    m = _PLURAL.search(body)
    if m:
        tok = m.group(1).lower()
        n = _WORD_NUM.get(tok) or int(tok)
        return n, f"plural cue '{m.group(0)}'"

    # 3. A subject official is PRESENT (runs BEFORE the zero rule).
    the_so = len(_THE_SO.findall(body))
    the_subj = len(_THE_SUBJ.findall(body))
    plural = bool(_ANY_PLURAL.search(body))
    if (the_so + the_subj) >= 1 and not plural:
        return 1, (f"singular present: 'the SO'x{the_so} "
                   f"'the subject official'x{the_subj}")

    # 4. Explicitly ZERO (witness-official-only cases; direct assertion only).
    for rx in _ZERO:
        if rx.search(body):
            return 0, "zero: witness-officer-only / 'not a subject official'"

    # 5. Needs a human read.
    return None, (f"UNRESOLVED: 'the SO'x{the_so} 'the subj off'x{the_subj}")


def siu_resolve_so(text: str | None = None, drid: int | None = None) -> dict:
    """Subject-official count: verified corpus first, rules second.

    Exactly one of ``text`` / ``drid`` must be given. A ``drid`` found in the
    panel-reviewed corpus returns the verified value (reason
    ``"panel-reviewed corpus (verified)"``); anything else goes through the
    zero-wrong rule engine.
    """
    if (text is None) == (drid is None):
        raise ValueError("give exactly one of text= or drid=")
    if drid is not None:
        df = siu_reports()
        row = df[df["drid"] == int(drid)]
        if len(row):
            reviewed = str(row.iloc[0].get("panel_reviewed", "")).lower() in (
                "true", "1", "1.0")
            val = row.iloc[0].get("number_of_subject_officials")
            if reviewed and str(val) not in ("", "nan", "None"):
                return {
                    "count": int(float(val)),
                    "reason": "panel-reviewed corpus (verified)",
                }
        raise KeyError(
            f"drid {drid} not in the reviewed corpus; fetch the report and "
            "call siu_resolve_so(text=...) or run siu_panel() on it")
    count, reason = resolve_subject_officials(text)
    return {"count": count, "reason": reason}


# ----------------------------------------------------------------- panel

#: The 16 panel-reviewed fields (name, is_count, per-field instruction) --
#: mirrors rmoriebricklayer's ``siu_schema.h`` (the single source of truth).
PANEL_FIELDS: tuple[tuple[str, bool, str], ...] = (
    ("police_service", False,
     "the police service(s) whose officials the SIU investigated"),
    ("date_of_incident_iso", False,
     "the date the incident occurred (ISO YYYY-MM-DD)"),
    ("date_siu_notified_iso", False,
     "the date the SIU was notified/invoked (ISO YYYY-MM-DD)"),
    ("date_of_director_decision_iso", False,
     "the date of the Director's decision (ISO YYYY-MM-DD)"),
    ("siu_investigators", True, "the number of SIU investigators assigned"),
    ("siu_forensics_investigators", True,
     "the number of SIU forensic investigators assigned"),
    ("number_of_witness_officials", True,
     "the count of distinct WITNESS officers/officials (WO)"),
    ("number_of_civilian_witnesses", True,
     "the count of distinct civilian witnesses (CW)"),
    ("number_of_subject_officers", True,
     "the count of distinct SUBJECT officers/officials (SO); a "
     "witness-officer-only investigation is 0"),
    ("age_affected", False,
     "the age (or age range) of the affected person/complainant"),
    ("sex_gender_affected", False,
     "the sex/gender of the affected person/complainant"),
    ("charges_recommended", False,
     "whether charges were recommended/laid, and against whom"),
    ("directors_name", False, "the name of the SIU Director who decided"),
    ("location_of_call", False,
     "the location/address where the incident occurred"),
    ("specific_injuries", False,
     "the specific injuries the affected person sustained"),
    ("relevant_legislation", False,
     "the legislation/Criminal Code sections the Director analysed"),
)

_TAGS = re.compile(r"<[^>]+>")
_JSON_BLOB = re.compile(r"\{[\s\S]*\}")


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", text)
    text = _TAGS.sub(" ", text)
    import html as _h

    text = _h.unescape(text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _field_block(name: str, is_count: bool, desc: str) -> str:
    tag = " (COUNT; 0 only for a witness-official-only case)" if is_count else ""
    return f"- {name}{tag}: {desc}"


_BASE_RULES = (
    "You are auditing an Ontario Special Investigations Unit director's "
    "report. Read the ENTIRE report before answering. Never answer 'None' "
    "or 'not stated' unless you have read the full report and the value is "
    "genuinely absent. Count fields must be COUNTED from the report (zero "
    "is a real answer only when the report is witness-official-only). "
    "Every answer MUST carry a short verbatim quote from the report as "
    'evidence. Answer as strict JSON: {"field": {"value": ..., '
    '"quote": ...}, ...}.'
)


def _panel_extract(raw: Any, field_names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {fn: None for fn in field_names}

    def parse_one(txt: str):
        m = _JSON_BLOB.search(txt or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(raw, dict):  # per-field replies
        for fn in set(raw) & set(field_names):
            j = parse_one(raw[fn])
            if isinstance(j, dict):
                v = j.get("value", next(iter(j.values()), None))
                if isinstance(v, dict):
                    v = v.get("value")
                if v is not None:
                    out[fn] = str(v)
        return out
    j = parse_one(raw)
    if isinstance(j, dict):
        for fn in set(j) & set(field_names):
            v = j[fn]
            if isinstance(v, dict):
                v = v.get("value", next(iter(v.values()), None))
            if v is not None:
                out[fn] = str(v)
    return out


def siu_panel(
    html: str | int,
    mode: int = 4,
    readers: list[str] | None = None,
    auditors: list[str] | None = None,
    reader_concurrency: int = 3,
    granularity: str = "all_fields",
    host: str | None = None,
    timeout: float = 300,
) -> dict:
    """Mixture-of-Agents reading panel for SIU reports not yet in the corpus.

    Modes mirror the panel that built the reviewed corpus: 1 = one reader;
    2 = reader + auditor; 3 = two readers + auditor; 4 = three readers +
    auditor (default). Explicit ``readers``/``auditors`` override the mode.
    Readers run concurrently (hard barrier before the auditor); auditors
    chain sequentially, each seeing its predecessors' verdicts.

    Models come from YOUR server: ``host`` (default ``$OLLAMA_HOST``) and
    ``$OLLAMA_MODEL`` (default: first model the server lists). Nothing is
    hardcoded.
    """
    import httpx

    if mode not in (1, 2, 3, 4):
        raise ValueError("mode must be 1-4")
    if granularity not in ("all_fields", "per_field"):
        raise ValueError("granularity must be 'all_fields' or 'per_field'")

    if isinstance(html, int):
        from .. import siu_fetch as _sf  # legacy fetch engine

        url = ("https://www.siu.on.ca/en/directors_report_details.php?"
               f"drid={int(html)}")
        html = _sf._http_get(url, timeout=int(timeout))
    text = _html_to_text(html) if "<" in html else html

    host = host or os.environ.get("OLLAMA_HOST", "")
    if not host:
        raise RuntimeError(
            "no Ollama server: set OLLAMA_HOST (local, tailnet, or a "
            "Cloudflare-tunnelled endpoint)")
    host = host.rstrip("/")

    tags = httpx.get(f"{host}/api/tags", timeout=30).json()
    available = [m["name"] for m in tags.get("models", [])]
    if not available:
        raise RuntimeError(f"Ollama at {host} serves no models")
    default_model = os.environ.get("OLLAMA_MODEL", available[0])

    n_readers = (1, 1, 2, 3)[mode - 1]
    n_auditors = (0, 1, 1, 1)[mode - 1]
    if readers is None:
        pool = list(dict.fromkeys([default_model, *available]))
        readers = [pool[i % len(pool)] for i in range(n_readers)]
    if auditors is None:
        auditors = [default_model] * n_auditors

    schema_txt = "\n".join(_field_block(*f) for f in PANEL_FIELDS)
    field_names = [f[0] for f in PANEL_FIELDS]

    def ask(model: str, prompt: str) -> str:
        r = httpx.post(
            f"{host}/api/chat",
            json={"model": model, "stream": False,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def read_once(model: str):
        if granularity == "per_field":
            return {
                name: ask(model, (f"{_BASE_RULES}\n\nAnswer ONLY this field:\n"
                                  f"{_field_block(name, cnt, desc)}\n\n"
                                  f"REPORT:\n{text}"))
                for name, cnt, desc in PANEL_FIELDS
            }
        return ask(model, f"{_BASE_RULES}\n\nFields:\n{schema_txt}\n\n"
                          f"REPORT:\n{text}")

    # Readers: concurrent up to the cap; the executor join is the barrier.
    cap = max(1, min(int(reader_concurrency), len(readers)))
    with ThreadPoolExecutor(max_workers=cap) as pool_ex:
        reader_out_list = list(pool_ex.map(read_once, readers))
    reader_out = {
        f"reader_{i + 1}:{m}": r
        for i, (m, r) in enumerate(zip(readers, reader_out_list))
    }

    audit_chain: dict[str, str] = {}
    prior = ""
    final_raw: Any = None
    for i, model in enumerate(auditors, start=1):
        readers_txt = "\n\n".join(
            f"{k}:\n" + (json.dumps(v) if isinstance(v, dict) else str(v))
            for k, v in reader_out.items())
        p = (f"{_BASE_RULES}\n\nYou are the AUDITOR. {len(readers)} reader(s) "
             "answered every field; their raw answers follow. Read the report "
             "yourself, weigh their answers and quotes, and issue the FINAL "
             "value for every field under its canonical key "
             "(number_of_subject_officers, never a variant spelling)."
             + (f"\nPrevious auditor verdicts:\n{prior}" if prior else "")
             + f"\n\nReader answers:\n{readers_txt}"
             + f"\n\nFields:\n{schema_txt}\n\nREPORT:\n{text}")
        final_raw = ask(model, p)
        audit_chain[f"auditor_{i}:{model}"] = final_raw
        prior = f"{prior}\n{final_raw}"

    fields = _panel_extract(
        final_raw if final_raw is not None else next(iter(reader_out.values())),
        field_names)
    return {
        "fields": fields,
        "readers": reader_out,
        "audit_chain": audit_chain,
        "models": {"readers": readers, "auditors": auditors, "host": host},
    }
