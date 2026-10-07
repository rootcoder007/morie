# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture-of-Agents audit panel over ANY model -- the Python twin of the C++ ``siu::audit`` panel.

Line-for-line port of ``src/siu_core_audit.cpp`` (compiled into rmorie / morie
R as ``morie_siu_audit_panel()`` and the ``siu`` CLI): the same prompts, the
same reader tier -> hard barrier -> auditor review chain, the same health
pre-flight, the same default-model rule and the same JSON output (keys
sorted, compact), so a given backend yields the same record in every arm.
"""

from __future__ import annotations

import contextlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import llm
from .corpus import PANEL_FIELDS

__all__ = ["default_model", "healthy_models", "readers_for", "siu_audit_panel"]

_RULES = (
    "CRITICAL RULES:\n"
    "1. READ THE ENTIRE REPORT before answering. Do not skim -- injuries, "
    "officer designations, charges and the decision often appear only in the "
    "narrative or the Analysis/Decision sections near the end.\n"
    "2. Answer ONLY from what the report actually says, and give the EXACT "
    "supporting quote. If you cannot quote it, you do not know it.\n"
    "3. NEVER answer None/not_stated/unknown out of laziness. None is correct "
    "ONLY after reading the whole report and finding the field genuinely "
    "absent. A lazy None is a serious error.\n"
    "4. COUNT fields (number_of_subject_officials, number_of_witness_officials, "
    "number_of_civilian_witnesses, investigators): give the COUNT of distinct "
    "entities -- counting is NOT inference. Officers are designated explicitly; "
    "a witness-officer-only case has number_of_subject_officials = 0, a REAL "
    "value, never not_stated. Use the canonical key number_of_subject_officials "
    "(never officials/officils).\n"
    "5. Dates in ISO (YYYY-MM-DD) when stated.\n"
)


def _dump(obj: Any, indent: int | None = None) -> str:
    """nlohmann::json ``dump()`` / ``dump(1)`` layout: sorted keys, UTF-8, compact or 1-space indent."""
    if indent is None:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=indent)


def _parse_reply(reply: str) -> Any:
    s = re.sub(r"<think>[\s\S]*?</think>", "", reply)
    a, z = s.find("{"), s.rfind("}")
    if a < 0 or z < 0 or z <= a:
        raise ValueError("no JSON object in reply")
    return json.loads(s[a : z + 1])


def _reader_prompt(parsed: str, text: str) -> str:
    return (
        "You are a meticulous reviewer extracting structured data from "
        "an Ontario SIU (Special Investigations Unit) director's report. "
        "60+ fields; a single wrong answer is unacceptable.\n\n"
        + _RULES
        + '\nOutput ONLY a JSON object: field -> {"value": <value>, '
        '"quote": "<exact supporting words, empty only if genuinely '
        'absent>", "confidence": "high|medium|low"}.\n\n'
        "PARSED (the parser's guesses -- verify EACH against the report):\n" + parsed + "\n\nREPORT:\n" + text
    )


def _auditor_prompt(parsed: str, text: str, reviewers: str) -> str:
    return (
        "You are the AUDITOR -- the final authority over the reviewers. "
        "You are given the FULL report and, per field, each reviewer's "
        "value + supporting quote + confidence. 60+ fields; a single "
        "wrong answer is unacceptable.\n\n"
        "YOUR JOB:\n"
        "1. READ THE REPORT YOURSELF, in full. Do NOT just take the "
        "reviewers' word -- reviewers skim and lazily answer None.\n"
        "2. For each field, check every reviewer's value AND quote "
        "against the report; accept a value only if the report's exact "
        "words support it.\n"
        "3. Where reviewers disagree, or a quote does not support the "
        "value, or a reviewer said None but the report states it -- "
        "decide by what the REPORT literally says and correct it.\n"
        + _RULES
        + "\nOutput ONLY a JSON object: field -> final CLEAN value (the actual "
        "value, never a verdict word like 'agree', never an annotation). Use "
        "the canonical field names exactly as given.\n\n"
        "PARSED:\n" + parsed + "\n\nREVIEWERS (value/quote/confidence per "
        "field):\n" + reviewers + "\n\nREPORT:\n" + text
    )


def _per_field_reader_prompt(f, parsed_val: str, text: str) -> str:
    return (
        "Read the ENTIRE Ontario SIU director's report below, "
        "then extract ONLY this one field and nothing else.\n\n"
        "FIELD: " + f[0] + " -- " + f[2] + "\n\n" + _RULES + '\nOutput ONLY a JSON object {"value": <value>, "quote": '
        '"<exact supporting words, empty only if genuinely absent>", '
        '"confidence": "high|medium|low"}.\n\n'
        "PARSER'S GUESS for this field: " + parsed_val + "\n\nREPORT:\n" + text
    )


def _per_field_auditor_prompt(f, parsed_val: str, text: str, reviewers: str) -> str:
    return (
        "You are the AUDITOR. Read the ENTIRE report yourself, "
        "then decide the FINAL value for ONLY this one field.\n\n"
        "FIELD: " + f[0] + " -- " + f[2] + "\n\n"
        "Check the reviewers' answers against the report; accept only what "
        "the exact words support; reject lazy None; count fields must be "
        "counted (witness-officer-only = 0). " + "\nOutput ONLY a JSON "
        'object {"' + f[0] + '": <final clean value>}.\n\n'
        "PARSER'S GUESS: " + parsed_val + "\n\nREVIEWERS for this field:\n" + reviewers + "\n\nREPORT:\n" + text
    )


def _get_str(j: dict, key: str) -> str:
    if key not in j:
        return ""
    v = j[key]
    return v if isinstance(v, str) else _dump(v)


def _values_only(reader: dict) -> dict:
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in reader.items()}


def _read_report(be: llm.Backend, model: str, report: str, parsed: dict, per_field: bool) -> dict:
    if not per_field:
        return _parse_reply(llm.chat(be, model, _reader_prompt(_dump(parsed, 1), report)))
    out = {}
    for f in PANEL_FIELDS:
        with contextlib.suppress(Exception):  # as the C++ core: a failed field is skipped
            out[f[0]] = _parse_reply(llm.chat(be, model, _per_field_reader_prompt(f, _get_str(parsed, f[0]), report)))
    return out


def _audit_fields(be: llm.Backend, model: str, report: str, parsed: dict, prior: str, per_field: bool) -> dict:
    if not per_field:
        return _parse_reply(llm.chat(be, model, _auditor_prompt(_dump(parsed, 1), report, prior)))
    try:
        prior_j = json.loads(prior)
    except ValueError:
        prior_j = {}
    out = {}
    for f in PANEL_FIELDS:
        rev = _dump(prior_j[f[0]], 1) if isinstance(prior_j, dict) and f[0] in prior_j else prior
        try:
            r = _parse_reply(llm.chat(be, model, _per_field_auditor_prompt(f, _get_str(parsed, f[0]), report, rev)))
            out[f[0]] = r[f[0]] if isinstance(r, dict) and f[0] in r else r
        except Exception:  # noqa: BLE001
            pass
    return out


def readers_for(mode: int) -> int:
    """Readers a mode uses: 1 -> 1, 2 -> 1, 3 -> 2, 4 -> 3."""
    return {1: 1, 2: 1, 3: 2, 4: 3}[mode]


def default_model(be: llm.Backend) -> str:
    """``$MORIE_LLM_MODEL``, else ``$OLLAMA_MODEL``, else the first model the backend lists."""
    return llm.default_model(be)


def healthy_models(be: llm.Backend, candidates: list[str]) -> list[str]:
    """Keep only the models that answer a trivial prompt."""
    ok = []
    for m in candidates:
        try:
            if llm.chat(be, m, "Reply with the word OK."):
                ok.append(m)
        except Exception:  # noqa: BLE001
            pass
    return ok


def siu_audit_panel(
    report_text: str,
    parsed: dict | str | None = None,
    mode: int = 4,
    readers=None,
    auditors=None,
    num_readers: int = 0,
    num_auditors: int = 0,
    reader_concurrency: int = 0,
    auditor_sequential: bool = True,
    reader_granularity: str = "all",
    auditor_granularity: str = "all",
    health_check: bool = True,
    api: str = "",
    base: str = "",
    key: str = "",
    timeout: float = 300.0,
    temperature: float = 0.0,
    chat=None,
) -> dict:
    """Run the Mixture-of-Agents audit panel on one SIU report with any model.

    ``mode`` 1 = one reader, no auditor; 2 = 1 reader + auditor; 3 = 2 readers +
    auditor; 4 = 3 readers + auditor. Readers (cycled over ``readers``) read
    the full report and answer every field with a quote and confidence; after
    a hard barrier the auditors form a review chain, each auditing the
    previous stage. Readers run concurrently up to ``reader_concurrency`` (0 =
    all) against an HTTP backend; with your own ``chat`` callable they run one
    after another. ``*_granularity="per-field"`` re-reads the whole report
    once per field. Models returning an empty reply are dropped first. The
    backend is ``api``/``base``/``key`` (see :mod:`morie.siu.llm`) or ``chat``.

    Returns ``{"fields": dict, "json": str}``.

    Examples
    --------
    >>> def fake(model, prompt):
    ...     if prompt.startswith("Reply with"):
    ...         return "OK"
    ...     if "You are the AUDITOR" in prompt:
    ...         return '{"police_service": "Barrie Police Service"}'
    ...     return '{"police_service": {"value": "Barrie", "quote": "Barrie", "confidence": "high"}}'
    >>> siu_audit_panel("The Barrie Police Service ...", {}, mode=2, readers=["r1"], chat=fake)["json"]
    '{"police_service":"Barrie Police Service"}'
    """
    if mode not in (1, 2, 3, 4):
        raise ValueError("mode must be 1, 2, 3 or 4")
    for g in (reader_granularity, auditor_granularity):
        if g not in ("all", "per-field", "per_field"):
            raise ValueError('granularity must be "all" or "per-field"')
    if parsed is None:
        from .native import parse_report_text

        parsed = parse_report_text(report_text)
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            parsed = {}
    be = llm.Backend(api=api, base=base, key=key, timeout=timeout, temperature=temperature, chat=chat)
    if chat is None:
        be = llm.resolve(be)
    readers = list(readers or [])
    auditors = list(auditors or [])
    if not readers:
        m = default_model(be)
        if not m:
            raise RuntimeError("no model: name one or set MORIE_LLM_MODEL")
        readers = [m]
    if not auditors and mode != 1:
        auditors = list(readers)

    n = num_readers if num_readers > 0 else readers_for(mode)
    pool = healthy_models(be, readers) if health_check else readers
    if not pool:
        raise RuntimeError("no healthy reader model")
    roster = [pool[i % len(pool)] for i in range(n)]
    per_r = reader_granularity != "all"

    def job(i_model):
        i, model = i_model
        try:
            return f"{model}#{i + 1}", _read_report(be, model, report_text, parsed, per_r)
        except Exception:  # noqa: BLE001 -- a flaky reader drops out (>= 1 must remain)
            return f"{model}#{i + 1}", None

    if chat is None:
        slots = len(roster) if reader_concurrency <= 0 else reader_concurrency
        with ThreadPoolExecutor(max_workers=max(1, slots)) as ex:  # join = the barrier
            results = list(ex.map(job, enumerate(roster)))
    else:  # your own callable runs on this thread only
        results = [job(im) for im in enumerate(roster)]
    reviews = {tag: r for tag, r in results if r is not None}
    if not reviews:
        raise RuntimeError("all readers failed")

    na = num_auditors
    if na == 0:
        na = 0 if mode == 1 else (1 if not auditors else len(auditors))
    if na <= 0:
        out = _values_only(reviews[sorted(reviews)[0]])
        return {"fields": out, "json": _dump(out)}
    apool = healthy_models(be, auditors) if health_check else auditors
    if not apool:
        raise RuntimeError("no healthy auditor model")
    prior = _dump(reviews, 1)
    final: Any = None
    per_a = auditor_granularity != "all"
    for i in range(na):
        final = _audit_fields(be, apool[i % len(apool)], report_text, parsed, prior, per_a)
        prior = _dump(final, 1)
    return {"fields": final, "json": _dump(final)}
