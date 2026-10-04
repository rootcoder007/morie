# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native 16-field SIU report parser -- the Python twin of the C++ core.

A line-for-line port of the C++17 parser compiled into rmorie / morie
(``src/siu/parse.hpp``, ``src/siu_core_parse.cpp``) and rmoriebricklayer, so
the three arms return identical fields. To reproduce ``std::regex`` exactly
(byte-oriented, ASCII ``\\s``/``\\b``/``\\w``, ASCII-only case folding,
repetition counts in bytes) every pattern runs on the UTF-8 bytes of the text;
results are decoded back to ``str``.

Public API: :func:`html_to_text`, :func:`parse_report_text`,
:func:`parse_report_html`, :func:`to_iso_date`, :data:`SCHEMA`.
"""

from __future__ import annotations

import re

from .corpus import PANEL_FIELDS

__all__ = ["SCHEMA", "html_to_text", "parse_report_html", "parse_report_text", "to_iso_date"]

#: The 16 panel-reviewed fields: (name, is_count, description), as schema.hpp.
SCHEMA = PANEL_FIELDS

_I = re.IGNORECASE
_META = re.compile(rb"[\^\$\.\|\?\*\+\(\)\[\]\{\}\\]")


def _b(s: str) -> bytes:
    return s.encode("utf-8")


def _s(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def _trim(s: bytes) -> bytes:
    a = len(s) - len(s.lstrip(b" \t\r\n"))
    if a == len(s):
        return b""
    z = len(s.rstrip(b" \t\r\n,;:"))
    return s[a:z]


def _flatten_ws(s: bytes) -> bytes:
    return re.sub(rb"\s+", b" ", s)


def _esc(h: bytes) -> bytes:
    return _META.sub(lambda m: b"\\" + m.group(0), h)


def _section_text(text: bytes, header: str, ends=()) -> bytes:
    # the LAST heading line: report pages open with a table of contents that repeats every
    # section title on its own line, and slicing from there gave each section the TOC text
    ms = list(re.finditer(rb"(^|\n)[ \t]*" + _esc(_b(header)) + rb"[ \t]*\n", text))
    if not ms:
        return b""
    start = ms[-1].end()
    end = len(text)
    tail = text[start:]
    for em in ends:
        mm = re.search(rb"(^|\n)[ \t]*" + _esc(_b(em)) + rb"[ \t]*\n", tail)
        if mm:
            end = min(end, start + mm.start())
    return text[start:end]


def _team_count(text: bytes, label: str) -> bytes:
    m = re.search(b"Number of " + _b(label) + rb"[^0-9\n]{0,30}(\d+)", text, _I)
    return m.group(1) if m else b""


def _count_tagged(section: bytes, prefix: str) -> bytes:
    if not section:
        return b""
    flat = _flatten_ws(section)
    mx = 0
    for m in re.finditer(rb"\b" + _b(prefix) + rb"\s*#?\s*(\d+)\b", flat):
        mx = max(mx, int(m.group(1)))
    if mx > 0:
        return str(mx).encode()
    if re.search(rb"\b" + _b(prefix) + rb"\b", flat):
        return b"1"
    return b""


_POLICE = re.compile(
    rb"((?:[A-Z][A-Za-z'\-]+[ \t]+){1,5}(?:Police Service|Provincial Police|Police|Constabulary))\b"
    rb"(?![ \t]+Services?[ \t]+(?:Act|Board))"
)
_NOTIFIER = re.compile(
    rb"((?:[A-Z][A-Za-z'\-]+[ \t]+){1,5}(?:Police Service|Provincial Police|Police|Constabulary))\s*"
    rb"(?:\(\s*[A-Z]{2,6}\s*\)\s*)?(?:notified|contacted)\s+the\s+SIU"
)
_LEAD0 = re.compile(rb"^(?:The|A|An|At|On|In|By)\s+")
_LEAD = re.compile(rb"^(?:The|A|An|Of|And|On|In|By|To|With|From|That|This|Local)\s+")
_HEADLINE = re.compile(
    rb"\b(?:Between|Involving|After|During|Following|Collision|Crash|Shooting|Death|Injury|Incident|Arrest)\b"
)
_ABBR = (
    (b"OPP", b"Ontario Provincial Police"),
    (b"TPS", b"Toronto Police Service"),
    (b"RCMP", b"RCMP"),
    (b"NRPS", b"Niagara Regional Police Service"),
)


def _detect_police_service(text: bytes) -> bytes:
    # the notification sentence names the force that called the SIU in: prefer it over counting
    m = _NOTIFIER.search(text)
    if m:
        name = _trim(m.group(1))
        for _ in range(3):
            name = _LEAD0.sub(b"", name)
        if name:
            return name
    counts: dict[bytes, int] = {}
    for m in _POLICE.finditer(text):
        name = _trim(m.group(1))
        for _ in range(3):
            name = _LEAD.sub(b"", name)
        if _HEADLINE.search(name):  # incident headlines are not service names
            continue
        if name:
            counts[name] = counts.get(name, 0) + 1
    best, bestc = b"", 0
    for k in sorted(counts):  # std::map order
        v = counts[k]
        if v > bestc or (v == bestc and len(k) > len(best)):
            best, bestc = k, v
    if best:
        return best
    for ab, full in _ABBR:
        if re.search(rb"\b" + ab + rb"\b", text):
            return full
    return b""


_ON_DATE = re.compile(rb"\b[Oo]n\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})")


def _detect_incident_date(text: bytes) -> bytes:
    for sec_name in ("Incident Narrative", "The Investigation"):
        sec = _section_text(
            text,
            sec_name,
            ("Nature of Injuries", "Evidence", "The Team", "Analysis and Director", "Relevant Legislation"),
        )
        if not sec:
            continue
        for m in _ON_DATE.finditer(sec):
            a = m.start() - 50 if m.start() > 50 else 0
            window = sec[a : a + (m.start() - a + (m.end() - m.start()) + 80)].lower()
            if b"notified" in window or b"contacted the siu" in window or b"notification of the siu" in window:
                continue
            return m.group(1).replace(b",", b"")
    return b""


def _detect_siu_notified(text: bytes) -> bytes:
    inv = _section_text(text, "The Investigation", ("The Team", "Incident Narrative"))
    hay = inv if inv else text
    m = re.search(
        rb"\b[Oo]n\s+(?:[A-Z][a-z]+,?\s+)?([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})[^\n]{0,200}?"
        rb"(?:notified|contacted)\s+the\s+SIU",
        hay,
    )
    if m:
        return m.group(1).replace(b",", b"")
    m = re.search(
        rb"(?:notified|contacted)\s+the\s+SIU[^\n]{0,200}?[Oo]n\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})",
        hay,
    )
    if m:
        return m.group(1).replace(b",", b"")
    notif = _section_text(text, "Notification of the SIU", ("The Team", "Incident Narrative", "Evidence"))
    if notif:
        m = re.search(rb"On\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})", notif)
        if m:
            return m.group(1).replace(b",", b"")
    return b""


def _detect_decision_date(text: bytes) -> bytes:
    m = re.search(rb"Date:\s*([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})", text)
    if m:
        return m.group(1).replace(b",", b"")
    m = re.search(rb"Date:\s*(\d{4}-\d{2}-\d{2})", text)
    return m.group(1) if m else b""


def _detect_location(text: bytes) -> bytes:
    inv = _section_text(text, "The Investigation", ("The Team", "Incident Narrative"))
    hay = _flatten_ws(inv if inv else text)
    m = re.search(
        rb"in the (Township|City|Town|Municipality|Region) of ([A-Z][A-Za-z \-]+?)(?:[\.,]|\s+(?:on|at|when))", hay
    )
    return m.group(1) + b" of " + _trim(m.group(2)) if m else b""


def _detect_age_sex(text: bytes) -> tuple[bytes, bytes]:
    m = re.search(
        rb"\b(\d{1,3})[\s\-]year[\s\-]old\s+(woman|man|female|male|girl|boy|person|individual|youth|child|adult)\b",
        text,
        _I,
    )
    return (m.group(1), m.group(2).lower()) if m else (b"", b"")


def _detect_specific_injuries(text: bytes) -> bytes:
    m = re.search(
        rb"((?:fractured?|broken|lacerat\w+|gunshot|stab\w+|burns?)[^\n]{1,200}?"
        rb"(?:rib|leg|arm|skull|wrist|ankle|jaw|nose|tooth|finger|spine|vertebra)[^\n]{0,80})",
        text,
        _I,
    )
    return _trim(m.group(1)) if m else b""


def _detect_legislation(text: bytes) -> bytes:
    sec = _section_text(text, "Relevant Legislation", ("Analysis and Director", "News Releases"))
    if not sec:
        return b""
    acts: list[bytes] = []
    for m in re.finditer(rb"Section\s+\d+(?:\.\d+)*(?:\([^)]+\))?,?\s+([A-Z][^\n,]{2,80}?)(?:\s*[-]|\s*$|\n)", sec):
        act = _trim(m.group(1))
        if act not in acts:
            acts.append(act)
    return b"; ".join(acts)


def _detect_charges(text: bytes) -> bytes:
    ends = ("Endnotes", "News Release", "Note:")
    sec = _section_text(text, "Analysis and Director's Decision", ends)
    if not sec:
        sec = _section_text(text, "Analysis and Director’s Decision", ends)
    if not sec:
        return b""
    low = sec.lower()
    for p in (
        b"no charges",
        b"shall issue",
        b"none shall issue",
        b"no basis for charges",
        b"no reasonable grounds",
        b"do not lay",
        b"decline to lay",
        b"lack the necessary grounds",
    ):
        if p in low:
            return b"FALSE"
    for p in (b"charged with", b"criminal charges have been laid", b"charges have been laid"):
        if p in low:
            return b"TRUE"
    return b""


_DIR_OWN_LINE = re.compile(
    rb"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})[ \t]*\n(?:[ \t]*\n)*[ \t]*"
    rb"(?:(?:Interim|Acting)[ \t]+)?(?:Director|Directeur|Directrice)"
    rb"(?:[ \t]+par[ \t]+int\xc3\xa9rim)?[ \t]*(?:\n|$)"
)
_DIR_COMMA = re.compile(
    rb"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})[ \t]*,[ \t]*"
    rb"(?:(?:Interim|Acting)[ \t]+)?(?:Director|Directeur|Directrice)[ \t]*(?:\n|$)"
)


def _detect_directors_name(text: bytes) -> bytes:
    # "Director" alone on its line (blank lines allowed before it) or after a
    # comma; navigation like "Public Reports\nDirector's Resource Committee"
    # is never a signature.
    for pat in (_DIR_OWN_LINE, _DIR_COMMA):
        for m in pat.finditer(text):
            name = _trim(m.group(1))
            if b"the" not in name.lower():
                return name
    return b""


_EN = (
    "The Investigation",
    "Notification of the SIU",
    "Mandate engaged",
    "Civilian Witnesses",
    "Witness Officers",
    "Subject Officers",
    "Analysis and Director's Decision",
    "Witness Officials",
)
_FR = (
    "L'enquête",
    "Exercice du mandat",
    "Éléments de preuve",
    "Dispositions législatives pertinentes",
    "Témoins civils",
    "Agents impliqués",
    "Mandat de l'UES",
)


def _detect_language(text: bytes) -> bytes:
    e = sum(1 for mk in _EN if _b(mk) in text)
    f = sum(1 for mk in _FR if _b(mk) in text)
    if e >= 2 and e > f:
        return b"en"
    if f >= 2 and f > e:
        return b"fr"
    return b"unknown"


_MONTHS = {
    b"january": 1,
    b"february": 2,
    b"march": 3,
    b"april": 4,
    b"may": 5,
    b"june": 6,
    b"july": 7,
    b"august": 8,
    b"september": 9,
    b"october": 10,
    b"november": 11,
    b"december": 12,
}


# French months and English abbreviations, as the C++ core reads them
_MONTHS.update(
    {
        b"janvier": 1,
        b"f\xc3\xa9vrier": 2,
        b"fevrier": 2,
        b"mars": 3,
        b"avril": 4,
        b"mai": 5,
        b"juin": 6,
        b"juillet": 7,
        b"ao\xc3\xbbt": 8,
        b"aout": 8,
        b"septembre": 9,
        b"octobre": 10,
        b"novembre": 11,
        b"d\xc3\xa9cembre": 12,
        b"decembre": 12,
        **{
            m[:3].encode(): i
            for i, m in enumerate(
                ("january february march april may june july august september " "october november december").split(), 1
            )
        },
        b"sept": 9,
    }
)
_ISO_MD = re.compile(rb"([^\s\d,.]+)\.?\s+(\d{1,2})(?:st|nd|rd|th|er|e)?,?\s+(\d{4})")  # August 3rd, 2017
_ISO_DM = re.compile(rb"(\d{1,2})(?:er|e|st|nd|rd|th)?\s+([^\s\d,.]+)\.?,?\s+(\d{4})")  # 3 août 2017


def _iso(human: bytes) -> bytes:
    m = _ISO_MD.search(human)
    if m and m.group(1).lower() in _MONTHS:
        mo, day, year = _MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3)
    else:
        m = _ISO_DM.search(human)
        if not (m and m.group(2).lower() in _MONTHS):
            return human if re.fullmatch(rb"\d{4}-\d{2}-\d{2}", human) else b""
        mo, day, year = _MONTHS[m.group(2).lower()], int(m.group(1)), m.group(3)
    return b"%s-%02d-%02d" % (year, mo, day)


def to_iso_date(human: str) -> str:
    """``"January 5, 2023"`` -> ``"2023-01-05"`` (``""`` if unparseable).

    Examples
    --------
    >>> to_iso_date("January 5, 2023"), to_iso_date("2023-01-05"), to_iso_date("soon")
    ('2023-01-05', '2023-01-05', '')
    """
    return _s(_iso(_b(human)))


_HTML_STEPS = (
    (re.compile(rb"\r\n?"), b"\n"),  # live pages are CRLF; \r defeats line-anchored rules
    (re.compile(rb"<(script|style)[^>]*>[\s\S]*?</\1>", _I), b" "),
    (re.compile(rb"</(p|div|h[1-6]|tr|li|br)>|<br\s*/?>", _I), b"\n"),
    (re.compile(rb"<[^>]+>"), b" "),
    (re.compile(rb"&nbsp;"), b" "),
    (re.compile(rb"&amp;"), b"&"),
    (re.compile(rb"&#8217;|&rsquo;"), b"'"),
    (re.compile(rb"&#8216;|&lsquo;"), b"'"),
    (re.compile(rb"&#8220;|&ldquo;|&#8221;|&rdquo;"), b'"'),
    (re.compile(rb"&quot;"), b'"'),
    (re.compile(rb"&#0?39;|&apos;"), b"'"),
    (re.compile(rb"&#8211;|&ndash;"), b"-"),
    (re.compile(rb"&#8212;|&mdash;"), b"--"),
    (re.compile(rb"&lt;"), b"<"),
    (re.compile(rb"&gt;"), b">"),
    (re.compile(rb"[ \t]+"), b" "),
    (re.compile(rb" ?\n ?"), b"\n"),
    (re.compile(rb"\n{3,}"), b"\n\n"),
)


def _html_to_text(html: bytes) -> bytes:
    import html as _html

    t = html
    for pat, rep in _HTML_STEPS:
        t = pat.sub(rep, t)
    # what the ASCII steps leave (&eacute;, &#233;, &hellip;) becomes its character
    return _b(_html.unescape(_s(t).replace("&hellip;", "...").replace("&#8230;", "...")))


def html_to_text(html: str) -> str:
    """Strip tags, scripts and entities from report HTML, keeping line structure.

    Examples
    --------
    >>> html_to_text("<p>The&nbsp;Team</p><p>SO #1</p>")
    ' The Team\\nSO #1\\n'
    """
    return _s(_html_to_text(_b(html)))


def _parse(t: bytes) -> dict[str, str]:
    f: dict[bytes, bytes] = {b"_language": _detect_language(t)}
    f[b"police_service"] = _detect_police_service(t)
    f[b"date_of_incident_iso"] = _iso(_detect_incident_date(t))
    f[b"date_siu_notified_iso"] = _iso(_detect_siu_notified(t))
    f[b"date_of_director_decision_iso"] = _iso(_detect_decision_date(t))
    f[b"siu_investigators"] = _team_count(t, "SIU Investigators")
    f[b"siu_forensics_investigators"] = _team_count(t, "SIU Forensic Investigators")
    so = _section_text(t, "Subject Officers", ("Incident Narrative", "Evidence", "Witness Officers"))
    if not so:
        so = _section_text(t, "Subject Officials", ("Incident Narrative", "Evidence", "Witness Officials"))
    f[b"number_of_subject_officers"] = _count_tagged(so if so else t, "SO")
    wo = _section_text(t, "Witness Officers", ("Incident Narrative", "Evidence", "Subject Officers"))
    if not wo:
        wo = _section_text(t, "Witness Officials", ("Incident Narrative", "Evidence", "Subject Officials"))
    if wo and re.search(rb"no police officers? witness", _flatten_ws(wo).lower()):
        f[b"number_of_witness_officials"] = b"0"
    else:
        f[b"number_of_witness_officials"] = _count_tagged(wo, "WO")
    cw = _section_text(
        t, "Civilian Witnesses", ("Incident Narrative", "Evidence", "Witness Officers", "Witness Officials")
    )
    f[b"number_of_civilian_witnesses"] = _count_tagged(cw, "CW")
    age, sex = _detect_age_sex(t)
    f[b"age_affected"] = age
    f[b"sex_gender_affected"] = sex
    f[b"charges_recommended"] = _detect_charges(t)
    f[b"directors_name"] = _detect_directors_name(t)
    f[b"location_of_call"] = _detect_location(t)
    f[b"specific_injuries"] = _detect_specific_injuries(t)
    f[b"relevant_legislation"] = _detect_legislation(t)
    return {_s(k): _s(f[k]) for k in sorted(f)}


def parse_report_text(text: str) -> dict[str, str]:
    """Parse plain report text into the 16 schema fields plus ``_language`` (``""`` when not stated)."""
    return _parse(_b(text))


def parse_report_html(html: str) -> dict[str, str]:
    """HTML -> fields in one call (:func:`html_to_text` then :func:`parse_report_text`).

    Examples
    --------
    >>> f = parse_report_html("<p>The Investigation</p><p>It happened in the City of Barrie, a 34-year-old man ...</p>")
    >>> f["age_affected"], f["sex_gender_affected"], f["location_of_call"]
    ('34', 'man', 'City of Barrie')
    """
    return _parse(_html_to_text(_b(html)))
