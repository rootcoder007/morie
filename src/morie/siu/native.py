# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native 16-field SIU report parser -- the Python twin of the C++ core.

A line-for-line port of the C++17 parser and subject-official resolver that
rmoriebricklayer keeps canonically (``src/siu_parse.cpp``,
``src/siu_resolve.cpp``) and that morie compiles into ``morie._core``
(``libmorie/siu/``). When the compiled core is present every public function
here calls it; this port is the fallback for a pure-Python install and must
return the same fields. To reproduce ``std::regex`` exactly (byte-oriented,
ASCII ``\\s``/``\\b``/``\\w``, ASCII-only case folding, ``.`` stopping at
``\\n`` and ``\\r``, ``$`` only at the very end) every pattern runs on the
UTF-8 bytes of the text; results are decoded back to ``str``.

Public API: :func:`html_to_text`, :func:`parse_report_text`,
:func:`parse_report_html`, :func:`to_iso_date`, :data:`SCHEMA`.
"""

from __future__ import annotations

import re

from .corpus import PANEL_FIELDS

__all__ = ["SCHEMA", "html_to_text", "parse_report_html", "parse_report_text", "to_iso_date"]

#: The 16 panel-reviewed fields: (name, is_count, description), as schema.hpp.
SCHEMA = PANEL_FIELDS

try:  # the compiled twin (libmorie/siu, vendored from rmoriebricklayer)
    from morie import _core as _cxx

    if not hasattr(_cxx, "siu_parse_report_html"):
        _cxx = None
except ImportError:  # pragma: no cover - pure-Python install
    _cxx = None

_I = re.IGNORECASE
_META = re.compile(rb"[\^\$\.\|\?\*\+\(\)\[\]\{\}\\]")
# ECMAScript "." (std::regex) stops at \n and \r; Python's stops at \n only
_DOT = rb"[^\n\r]"


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


def _no_commas(s: bytes) -> bytes:
    return s.replace(b",", b"")


def _section_text(text: bytes, header: bytes, ends=()) -> bytes:
    """Text after the LAST line reading exactly ``header`` (the page opens with a table of
    contents repeating every title) up to the first end-marker line."""
    start = -1
    for m in re.finditer(rb"(^|\n)[ \t]*" + _esc(header) + rb"[ \t]*\n", text):
        start = m.end()
    if start < 0:
        return b""
    end = len(text)
    tail = text[start:]
    for em in ends:
        m = re.search(rb"(^|\n)[ \t]*" + _esc(em) + rb"[ \t]*\n", tail)
        if m and start + m.start() < end:
            end = start + m.start()
    return text[start:end]


def _team_count(text: bytes, label: bytes) -> bytes:
    m = re.search(rb"Number of " + label + rb"[^0-9\n]{0,30}(\d+)", text, _I)
    return m.group(1) if m else b""


_NO = rb"(?:#|n\s?o\.?|n\xc2[\xb0\xba])"


def _count_tagged(section: bytes, prefix: bytes) -> bytes:
    """Highest "PFX #n" in a section ("SO #1", "AI no 1", "AI n\xb0 1"); a bare mention is 1."""
    if not section:
        return b""
    flat = _flatten_ws(section)
    mx = max((int(m.group(1)) for m in re.finditer(rb"\b" + prefix + rb"\s*" + _NO + rb"?\s*(\d+)\b", flat)), default=0)
    if mx > 0:
        return str(mx).encode()
    return b"1" if re.search(rb"\b" + prefix + rb"\b", flat) else b""


_ROLE_HEADS = (
    rb"Subject Offic(?:ials?|ers?)",
    rb"Witness Offic(?:ials?|ers?)",
    rb"Civilian Witness(?:es)?",
    rb"Agents? impliqu\xc3\xa9(?:e|s|es)?",
    rb"Agents? t\xc3\xa9moins?",
    rb"T\xc3\xa9moins? civils?",
    rb"Incident Narrative",
    rb"Evidence",
    rb"\xc3\x89l\xc3\xa9ments de preuve",
    rb"R\xc3\xa9cit de l" + _DOT + rb"incident",
)


def _role_line(head: bytes) -> bytes:
    return rb"(^|\n)[ \t]*(?:" + head + rb")(?:[ \t]*\([ \t]*[A-Z]{2}[ \t]*\))?[ \t]*\n"


def _role_section(text: bytes, head: bytes) -> bytes:
    """The section under a role heading (LAST heading line), to the next role or evidence heading."""
    start = -1
    for m in re.finditer(_role_line(head), text):
        start = m.end()
    if start < 0:
        return b""
    end = len(text)
    tail = text[start:]
    for other in _ROLE_HEADS:
        if other == head:
            continue
        m = re.search(_role_line(other), tail)
        if m:
            end = min(end, start + m.start())
    return text[start:end]


def _count_labelled(text: bytes, label: bytes) -> bytes:
    """Highest numbered spelled-out label ("Witness Officer #3", "agent t\xe9moin no 1")."""
    flat = _flatten_ws(text)
    pat = rb"\b" + label + rb"s?\s*" + _NO + rb"\s*(\d+)\b"
    mx = max((int(m.group(1)) for m in re.finditer(pat, flat, _I)), default=0)
    return str(mx).encode() if mx > 0 else b""


_EN_NUM = (b"zero", b"one", b"two", b"three", b"four", b"five", b"six", b"seven", b"eight", b"nine", b"ten",
           b"eleven", b"twelve")  # fmt: skip
_FR_NUM = (b"z\xc3\xa9ro", b"un", b"deux", b"trois", b"quatre", b"cinq", b"six", b"sept", b"huit", b"neuf", b"dix",
           b"onze", b"douze")  # fmt: skip


def _team_words(text: bytes, what: bytes, fr: bool) -> bytes:
    """Spelled-out team sizes: "Three SIU investigators and two forensic investigators were dispatched" -> "3"."""
    names = _FR_NUM if fr else _EN_NUM
    alt = rb"\d+|une" + b"".join(b"|" + n for n in names)
    m = re.search(rb"(?:^|[^A-Za-z\xc3])(" + alt + rb")\s+" + what + rb"\b", text, _I)
    if not m:
        return b""
    w = m.group(1).lower()
    if w == b"une":
        return b"1"
    for i, n in enumerate(names):
        if w == n:
            return str(i).encode()
    return w


# ---- individual field extractors ----------------------------------------

_SVC_NOTIF = re.compile(
    rb"((?:[A-Z][A-Za-z'\-]+[ \t]+){1,5}(?:Police Service|Provincial Police|Police|Constabulary))"
    rb"\s*(?:\(\s*[A-Z]{2,6}\s*\)\s*)?(?:notified|contacted)\s+the\s+SIU"
)
_SVC_LEAD0 = re.compile(rb"^(?:The|A|An|At|On|In|By)\s+")
_SVC_NAME = re.compile(
    rb"((?:[A-Z][A-Za-z'\-]+[ \t]+){1,5}(?:Police Service|Provincial Police|Police|Constabulary))"
    rb"\b(?![ \t]+Services?[ \t]+(?:Act|Board))"
)
_SVC_LEAD = re.compile(rb"^(?:The|A|An|Of|And|On|In|By|To|With|From|That|This|Local)\s+")
_SVC_HEADLINE = re.compile(
    rb"\b(?:Between|Involving|After|During|Following|Collision|Crash|Shooting|Death|Injury|Incident|Arrest)\b"
)
_SVC_ABBR = ((b"OPP", b"Ontario Provincial Police"), (b"TPS", b"Toronto Police Service"), (b"RCMP", b"RCMP"),
             (b"NRPS", b"Niagara Regional Police Service"))  # fmt: skip


def _most_frequent(counts: dict[bytes, int]) -> bytes:
    """std::map order (sorted keys); the highest count wins, ties to the longer name."""
    best, bestc = b"", 0
    for k in sorted(counts):
        v = counts[k]
        if v > bestc or (v == bestc and len(k) > len(best)):
            best, bestc = k, v
    return best


def _detect_police_service(text: bytes) -> bytes:
    m = _SVC_NOTIF.search(text)
    if m:
        name = _trim(m.group(1))
        for _ in range(3):
            name = _SVC_LEAD0.sub(b"", name)
        if name:
            return name
    counts: dict[bytes, int] = {}
    for m in _SVC_NAME.finditer(text):
        name = _trim(m.group(1))
        for _ in range(3):
            name = _SVC_LEAD.sub(b"", name)
        if _SVC_HEADLINE.search(name):
            continue
        if name:
            counts[name] = counts.get(name, 0) + 1
    best = _most_frequent(counts)
    if best:
        return best
    for ab, full in _SVC_ABBR:
        if re.search(rb"\b" + ab + rb"\b", text):
            return full
    return b""


_INC_HEAD = re.compile(
    rb"Incident date\s*:\s*(?:[A-Z][a-z]+,?\s+)?([A-Z][a-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"
)
_INC_SECTIONS = (b"Incident Narrative", b"The Investigation", b"Analysis and Director's Decision",
                 b"Analysis and Director\xe2\x80\x99s Decision")  # fmt: skip
_INC_ENDS = (b"Nature of Injuries", b"Evidence", b"The Team", b"Analysis and Director", b"Relevant Legislation",
             b"Conclusion")  # fmt: skip
_INC_ON = re.compile(rb"\b(?:[Oo]n|of)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})")
_INC_OWN = re.compile(
    rb"notified|contacted the siu|notification of the siu|siu investigators|forensic investigators?|"
    rb"were assigned|was assigned|designated as|provided (?:a|his|her|their) statement|interviewed|"
    rb"the siu arrived|director"
)
_EN_MONTHS = re.compile(rb"January|February|March|April|May|June|July|August|September|October|November|December")


def _detect_incident_date(text: bytes) -> bytes:
    """A legacy header's "Incident date:", else the first "On <date>" of the narrative whose
    sentence is not the notification or the SIU's own work."""
    m = _INC_HEAD.search(text)
    if m:
        return _no_commas(m.group(1))
    for sec_name in _INC_SECTIONS:
        sec = _section_text(text, sec_name, _INC_ENDS)
        if not sec:
            continue
        for it in _INC_ON.finditer(sec):
            pos = it.start()
            a = sec.rfind(b". ", 0, pos + 2)
            nl = sec.rfind(b"\n", 0, pos + 1)
            a = 0 if a < 0 else a + 2
            if nl >= 0 and nl + 1 > a:
                a = nl + 1
            b = sec.find(b". ", pos + len(it.group(0)))
            nl2 = sec.find(b"\n", pos)
            if b < 0 or (nl2 >= 0 and nl2 < b):
                b = nl2
            if b < 0:
                b = len(sec)
            if _INC_OWN.search(sec[a:b].lower()):
                continue
            g = it.group(1)
            sp = g.find(b" ")
            if not _EN_MONTHS.fullmatch(g if sp < 0 else g[:sp]):
                continue
            return _no_commas(g)
    return b""


_DATE_EN = rb"([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"
_NOTIF_A = re.compile(
    rb"\b[Oo]n\s+(?:[A-Z][a-z]+,?\s+)?" + _DATE_EN + rb"[^\n]{0,200}?(?:notified|contacted)\s+the\s+SIU"
)
_NOTIF_B = re.compile(rb"(?:notified|contacted)\s+the\s+SIU[^\n]{0,200}?[Oo]n\s+" + _DATE_EN)
_NOTIF_B2 = re.compile(rb"SIU\s+was\s+(?:notified|contacted)[^\n]{0,200}?\b[Oo]n\s+(?:[A-Z][a-z]+,?\s+)?" + _DATE_EN)
_NOTIF_C = re.compile(rb"On\s+" + _DATE_EN)


def _detect_siu_notified(text: bytes) -> bytes:
    inv = _section_text(text, b"The Investigation", (b"The Team", b"Incident Narrative"))
    hay = inv or text
    for pat in (_NOTIF_A, _NOTIF_B, _NOTIF_B2):
        m = pat.search(hay)
        if m:
            return _no_commas(m.group(1))
    notif = _section_text(text, b"Notification of the SIU", (b"The Team", b"Incident Narrative", b"Evidence"))
    m = _NOTIF_C.search(notif) if notif else None
    return _no_commas(m.group(1)) if m else b""


def _detect_decision_date(text: bytes) -> bytes:
    m = re.search(rb"Date:\s*" + _DATE_EN, text)
    if m:
        return _no_commas(m.group(1))
    m = re.search(rb"Date:\s*(\d{4}-\d{2}-\d{2})", text)
    return m.group(1) if m else b""


_LOCATION = re.compile(
    rb"in the (Township|City|Town|Municipality|Region) of ([A-Z][A-Za-z \-]+?)(?:[\.,]|\s+(?:on|at|when))"
)


def _detect_location(text: bytes) -> bytes:
    inv = _section_text(text, b"The Investigation", (b"The Team", b"Incident Narrative"))
    m = _LOCATION.search(_flatten_ws(inv or text))
    return m.group(1) + b" of " + _trim(m.group(2)) if m else b""


_AGE_SEX = re.compile(
    rb"\b(\d{1,3})[\s\-]year[\s\-]old\s+(woman|man|female|male|girl|boy|person|individual|youth|child|adult)\b", _I
)


def _detect_age_sex(text: bytes) -> tuple[bytes, bytes]:
    m = _AGE_SEX.search(text)
    return (m.group(1), m.group(2).lower()) if m else (b"", b"")


_INJ_EN = re.compile(
    rb"\b((?:fracture[sd]?|broken|lacerat\w*|gunshot|stab(?:bed|bing| wounds?)?|burns?)\b[^\n.]{0,200}?"
    rb"\b(?:ribs?|legs?|arms?|skull|wrists?|ankles?|jaw|nose|teeth|tooth|fingers?|spine|vertebrae?|shoulders?|"
    rb"hips?|pelvis|orbit(?:al)?|face|hands?|feet|foot|elbows?|knees?|clavicle|collarbone|femur|humerus|tibia|"
    rb"fibula|sternum|neck|back|head|chest|abdomen)\b[^\n.]{0,80})",
    _I,
)
_INJ_FR = re.compile(
    rb"\b((?:fractures?|fractur\xc3\xa9e?s?|lac\xc3\xa9rations?|blessures? par balle|coups? de couteau|"
    rb"br\xc3\xbblures?)[^\n.]{0,200}?(?:\b|(?=\xc3))(?:c\xc3\xb4tes?|jambes?|bras|cr\xc3\xa2ne|poignets?|"
    rb"chevilles?|m\xc3\xa2choire|nez|dents?|doigts?|colonne|vert\xc3\xa8bres?|\xc3\xa9paules?|hanches?|bassin|"
    rb"orbite|visage|mains?|pieds?|coudes?|genoux|clavicule|f\xc3\xa9mur|tibia|p\xc3\xa9ron\xc3\xa9|sternum|cou|"
    rb"dos|t\xc3\xaate|thorax|abdomen)[^\n.]{0,80})",
    _I,
)
_INJ_HEAD_EN = re.compile(rb"(^|\n)[ \t]*The Investigation[ \t]*\n")
_INJ_HEAD_FR = re.compile(rb"(^|\n)[ \t]*L(?:'|\xe2\x80\x99)enqu\xc3\xaate[ \t]*\n")


def _detect_specific_injuries(text: bytes, fr: bool) -> bytes:
    """The injury as the narrative states it, from the investigation on, skipping any sentence
    that quotes the legal definition of a serious injury."""
    start = 0
    for m in (_INJ_HEAD_FR if fr else _INJ_HEAD_EN).finditer(text):
        start = m.end()
    hay = text[start:]
    for it in (_INJ_FR if fr else _INJ_EN).finditer(hay):
        pos = it.start()
        a, b = hay.rfind(b"\n", 0, pos + 1), hay.find(b"\n", pos)
        line = hay[a + 1 if a >= 0 else 0 : b if b >= 0 else len(hay)].lower()
        if b"serious injury" in line or b"blessure grave" in line:
            continue
        return _trim(it.group(1))
    return b""


_LEG_EN = re.compile(rb"Section\s+\d+(?:\.\d+)*(?:\([^)]+\))?,?\s+([A-Z][^\n,]{2,80}?)(?:\s*[-]|\s*\Z|\n)")
_LEG_FR = re.compile(
    rb"(?:Articles?|Paragraphes?|Alin\xc3\xa9as?)\s+\d+[^\s]*\s+(?:du |de la |de l'|de l\xe2\x80\x99|des )"
    rb"([A-Z\xc3][^\n,]{2,80}?)(?:\s*--|\s+-|\s*\Z|\n)"
)


def _detect_legislation(text: bytes, fr: bool) -> bytes:
    if fr:
        sec = _section_text(text, b"Dispositions l\xc3\xa9gislatives pertinentes",
                            (b"Analyse et d\xc3\xa9cision du directeur", b"Communiqu\xc3\xa9s de presse"))  # fmt: skip
    else:
        sec = _section_text(text, b"Relevant Legislation", (b"Analysis and Director", b"News Releases"))
    if not sec:
        return b""
    acts: list[bytes] = []
    for m in (_LEG_FR if fr else _LEG_EN).finditer(sec):
        act = _trim(m.group(1))
        if act not in acts:
            acts.append(act)
    return b"; ".join(acts)


_CHARGE_NO_FR = (b"aucun motif raisonnable", b"pas lieu de porter", b"aucune accusation", b"ne porterai pas",
                 b"ne sera port\xc3\xa9e", b"dossier est clos")  # fmt: skip
_CHARGE_YES_FR = (b"a \xc3\xa9t\xc3\xa9 accus\xc3\xa9", b"accusations ont \xc3\xa9t\xc3\xa9 port\xc3\xa9es",
                  b"accusation a \xc3\xa9t\xc3\xa9 port\xc3\xa9e", b"inculp\xc3\xa9")  # fmt: skip
_CHARGE_NO_EN = (b"no charges", b"shall issue", b"none shall issue", b"no basis for charges", b"no reasonable grounds",
                 b"do not lay", b"decline to lay", b"lack the necessary grounds")  # fmt: skip
_CHARGE_YES_EN = (b"charged with", b"criminal charges have been laid", b"charges have been laid")


def _detect_charges(text: bytes, fr: bool) -> bytes:
    if fr:
        sec = _section_text(text, b"Analyse et d\xc3\xa9cision du directeur",
                            (b"Notes de fin", b"Communiqu\xc3\xa9", b"Remarque :"))  # fmt: skip
        no, yes = _CHARGE_NO_FR, _CHARGE_YES_FR
    else:
        ends = (b"Endnotes", b"News Release", b"Note:")
        sec = _section_text(text, b"Analysis and Director's Decision", ends) or _section_text(
            text, b"Analysis and Director\xe2\x80\x99s Decision", ends
        )
        no, yes = _CHARGE_NO_EN, _CHARGE_YES_EN
    if not sec:
        return b""
    low = sec.lower()
    if any(p in low for p in no):
        return b"FALSE"
    if any(p in low for p in yes):
        return b"TRUE"
    return b""


_DIR_OWN_LINE = re.compile(
    rb"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})[ \t]*\n(?:[ \t]*\n)*[ \t]*(?:(?:Interim|Acting)[ \t]+)?"
    rb"(?:Director|Directeur|Directrice)(?:[ \t]+par[ \t]+int\xc3\xa9rim)?[ \t]*(?:\n|\Z)"
)
_DIR_COMMA = re.compile(
    rb"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})[ \t]*,[ \t]*(?:(?:Interim|Acting)[ \t]+)?"
    rb"(?:Director|Directeur|Directrice)[ \t]*(?:\n|\Z)"
)


def _detect_directors_name(text: bytes) -> bytes:
    """Signature block: "<Name>\\n[\\n]Director" or "<Name>, Director" ("Director" alone on its line)."""
    for rx in (_DIR_OWN_LINE, _DIR_COMMA):
        for m in rx.finditer(text):
            name = _trim(m.group(1))
            if b"the" not in name.lower():
                return name
    return b""


_LANG_EN = (b"The Investigation", b"Notification of the SIU", b"Mandate engaged", b"Civilian Witnesses",
            b"Witness Officers", b"Subject Officers", b"Analysis and Director's Decision", b"Witness Officials",
            b"Police service:", b"Incident date:", b"Witness Officer #", b"Civilian Witness #",
            b"Subject Officer #")  # fmt: skip
_LANG_FR = (b"L'enqu\xc3\xaate", b"L\xe2\x80\x99enqu\xc3\xaate", b"Exercice du mandat", b"\xc3\x89l\xc3\xa9ments de preuve",
            b"Dispositions l\xc3\xa9gislatives pertinentes", b"T\xc3\xa9moins civils", b"Agents impliqu\xc3\xa9s",
            b"Mandat de l'UES", b"Mandat de l\xe2\x80\x99UES", b"Service de police :", b"Agents t\xc3\xa9moins")  # fmt: skip


def _detect_language(text: bytes) -> bytes:
    e = sum(mk in text for mk in _LANG_EN)
    f = sum(mk in text for mk in _LANG_FR)
    if e >= 2 and e > f:
        return b"en"
    if f >= 2 and f > e:
        return b"fr"
    return b"unknown"


# ---- dates ----------------------------------------------------------------

_MONTH_NAMES = (
    b"january", b"february", b"march", b"april", b"may", b"june", b"july", b"august", b"september", b"october",
    b"november", b"december",
    b"janvier", b"f\xc3\xa9vrier", b"fevrier", b"mars", b"avril", b"mai", b"juin", b"juillet", b"ao\xc3\xbbt",
    b"aout", b"septembre", b"octobre", b"novembre", b"d\xc3\xa9cembre", b"decembre",
    b"jan", b"feb", b"mar", b"apr", b"jun", b"jul", b"aug", b"sep", b"sept", b"oct", b"nov", b"dec",
    b"janv", b"f\xc3\xa9vr", b"fevr", b"avr", b"juil", b"d\xc3\xa9c",
)  # fmt: skip
_MONTH_NUMS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12,
               1, 2, 3, 4, 6, 7, 8, 9, 9, 10, 11, 12,
               1, 2, 2, 4, 7, 12)  # fmt: skip
_MONTHS = dict(zip(_MONTH_NAMES, _MONTH_NUMS))
_ISO_SEP_ER = re.compile(rb"(\d)(?:\s|\xc2\xa0)+er\b")
_ISO_MD = re.compile(rb"([^\s\d,.]+)\.?\s+(\d{1,2})(?:st|nd|rd|th|er|e)?\s*,?\s+(\d{4})")  # August 3rd, 2017
_ISO_DM = re.compile(rb"(\d{1,2})(?:er|e|st|nd|rd|th)?\s+(?:of\s+)?([^\s\d,.]+)\.?,?\s+(\d{4})")  # 3 août 2017
_ISO = re.compile(rb"\d{4}-\d{2}-\d{2}")
_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _month_key(w: bytes) -> bytes:
    """ASCII lower-case plus the Latin-1 accented capitals ("AOÛT" -> "août")."""
    k = bytearray(w.lower())
    for i in range(len(k) - 1):
        if k[i] == 0xC3 and 0x80 <= k[i + 1] <= 0x9E and k[i + 1] != 0x97:
            k[i + 1] += 0x20
    return bytes(k)


def _iso(human: bytes) -> bytes:
    human = _ISO_SEP_ER.sub(rb"\g<1>er", human)
    m = _ISO_MD.search(human)
    if m and _month_key(m.group(1)) in _MONTHS:
        month, day, year = m.group(1), m.group(2), m.group(3)
    else:
        m = _ISO_DM.search(human)
        if m and _month_key(m.group(2)) in _MONTHS:
            day, month, year = m.group(1), m.group(2), m.group(3)
        else:
            return human if _ISO.fullmatch(human) else b""
    y, mo, d = int(year), _MONTHS[_month_key(month)], int(day)
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    if d < 1 or d > _DAYS[mo - 1] + (1 if mo == 2 and leap else 0):
        return b""  # "February 30, 2019" is not a date
    return year + b"-%02d-%02d" % (mo, d)


def to_iso_date(human: str) -> str:
    """``"January 5, 2023"`` -> ``"2023-01-05"`` (``""`` if unparseable).

    Examples
    --------
    >>> to_iso_date("January 5, 2023"), to_iso_date("2023-01-05"), to_iso_date("soon")
    ('2023-01-05', '2023-01-05', '')
    """
    if _cxx is not None:
        return _cxx.siu_to_iso_date(human)
    return _s(_iso(_b(human)))


# ---- HTML -> text ---------------------------------------------------------

_HTML_STEPS = (
    (re.compile(rb"\r\n?"), b"\n"),  # live pages are CRLF; \r defeats line-anchored rules
    (re.compile(rb"<(script|style)[^>]*>[\s\S]*?</\1>", _I), b" "),
    (re.compile(rb"</(p|div|h[1-6]|tr|li|br)>|<br\s*/?>", _I), b"\n"),
    (re.compile(rb"<[^>]+>"), b" "),
    (re.compile(rb"&nbsp;"), b" "),
    (re.compile(rb"&amp;"), b"&"),
    (re.compile(rb"&#8217;|&rsquo;|&#x2019;", _I), b"'"),
    (re.compile(rb"&#8216;|&lsquo;|&#x2018;", _I), b"'"),
    (re.compile(rb"&#8220;|&ldquo;|&#8221;|&rdquo;|&#x201c;|&#x201d;", _I), b'"'),
    (re.compile(rb"&quot;"), b'"'),
    (re.compile(rb"&#0?39;|&apos;"), b"'"),
    (re.compile(rb"&#8211;|&ndash;|&#x2013;", _I), b"-"),
    (re.compile(rb"&#8212;|&mdash;|&#x2014;", _I), b"--"),
)
# accented named entities of the French pages, as UTF-8
_NAMED = (
    (b"&eacute;", b"\xc3\xa9"), (b"&egrave;", b"\xc3\xa8"), (b"&ecirc;", b"\xc3\xaa"), (b"&euml;", b"\xc3\xab"),
    (b"&agrave;", b"\xc3\xa0"), (b"&acirc;", b"\xc3\xa2"), (b"&ccedil;", b"\xc3\xa7"), (b"&icirc;", b"\xc3\xae"),
    (b"&iuml;", b"\xc3\xaf"), (b"&ocirc;", b"\xc3\xb4"), (b"&ouml;", b"\xc3\xb6"), (b"&ucirc;", b"\xc3\xbb"),
    (b"&ugrave;", b"\xc3\xb9"), (b"&uuml;", b"\xc3\xbc"), (b"&auml;", b"\xc3\xa4"), (b"&copy;", b"\xc2\xa9"),
    (b"&reg;", b"\xc2\xae"), (b"&Eacute;", b"\xc3\x89"), (b"&Agrave;", b"\xc3\x80"), (b"&Egrave;", b"\xc3\x88"),
    (b"&Ecirc;", b"\xc3\x8a"), (b"&Ccedil;", b"\xc3\x87"), (b"&Ocirc;", b"\xc3\x94"), (b"&Icirc;", b"\xc3\x8e"),
    (b"&Acirc;", b"\xc3\x82"), (b"&laquo;", b"\xc2\xab"), (b"&raquo;", b"\xc2\xbb"), (b"&oelig;", b"\xc5\x93"),
    (b"&OElig;", b"\xc5\x92"), (b"&thinsp;", b" "),
)  # fmt: skip
_NUMERIC_ENTITY = re.compile(rb"&#(x[0-9A-Fa-f]+|[0-9]+);")


def _utf8_bytes(cp: int) -> bytes:
    """The code point as UTF-8 bytes, surrogates included (as the C++ writes them)."""
    if cp < 0x80:
        return bytes((cp,))
    if cp < 0x800:
        return bytes((0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)))
    if cp < 0x10000:
        return bytes((0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)))
    return bytes((0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)))


def _numeric_entity(m: re.Match) -> bytes:
    g = m.group(1)
    cp = int(g[1:], 16) if g[:1] == b"x" else int(g)
    return b"" if cp == 0 or cp > 0x10FFFF else _utf8_bytes(cp)


def _html_to_text(html: bytes) -> bytes:
    t = html
    for pat, rep in _HTML_STEPS:
        t = pat.sub(rep, t)
    for ent, ch in _NAMED:
        t = t.replace(ent, ch)
    t = re.sub(rb"&hellip;|&#8230;", b"...", t)
    t = _NUMERIC_ENTITY.sub(_numeric_entity, t)
    # angle brackets last: the markup is gone, so a decoded "<" is never read as a tag
    t = t.replace(b"&lt;", b"<").replace(b"&gt;", b">")
    t = re.sub(rb"[ \t]+", b" ", t)
    t = re.sub(rb" ?\n ?", b"\n", t)
    return re.sub(rb"\n{3,}", b"\n\n", t)


def html_to_text(html: str) -> str:
    """Strip tags, scripts and entities from report HTML, keeping line structure.

    Examples
    --------
    >>> html_to_text("<p>The&nbsp;Team</p><p>SO #1</p>")
    ' The Team\\nSO #1\\n'
    """
    if _cxx is not None:
        return _cxx.siu_html_to_text(html)
    return _s(_html_to_text(_b(html)))


# ---- French reports: the SIU publishes every report in both languages, and the French copy has its
# ---- own headings and phrasing ("Le 12 novembre 2022", "a communiqué ... à l'UES")

_FR_MONTHS = (
    rb"(?:janvier|f\xc3\xa9vrier|fevrier|mars|avril|mai|juin|juillet|ao\xc3\xbbt|aout|septembre|octobre|novembre|"
    rb"d\xc3\xa9cembre|decembre)"
)
_FR_DATE = rb"(\d{1,2}(?:(?:\s|\xc2\xa0)?er)?\s+" + _FR_MONTHS + rb"\s+\d{4})"
_APOS = rb"(?:'|\xe2\x80\x99)"


def _sentence_at(s: bytes, pos: int) -> bytes:
    """The sentence from ``pos`` to its first ". " or the end of the line (at most 300 bytes)."""
    end = s.find(b"\n", pos)
    stop = s.find(b". ", pos)
    if stop >= 0 and (end < 0 or stop < end):
        end = stop
    if end < 0 or end > pos + 300:
        end = min(len(s), pos + 300)
    return s[pos:end]


_SVC_NOTIF_FR = re.compile(
    rb"(Service de police[^\n,.;()]*?|Police provinciale de l" + _APOS + rb"Ontario)\s*(?:\(\s*[A-Z]{2,6}\s*\)\s*)?"
    rb"(?:a|ont) (?:communiqu|avis|inform)"
)
_SVC_NAME_FR = re.compile(
    rb"(Service de police(?: [a-z\xc3\xa9]+){0,2} (?:de la |de |du |des |d" + _APOS + rb")[A-Z\xc3][^\s,.;()]*"
    rb"(?: [A-Z\xc3][^\s,.;()]*)*|Police provinciale de l" + _APOS + rb"Ontario)"
)


def _detect_police_service_fr(text: bytes) -> bytes:
    m = _SVC_NOTIF_FR.search(text)
    if m:
        return _trim(m.group(1))
    counts: dict[bytes, int] = {}
    for m in _SVC_NAME_FR.finditer(text):
        k = _trim(m.group(1))
        counts[k] = counts.get(k, 0) + 1
    return _most_frequent(counts)


_INC_HEAD_FR = re.compile(rb"Date de l" + _APOS + rb"incident\s*:\s*(?:[Ll]e\s+)?" + _FR_DATE)
_INC_LE_FR = re.compile(rb"\b(?:[Ll]e|du)\s+" + _FR_DATE)
_UES = re.compile(rb"l" + _APOS + rb"\s*UES\b")
_LOGISTICS = re.compile(rb"entrevue|Date et heure|\xc3\xa9quipe|enqu\xc3\xaateurs", _I)


def _detect_incident_date_fr(text: bytes) -> bytes:
    """First "Le <date>" of the investigation that is not the call to the SIU or the team's dispatch."""
    m = _INC_HEAD_FR.search(text)
    if m:
        return m.group(1)
    sec = _section_text(text, b"L\xe2\x80\x99enqu\xc3\xaate") or _section_text(text, b"L'enqu\xc3\xaate") or text
    for it in _INC_LE_FR.finditer(sec):
        p0 = it.start()
        sent = _sentence_at(sec, p0)
        a = p0 - 60 if p0 > 60 else 0
        before = sec[a:p0].lower()
        ls, le = sec.rfind(b"\n", 0, p0 + 1), sec.find(b"\n", p0)
        line = sec[ls + 1 if ls >= 0 else 0 : le if le >= 0 else len(sec)]
        if (
            _UES.search(sent)
            or _LOGISTICS.search(sent)
            or _LOGISTICS.search(line)
            or _UES.search(line)
            or b"envoi de l" in before
            or b"arriv\xc3\xa9e de l" in before
        ):
            continue
        return it.group(1)
    return b""


_NOTIF_HEAD_FR = re.compile(rb"(^|\n)Notification de l" + _APOS + rb"\s*UES[^\n]*\n")
_NOTIF_TOLD_FR = re.compile(rb"UES\s+a\s+\xc3\xa9t\xc3\xa9\s+avis\xc3\xa9e[^\n]{0,200}?\b(?:le|du)\s+" + _FR_DATE)
_FR_DATE_RX = re.compile(_FR_DATE)


def _detect_siu_notified_fr(text: bytes) -> bytes:
    """First date after the "Notification de l'UES" heading."""
    start = -1
    for m in _NOTIF_HEAD_FR.finditer(text):
        start = m.end()
    if start < 0:
        m = _NOTIF_TOLD_FR.search(text)
        return m.group(1) if m else b""
    m = _FR_DATE_RX.search(text[start : start + 1500])
    return m.group(1) if m else b""


def _detect_decision_date_fr(text: bytes) -> bytes:
    m = re.search(rb"Date\s*:\s*(?:[Ll]e\s+)?" + _FR_DATE, text)
    return m.group(1) if m else b""


_AGE_SEX_FR = re.compile(
    rb"\b(femme|homme|fille|gar\xc3\xa7on|adolescente|adolescent|personne)\s+de\s+(\d{1,3})\s+ans\b", _I
)
_SEX_FR_EN = {b"femme": b"woman", b"homme": b"man", b"fille": b"girl", b"gar\xc3\xa7on": b"boy",
              b"adolescente": b"youth", b"adolescent": b"youth", b"personne": b"person"}  # fmt: skip


def _detect_age_sex_fr(text: bytes) -> tuple[bytes, bytes]:
    m = _AGE_SEX_FR.search(text)
    if not m:
        return b"", b""
    w = m.group(1).lower()
    return m.group(2), _SEX_FR_EN.get(w, w)


# ---- the subject officials' service -------------------------------------
# police_service is the service whose officers are the subject officials. The force that notified the
# SIU is often another one, so the director's analysis decides: the first service named in a sentence
# that names a subject official, else the service the analysis names most. The case number's letter
# (T Toronto, P OPP, I First Nations, O any other) rules out services of the wrong kind. Legacy reports
# name the service in a "Police service:" header.

_CAP = rb"(?:[A-Z]|\xc3[\x80-\x9d])"
_WORD = rb"[^\s,.;()]*"
_EN_NAME = (
    rb"((?:[A-Z][A-Za-z'\-]+[ \t]+){1,5}(?:Police Service|Police Department|Provincial Police|Police|Constabulary))"
    rb"\b(?![ \t]+Services?[ \t]+(?:Act|Board))"
)
_FR_NAME = (
    rb"([Ss]ervice(?: [a-z\xc3\xa9]+){0,2} de (?:la )?police(?: [a-z\xc3\xa9]+){0,2} (?:de la |de |du |des |d"
    + _APOS + rb")(?:grand )?" + _CAP + _WORD + rb"(?: " + _CAP + _WORD + rb")*"
    + rb"|[Ss]ervice de police (?:Nishnawbe[- ]Aski|Anishinabek|Akwesasne|Wikwemikong|UCCM)"
    + rb"|[Pp]olice r\xc3\xa9gionale (?:de |du |d" + _APOS + rb")" + _CAP + _WORD + rb"(?: " + _CAP + _WORD + rb")*"
    + rb"|[Pp]olice [Pp]rovinciale(?: de l" + _APOS + rb"Ontario)?"
    + rb"|[Pp]olice (?:de |d" + _APOS + rb")" + _CAP + _WORD + rb"(?: " + _CAP + _WORD + rb")*)"
)  # fmt: skip
_EN_NAME_RX = re.compile(_EN_NAME)
_FR_NAME_RX = re.compile(_FR_NAME)
_EN_ABBR_RX = re.compile(_EN_NAME + rb"\s*\(\s*([A-Z]{2,6})\s*\)")
_FR_ABBR_RX = re.compile(_FR_NAME + rb"\s*\(\s*([A-Z]{2,6})\s*\)")
_KIND_FN = re.compile(
    rb"Nishnawbe|Anishinabek|Treaty|Trait\xc3\xa9|Akwesasne|Wikwemikong|UCCM|Lac Seul|Rama|"
    rb"Six Nations|Premi\xc3\xa8res? Nations?|First Nations?"
)
_KIND_PROV = re.compile(rb"Provin\w*al|[Pp]rovinciale")
_CLEAN_LEAD = re.compile(rb"^(?:The|A|An|Of|And|On|In|By|To|With|From|That|This|Local|While|When|As)\s+")
_OPP_FR = re.compile(rb"[Pp]olice [Pp]rovinciale\b" + _DOT + rb"*")
_OPP_EN = re.compile(rb"(?:OPP|Ontario Provincial Police)\b" + _DOT + rb"*")
_NOT_SERVICE = re.compile(
    rb"\b(?:Independent|Review|Office|Between|Involving|After|During|Following|Collision|"
    rb"Crash|Shooting|Death|Injury|Incident|Arrest)\b"
)
_SO_EN = re.compile(rb"\bSOs?\b|[Ss]ubject [Oo]ffic(?:er|ial)s?")
_SO_FR = re.compile(rb"\bAIs?\b|agente?s? impliqu")
_NOTIFIER_EN = re.compile(
    rb"^[^.,;]{0,60}?\b(?:notified|contacted|alerted|advised|called)\s+the\s+SIU\b|"
    rb"^[^.,;]{0,60}?\breported\s+(?:to\s+the\s+SIU\b|that\b)"
)
_NOTIFIER_FR = re.compile(
    rb"^[^.,;]{0,60}?\ba\s+(?:avis\xc3\xa9|inform\xc3\xa9|signal\xc3\xa9|communiqu\xc3\xa9\s+avec)\s+l(?:'|\xe2\x80\x99)\s?UES"
)
_ABBR_FR = ((b"PPO", b"Police provinciale de l'Ontario"), (b"SPT", b"Service de police de Toronto"),
            (b"PRY", b"Police r\xc3\xa9gionale de York"), (b"PRP", b"Police r\xc3\xa9gionale de Peel"),
            (b"SPRP", b"Service de police r\xc3\xa9gional de Peel"),
            (b"SPRD", b"Service de police r\xc3\xa9gional de Durham"),
            (b"SPRN", b"Service de police r\xc3\xa9gional de Niagara"),
            (b"SPRH", b"Service de police r\xc3\xa9gional de Halton"),
            (b"SPRW", b"Service de police r\xc3\xa9gional de Waterloo"))  # fmt: skip
_ABBR_EN = ((b"OPP", b"Ontario Provincial Police"), (b"TPS", b"Toronto Police Service"),
            (b"YRP", b"York Regional Police"), (b"PRP", b"Peel Regional Police"),
            (b"DRPS", b"Durham Regional Police Service"), (b"NRPS", b"Niagara Regional Police Service"),
            (b"HRPS", b"Halton Regional Police Service"), (b"WRPS", b"Waterloo Regional Police Service"))  # fmt: skip


def _service_kind(n: bytes) -> bytes:
    if _KIND_FN.search(n):
        return b"I"
    if _KIND_PROV.search(n):
        return b"P"
    return b"T" if b"Toronto" in n else b"O"


def _clean_service(n: bytes) -> bytes:
    n = _trim(n)
    for _ in range(3):
        n = _CLEAN_LEAD.sub(b"", n)
    # one name per service: "la Police provinciale" back-references and detachment headers are the OPP
    if _OPP_FR.fullmatch(n):
        return b"Police provinciale de l'Ontario"
    if _OPP_EN.fullmatch(n):
        return b"Ontario Provincial Police"
    return n


def _is_word(c: int) -> bool:
    return 65 <= c <= 90 or 97 <= c <= 122 or 48 <= c <= 57 or c == 95


def _pick_service(sec: bytes, text: bytes, fr: bool, letter: bytes) -> bytes:
    abbr = [list(p) for p in (_ABBR_FR if fr else _ABBR_EN)]
    # the page's own "<name> ( ABBR )" pairs override the defaults; the first definition wins
    seen: set[bytes] = set()
    for m in (_FR_ABBR_RX if fr else _EN_ABBR_RX).finditer(text):
        k = m.group(m.re.groups)
        if k in seen:
            continue
        seen.add(k)
        v = _clean_service(m.group(1))
        hit = [p for p in abbr if p[0] == k]
        for p in hit:
            p[1] = v
        if not hit:
            abbr.append([k, v])
    ments: list[tuple[int, bytes]] = []
    for m in (_FR_NAME_RX if fr else _EN_NAME_RX).finditer(sec):
        n = _clean_service(m.group(1))
        if n and not _NOT_SERVICE.search(n):
            ments.append((m.start(), n))
    for k, v in abbr:
        if not v or _NOT_SERVICE.search(v):
            continue
        at = sec.find(k)
        while at >= 0:
            if (at == 0 or not _is_word(sec[at - 1])) and (at + len(k) == len(sec) or not _is_word(sec[at + len(k)])):
                ments.append((at, v))
            at = sec.find(k, at + 1)
    ments.sort()
    if letter:
        keep = [m for m in ments if _service_kind(m[1]) == letter]
        if keep:
            ments = keep
    if not ments:
        return b""
    ments = [(p, n[:1].upper() + n[1:] if 97 <= n[0] <= 122 else n) for p, n in ments]
    notif = _NOTIFIER_FR if fr else _NOTIFIER_EN

    def notifier(pos: int) -> bool:
        # the service that notified the SIU is the notifier, not the subject officials' service
        return bool(notif.search(sec[pos : pos + 160]))

    for it in (_SO_FR if fr else _SO_EN).finditer(sec):
        pos = it.start()
        dot = -1 if pos == 0 else sec.rfind(b".", 0, pos)
        a = 0 if dot < 0 else dot + 1
        nd = sec.find(b".", pos)
        b = len(sec) if nd < 0 else nd
        for p, n in ments:
            if a <= p <= b and not notifier(p):
                return n
        # "... arrest by NRPS officers. The SIU named the SO ..." -- the sentence before names it
        if a > 0:
            pdot = sec.rfind(b".", 0, a - 1) if a >= 2 else -1
            pa = 0 if pdot < 0 else pdot + 1
            for p, n in ments:
                if pa <= p < a and not notifier(p):
                    return n
    counts: dict[bytes, int] = {}
    for _, n in ments:
        counts[n] = counts.get(n, 0) + 1
    best = ments[0]
    for m in ments:
        if counts[m[1]] > counts[best[1]]:
            best = m
    return best[1]


_LEGACY_SVC_EN = re.compile(rb"Police service\s*:\s*(" + _DOT + rb"+?)\s+Incident date")
_LEGACY_SVC_FR = re.compile(rb"Service de police\s*:\s*(" + _DOT + rb"+?)\s+Date de l")
_CASE_NO = re.compile(rb"\b\d\d-([TPOI])[A-Z]{2}-\d{3}\b")


def _detect_subject_service(text: bytes, fr: bool) -> bytes:
    m = (_LEGACY_SVC_FR if fr else _LEGACY_SVC_EN).search(text)
    if m:
        h = _trim(m.group(1)).lower()
        for it in (_FR_NAME_RX if fr else _EN_NAME_RX).finditer(text):
            n = _clean_service(it.group(1))
            if h in n.lower():
                return n
        return _clean_service(m.group(1))
    m = _CASE_NO.search(text)
    letter = m.group(1) if m else b""
    head = b"analyse et d\xc3\xa9cision du directeur" if fr else b"analysis and director"
    i = text.lower().rfind(head)
    r = _pick_service(text[i:], text, fr, letter) if i >= 0 else b""
    return r or _pick_service(text, text, fr, letter)


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Days since 1970-01-01 (Hinnant), as the C++ computes them: no time zones involved."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _civil_from_days(z: int) -> bytes:
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return b"%04d-%02d-%02d" % (yoe + era * 400 + (m <= 2), m, d)


_REL_NOTIF_EN = re.compile(
    rb"\b(?:notified|contacted|advised|alerted|reported)\b[^.]{0,40}\bSIU\b|\bSIU\b[^.]{0,20}\bnotified\b"
)
_REL_NOTIF_FR = re.compile(rb"\ba\s+(?:avis\xc3\xa9|inform\xc3\xa9|communiqu\xc3\xa9)[^.]{0,40}UES")
_REL_PREV_EN = re.compile(
    rb"\bthe (?:day|night|evening|morning) before\b|\bthe previous (?:day|night|evening|morning)\b", _I
)
_REL_SAME_EN = re.compile(
    rb"\b(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|a few)\s+"
    rb"(?:hours?|minutes?)\s+(?:prior|earlier|before)\b|\bearlier (?:that|the same) (?:day|morning|afternoon|evening|night)\b|"
    rb"\bthe same (?:day|morning|afternoon|evening)\b",
    _I,
)
_REL_PREV_FR = re.compile(
    rb"\bla veille\b|\ble jour pr\xc3\xa9c\xc3\xa9dent\b|\bla nuit pr\xc3\xa9c\xc3\xa9dente\b", _I
)
_REL_SAME_FR = re.compile(
    rb"\b(?:\d+|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|plusieurs|quelques)\s+(?:heures?|minutes?)\s+plus\s+t\xc3\xb4t|"
    rb"plus\s+t\xc3\xb4t\s+(?:ce jour-l\xc3\xa0|dans la journ\xc3\xa9e|ce matin-l\xc3\xa0|ce soir-l\xc3\xa0)|\ble m\xc3\xaame jour\b",
    _I,
)


def _relative_incident(text: bytes, notified_iso: bytes, fr: bool) -> bytes:
    """The incident date a notification states relative to itself ("two hours prior": the same day,
    "the day before"); "" when no notification sentence says."""
    if len(notified_iso) != 10:
        return b""
    notif, prev, same = (
        (_REL_NOTIF_FR, _REL_PREV_FR, _REL_SAME_FR) if fr else (_REL_NOTIF_EN, _REL_PREV_EN, _REL_SAME_EN)
    )
    flat = _flatten_ws(text)
    shift = 1  # 1: none found
    start = 0
    while start < len(flat):
        end = flat.find(b". ", start)
        if end < 0:
            end = len(flat)
        sent = flat[start:end]
        start = end + 2
        if not notif.search(sent):
            continue
        if prev.search(sent):
            shift = -1
            break
        if same.search(sent):
            shift = 0
            break
    if shift == 1:
        return b""
    y, m, d = int(notified_iso[:4]), int(notified_iso[5:7]), int(notified_iso[8:10])
    return _civil_from_days(_days_from_civil(y, m, d) + shift)


# ---- the subject-official resolver (siu_resolve.cpp) -----------------------

_BOILER = re.compile(rb"this information may include[\s\S]*?(?:affected person|evidence)\.?", _I)
_GLOSSARY = re.compile(
    rb"who,?\s+in the (?:opinion of the SIU Director|SIU Director(?:'|\xe2\x80\x99)?s opinion),?"
    rb"[\s\S]{0,120}?not a subject offic(?:er|ial)[^.]*\.?",
    _I,
)


def _strip_boilerplate(t: bytes) -> bytes:
    """No-break spaces to spaces, then drop the privacy paragraph and the witness-officer glossary."""
    return _GLOSSARY.sub(b" ", _BOILER.sub(b" ", t.replace(b"\xc2\xa0", b" ")))


_FR_NOTE = re.compile(rb"Remarque\s*:\s*Un agent (?:impliqu|t\xc3\xa9moin)[^\]\n]*\]?", _I)
_FR_LEGAL = re.compile(
    rb"[^.;\n]*(?:sont invit\xc3\xa9s|y compris des|le nom (?:d(?:'|\xe2\x80\x99)un|de tout)|On entend par|"
    rb"n(?:'|\xe2\x80\x99)est pas un agent impliqu|n(?:'|\xe2\x80\x99)y sont pas)[^.;\n]*[.;]?",
    _I,
)
_FR_ORD = re.compile(rb"(?:\bAI|agent(?:\(e\)|e)?\s+impliqu\xc3\xa9(?:\(e\)|e)?)\s*" + _NO + rb"\s*(\d{1,2})\b")
_FR_HEAD = re.compile(rb"(?:^|\n)[ \t]*Agents? impliqu\xc3\xa9(?:e|s|es)?[ \t]*(?:\([ \t]*AI[ \t]*\))?[ \t]*\n")
_FR_NEXT = re.compile(
    rb"\n[ \t]*(?:Agents? t\xc3\xa9moins?|T\xc3\xa9moins? civils?|\xc3\x89l\xc3\xa9ments de preuve|Remarque|En vertu)"
)
_FR_ENTRY = re.compile(rb"(?:^|\n)[ \t]*AI\b(?:\s*" + _NO + rb"\s*(\d{1,2}))?")
_FR_NUMS = {b"un": 1, b"une": 1, b"deux": 2, b"trois": 3, b"quatre": 4, b"cinq": 5, b"six": 6, b"sept": 7,
            b"huit": 8, b"neuf": 9, b"dix": 10}  # fmt: skip
_FR_PLURAL = re.compile(rb"\b(\d{1,2}|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+agent(?:e)?s\s+impliqu", _I)
_FR_ZERO = re.compile(rb"\baucun(?:e)?\s+agent(?:e)?\s+impliqu", _I)
_FR_THE = re.compile(rb"\bl(?:'|\xe2\x80\x99)\s*(?:AI\b|agent(?:e)?\s+impliqu)", _I)
_FR_PL = re.compile(rb"\bles\s+AI\b|\bagent(?:e)?s\s+impliqu", _I)
_FR_AT = re.compile(rb"\bAT\s*" + _NO + rb"\s*\d|\bagent(?:e)?s?\s+t\xc3\xa9moin", _I)
_FR_ANY_AI = re.compile(rb"\bAI\b|agent(?:e)?s?\s+impliqu", _I)


def _resolve_fr(text: bytes) -> tuple[int | None, str]:
    """French reports: the subject official is the "agent impliqué" ("AI no 1")."""
    body = _FR_LEGAL.sub(b" ", _FR_NOTE.sub(b" ", text))
    start = -1
    for m in _FR_HEAD.finditer(body):
        start = m.end()
    if start >= 0:
        win = body[start : start + 2500]
        nx = _FR_NEXT.search(win)
        if nx:
            win = win[: nx.start()]
        entries = ord_ = 0
        for m in _FR_ENTRY.finditer(win):
            entries += 1
            if m.group(1) is not None:
                ord_ = max(ord_, int(m.group(1)))
        n = max(entries, ord_)
        if n > 0:
            return n, f"section: max(ordinal {ord_}, entries {entries})"
    ks = [int(m.group(1)) for m in _FR_ORD.finditer(body)]
    mo = max(ks, default=0)
    if mo > 0 and 1 in ks:
        return mo, f"max ordinal AI no {mo}"
    m = _FR_PLURAL.search(body)
    if m:
        tok = m.group(1).lower()
        return _FR_NUMS.get(tok) or int(tok), f"plural cue '{_s(m.group(0))}'"
    if _FR_ZERO.search(body):
        return 0, "zero: 'aucun agent impliqué'"
    the = len(_FR_THE.findall(body))
    if the >= 1 and not _FR_PL.search(body):
        return 1, f"singular present: 'l'AI / l'agent impliqué'x{the}"
    if _FR_AT.search(body) and not _FR_ANY_AI.search(body):
        return 0, "zero: witness officials only, no agent impliqué named"
    return None, f"UNRESOLVED (fr): 'l'AI'x{the}"


_EN_WORD_NUM = {b"one": 1, b"two": 2, b"three": 3, b"four": 4, b"five": 5, b"six": 6, b"seven": 7, b"eight": 8,
                b"nine": 9, b"ten": 10}  # fmt: skip
_R_UES, _R_SIU = re.compile(rb"\bUES\b"), re.compile(rb"\bSIU\b")
_R_FR_ROLE = re.compile(rb"agent(?:\(e\)|e)?s?\s+impliqu")
_R_EN_ROLE = re.compile(rb"subject offic", _I)
_R_EN_TAG = re.compile(rb"\bSOs?\b")
# "SO" is case-strict and "#" is REQUIRED (an icase optional-# variant matched "...also 59...")
_R_ORD_SO = re.compile(rb"\bSO\s*#\s*(\d{1,2})\b")
_R_ORD_SPELLED = re.compile(rb"subject offic(?:er|ial)\s*#\s*(\d{1,2})\b", _I)
_R_SECTION = re.compile(rb"Subject Offic(?:er|ial)s\b")
_R_NEXT_SECTION = re.compile(
    rb"\n\s{0,3}(?:Witness Offic(?:er|ial)s|Civilian Witness(?:es)?|Service Employee Witness|Incident Narrative|"
    rb"Materials [Oo]btained|The Scene|Evidence\n|Nature of Injur)"
)
_R_ENTRY = re.compile(rb"\bSO\s*(?:#\s*\d{1,2})?\s{0,3}(?:Interviewed|Declined|Did not consent|Not interviewed)")
_R_ANCHOR_SO = re.compile(rb"\bSO\s*#\s*1\b")
_R_ANCHOR_SPELLED = re.compile(rb"subject offic(?:er|ial)\s*#\s*1\b", _I)
_R_PLURAL = re.compile(
    rb"\b(?:the\s+)?(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+subject offic(?:er|ial)s\b", _I
)
_R_THE_SO = re.compile(rb"\b[Tt]he SO\b")
_R_THE_SUBJ = re.compile(rb"\bthe subject offic(?:er|ial)\b", _I)
_R_ANY_PLURAL = re.compile(rb"\bthe SOs\b|the subject offic(?:er|ial)s\b", _I)
_R_ZERO = (
    re.compile(rb"no subject offic(?:er|ial)s?\b", _I),
    re.compile(rb"(?:did not|not|never)\s+designate[d]?\s+(?:a\s+|any\s+)?subject offic", _I),
    re.compile(rb"\bno (?:police )?(?:official|officer) was (?:a |the )?subject offic", _I),
)
_R_WO = re.compile(rb"\bWO\s*#\s*\d|\bwitness offic(?:er|ial)s?\b", _I)
_R_ANY_SO = re.compile(rb"\bSOs?\b|subject offic", _I)


def _max_ordinal(s: bytes) -> int:
    return max((int(m.group(1)) for rx in (_R_ORD_SO, _R_ORD_SPELLED) for m in rx.finditer(s)), default=0)


def _resolve(report_text: bytes) -> tuple[int | None, str]:
    body = _strip_boilerplate(report_text)
    if len(_R_UES.findall(body)) > len(_R_SIU.findall(body)) or (
        _R_FR_ROLE.search(body) and not _R_EN_ROLE.search(body) and not _R_EN_TAG.search(body)
    ):
        return _resolve_fr(body)
    # 0. the Team block under "Subject Officials/Officers" is authoritative
    sec = _R_SECTION.search(body)
    if sec:
        window = body[sec.end() : sec.end() + 2500]
        nxt = _R_NEXT_SECTION.search(window)
        if nxt:
            window = window[: nxt.start()]
        sec_ord = _max_ordinal(window)
        entries = len(_R_ENTRY.findall(window))
        if max(sec_ord, entries) > 0:
            return max(sec_ord, entries), f"section: max(ordinal {sec_ord}, entries {entries})"
    # 1. document-wide highest ordinal, only with the "#1" roster anchor
    max_ord = _max_ordinal(body)
    if max_ord > 0 and (_R_ANCHOR_SO.search(body) or _R_ANCHOR_SPELLED.search(body)):
        return max_ord, f"max ordinal SO #{max_ord}"
    # 2. spelled-out / numeric plural: "the two subject officials", "designated two subject officers"
    m = _R_PLURAL.search(body)
    if m:
        tok = m.group(1).lower()
        return _EN_WORD_NUM.get(tok) or int(tok), f"plural cue '{_s(m.group(0))}'"
    # 3. a subject official is PRESENT (before the zero rule)
    the_so = len(_R_THE_SO.findall(body))
    the_subj = len(_R_THE_SUBJ.findall(body))
    if the_so + the_subj >= 1 and not _R_ANY_PLURAL.search(body):
        return 1, f"singular present: 'the SO'x{the_so} 'the subject official'x{the_subj}"
    # 4. explicitly ZERO (a direct assertion only)
    if any(rx.search(body) for rx in _R_ZERO):
        return 0, "zero: witness-officer-only / 'not a subject official'"
    # 4b. witness officials and no subject official anywhere
    if _R_WO.search(body) and not _R_ANY_SO.search(body):
        return 0, "zero: witness officials only, no subject official named"
    return None, f"UNRESOLVED: 'the SO'x{the_so} 'the subj off'x{the_subj}"


def _resolve_so(report_text: str) -> tuple[int | None, str]:
    """``(count | None, reason)`` for one report's text (``None``: needs a human read)."""
    if _cxx is not None:
        return _cxx.siu_resolve_so(report_text)
    return _resolve(_b(report_text))


# ---- the 16 fields ----------------------------------------------------------

_NO_WO = re.compile(rb"no police officers? witness|aucun agent t\xc3\xa9moin")
_NO_CW = re.compile(rb"no civilian witness|aucun t\xc3\xa9moin civil")
_INV_FR = re.compile(rb"Nombre d(?:'|\xe2\x80\x99)enqu\xc3\xaateurs de l[^0-9\n]{0,30}?(\d+)")
_FOR_FR = re.compile(rb"Nombre d(?:'|\xe2\x80\x99)enqu\xc3\xaateurs sp\xc3\xa9cialistes[^0-9\n]{0,90}?(\d+)")


def _witnesses(text: bytes, head: bytes, none_rx: re.Pattern, tag: bytes, spelled_label: bytes, legacy: bytes) -> bytes:
    sec = _role_section(text, head)
    if not sec:
        return _count_labelled(text, legacy)
    if none_rx.search(_flatten_ws(sec).lower()):
        return b"0"
    n = _count_tagged(sec, tag)
    spelled = _count_labelled(sec, spelled_label)
    if spelled and (not n or int(spelled) > int(n)):
        n = spelled
    return n


def _parse(t: bytes) -> dict[str, str]:
    f: dict[str, bytes] = {}
    lang = f["_language"] = _detect_language(t)
    fr = lang == b"fr"
    f["police_service"] = _detect_subject_service(t, fr) or (
        (_detect_police_service_fr(t) or _detect_police_service(t)) if fr else _detect_police_service(t)
    )
    inc = (_detect_incident_date_fr(t) or _detect_incident_date(t)) if fr else _detect_incident_date(t)
    f["date_of_incident_iso"] = _iso(inc)
    notified = (_detect_siu_notified_fr(t) or _detect_siu_notified(t)) if fr else _detect_siu_notified(t)
    f["date_siu_notified_iso"] = _iso(notified)
    # a notification that dates the incident relative to itself wins over the first dated sentence
    rel = _relative_incident(t, f["date_siu_notified_iso"], fr)
    if rel:
        f["date_of_incident_iso"] = rel
    dec = (_detect_decision_date_fr(t) or _detect_decision_date(t)) if fr else _detect_decision_date(t)
    f["date_of_director_decision_iso"] = _iso(dec)

    f["siu_investigators"] = _team_count(t, b"SIU Investigators")
    f["siu_forensics_investigators"] = _team_count(t, b"SIU Forensic Investigators")
    if fr:
        m = _INV_FR.search(t)
        if not f["siu_investigators"] and m:
            f["siu_investigators"] = m.group(1)
        m = _FOR_FR.search(t)
        if not f["siu_forensics_investigators"] and m:
            f["siu_forensics_investigators"] = m.group(1)
    if not f["siu_investigators"]:
        f["siu_investigators"] = (
            _team_words(t, b"enqu\xc3\xaateurs", True) if fr else _team_words(t, b"SIU investigators", False)
        )
    if not f["siu_forensics_investigators"]:
        f["siu_forensics_investigators"] = (
            _team_words(t, b"techniciens en identification", True)
            if fr
            else _team_words(t, rb"(?:SIU )?forensic investigators", False)
        )

    so = _role_section(t, _ROLE_HEADS[3] if fr else _ROLE_HEADS[0])
    so_legacy = b"" if so else _count_labelled(t, b"agent impliqu\xc3\xa9" if fr else b"Subject Officer")
    # with no subject-official section only a numbered tag counts: a bare "SO" is no evidence of how many
    if so_legacy:
        n_so = so_legacy
    elif so:
        n_so = _count_tagged(so, b"AI" if fr else b"SO")
    else:
        n_so = _count_labelled(t, b"AI" if fr else b"SO")
    if not n_so:
        count, _ = _resolve(t)
        if count is not None:
            n_so = str(count).encode()
    f["number_of_subject_officials"] = n_so

    if fr:
        f["number_of_witness_officials"] = _witnesses(
            t, _ROLE_HEADS[4], _NO_WO, b"AT", b"agent t\xc3\xa9moin", b"agent t\xc3\xa9moin"
        )
        f["number_of_civilian_witnesses"] = _witnesses(
            t, _ROLE_HEADS[5], _NO_CW, b"TC", b"t\xc3\xa9moin civil", b"t\xc3\xa9moin civil"
        )
    else:
        f["number_of_witness_officials"] = _witnesses(
            t, _ROLE_HEADS[1], _NO_WO, b"WO", rb"Witness Offic(?:er|ial)", b"Witness Officer"
        )
        f["number_of_civilian_witnesses"] = _witnesses(
            t, _ROLE_HEADS[2], _NO_CW, b"CW", b"Civilian Witness", b"Civilian Witness"
        )

    age, sex = _detect_age_sex(t)
    if fr and not age:
        age, sex = _detect_age_sex_fr(t)
    f["age_affected"] = age
    f["sex_gender_affected"] = sex
    f["charges_recommended"] = _detect_charges(t, fr)
    f["directors_name"] = _detect_directors_name(t)
    f["location_of_call"] = _detect_location(t)
    f["specific_injuries"] = _detect_specific_injuries(t, fr)
    f["relevant_legislation"] = _detect_legislation(t, fr)
    return {k: _s(f[k]) for k in sorted(f)}


def parse_report_text(text: str) -> dict[str, str]:
    """Parse plain report text into the 16 schema fields plus ``_language`` (``""`` when not stated)."""
    if _cxx is not None:
        return _cxx.siu_parse_report_text(text)
    return _parse(_b(text))


def parse_report_html(html: str) -> dict[str, str]:
    """HTML -> fields in one call (:func:`html_to_text` then :func:`parse_report_text`).

    Examples
    --------
    >>> f = parse_report_html("<p>The Investigation</p><p>It happened in the City of Barrie, a 34-year-old man ...</p>")
    >>> f["age_affected"], f["sex_gender_affected"], f["location_of_call"]
    ('34', 'man', 'City of Barrie')
    """
    if _cxx is not None:
        return _cxx.siu_parse_report_html(html)
    return _parse(_html_to_text(_b(html)))
