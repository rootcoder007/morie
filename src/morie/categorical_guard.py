"""Categorical-integrity guards: the Python arm of rmorie's module 25.

Category-mapping errors are among the most damaging silent failures in
applied statistics: numeric codes imported without their value labels,
positional recodes that reassign groups wholesale, and alphabetical
releveling that silently changes the reference category. Any of them can
relabel entire demographic groups and multiply a reported odds ratio
severalfold without a warning. The functions here make every recode
explicit and name-based, refuse anything unmapped, audit imported
columns for the known hazards, prove a recode with a before/after
cross-tabulation, verify recoded counts against the counts a release
published, and check reported odds ratios against a labelled table under
every relabelling, so a swapped label is named as such rather than
blamed on the software that carried it.

Every function here agrees with the R arms (`morie` and `rmorie`) on
identical inputs to twelve digits, including the wording of its errors.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
from typing import Any

__all__ = [
    "transfer_verify",
    "safe_relabel",
    "decode_labelled",
    "relabel_forensics",
    "safe_recode", "safe_factor", "audit_categories", "crosstab_verify",
    "marginals_verify", "odds_ratio_check", "guard_binary_treatment",
]


def _squote(v: str) -> str:
    return "‘" + str(v) + "’"


def _is_na(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _uniq(seq):
    out = []
    for v in seq:
        if not _is_na(v) and v not in out:
            out.append(v)
    return out


def safe_recode(x, mapping: dict, keep=()) -> dict:
    """Recode by explicit name-to-name mapping; unmapped values are an error.

    Returns ``{"values": [...], "audit": {"mapping", "kept", "checksum"}}``.
    The checksum is the SHA-256 of ``old=new;old=new;...`` in mapping order,
    the same digest the R arms attach as the ``morie_recode_audit`` attribute.
    """
    if not isinstance(mapping, dict) or not mapping or any(not k for k in mapping):
        raise ValueError("morie_safe_recode: `mapping` must be a non-empty named mapping")
    keep = list(keep)
    seen = _uniq(x)
    unmapped = [v for v in seen if v not in mapping and v not in keep]
    if unmapped:
        raise ValueError(
            "morie_safe_recode: values with NO mapping: "
            + ", ".join(_squote(u) for u in unmapped)
            + ". Every observed category must be mapped by name (or "
            "listed in `keep`); silent pass-through is how group "
            "labels get corrupted.")
    out = [None if _is_na(v) else mapping.get(v, v) for v in x]
    digest = hashlib.sha256(";".join(f"{k}={v}" for k, v in mapping.items()).encode("utf-8")).hexdigest()
    return {"values": out, "audit": {"mapping": dict(mapping), "kept": keep, "checksum": digest}}


def safe_factor(x, levels, reference=None) -> dict:
    """A factor with explicit, verified levels: the first level is the reference.

    Returns ``{"values": [...], "levels": [...], "codes": [...], "reference": str}``
    where ``codes`` are 1-based level indices (R's integer codes) or None.
    """
    levels = list(levels)
    stray = [v for v in _uniq(x) if v not in levels]
    if stray:
        raise ValueError("morie_safe_factor: values outside the declared levels: "
                         + ", ".join(_squote(s) for s in stray))
    if reference is not None and reference != levels[0]:
        raise ValueError(
            "morie_safe_factor: declared reference " + _squote(reference)
            + " is not levels[1] (" + _squote(levels[0]) + "); reorder "
            "`levels` so the reference is explicit and first.")
    codes = [None if _is_na(v) else levels.index(v) + 1 for v in x]
    return {"values": [None if _is_na(v) else v for v in x], "levels": levels,
            "codes": codes, "reference": levels[0]}


def audit_categories(data: dict, cols=None, factor_levels: dict | None = None) -> list:
    """Audit categorical columns for coding hazards.

    ``data`` maps column name to a list of values. ``factor_levels`` may give
    the declared level order of a column (an R factor); such columns are
    reported with storage ``factor`` and their unused levels flagged.
    Returns one dict per column with ``column``, ``storage``, ``n_levels``,
    ``levels`` (first eight, joined by ``|``), ``reference`` and ``hazards``
    (joined by ``' ;; '``), plus ``clean`` on the last entry's parent list
    via ``audit_categories.clean`` semantics mirrored in the R print method.
    """
    factor_levels = factor_levels or {}
    if cols is None:
        cols = [c for c, v in data.items()
                if c in factor_levels or all(_is_na(u) or isinstance(u, str) for u in v)]
    rows = []
    for cn in cols:
        v = data[cn]
        hazards = []
        if cn in factor_levels:
            lv = list(factor_levels[cn])
            storage = "factor"
        else:
            lv = sorted({str(u) for u in v if not _is_na(u)})
            storage = "character"
        obs = _uniq(str(u) for u in v if not _is_na(u))
        if lv and all(all(ch in "0123456789." for ch in lab) and lab for lab in lv):
            hazards.append("all labels numeric-looking (" + ",".join(lv[:4])
                           + "...): likely imported CODES whose value labels were lost; "
                           "as.numeric() on this column returns level INDICES, not data")
        if lv and all(re.match(r"^[0-9]+[.):]? ?[A-Za-z]", lab) for lab in lv):
            hazards.append("labels carry code prefixes (" + ",".join(lv[:3])
                           + "...): the code is part of the string, so the level order is "
                           "the CODE order; any positional relabel with labels in another "
                           "order rotates the groups. Decode by code (decode_labelled / "
                           "safe_relabel), never by position")
        inv = "\\s\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff"
        padded = [lab for lab in lv if re.search("^[" + inv + "]|[" + inv + "]$", lab)]
        if padded:
            hazards.append("labels with leading/trailing whitespace (incl. non-breaking): "
                           + ", ".join(_squote(p) for p in padded)
                           + ": a space splits one category into two")
        core = [re.sub("^[" + inv + "]+|[" + inv + "]+$", "", lab) for lab in lv]
        dupc = [lab for lab, c in zip(lv, core) if core.count(c) > 1]
        if dupc:
            hazards.append("whitespace-variant duplicate labels: " + ", ".join(_squote(d) for d in dupc))
        if any(c == "" for c in core):
            hazards.append('empty-string label "": missingness stored as a category')
        sentinels = {"NA", "N/A", "NAN", "NULL", "NONE", ".", "-", "?"}
        sentinel = [lab for lab, c in zip(lv, core) if c.upper() in sentinels]
        if sentinel:
            hazards.append("missing-value sentinel stored as a label: "
                           + ", ".join(_squote(x) for x in sentinel))
        if lv and (core[0] == "" or lv[0] != core[0] or lv[0] in sentinel):
            hazards.append("the REFERENCE level " + _squote(lv[0])
                           + " is empty, a sentinel, or differs from a real label only by "
                           "invisible characters: every model on this column is baselined on it")
        lc = [c.lower() for c in core]
        # a case-variant pair is two DIFFERENT trimmed labels that agree once
        # lower-cased; a whitespace-only pair was reported above
        groups = {}
        for c, low in zip(core, lc):
            groups.setdefault(low, set()).add(c)
        case_keys = {k for k, g in groups.items() if len(g) > 1}
        cv = [lab for lab, low in zip(lv, lc) if low in case_keys]
        if cv:
            hazards.append("case-variant duplicate labels: " + ", ".join(_squote(d) for d in cv))
        if cn in factor_levels:
            unused = [lab for lab in lv if lab not in obs]
            if unused:
                hazards.append("unused levels: " + ", ".join(_squote(u) for u in unused))
        if len(lv) > 50:
            hazards.append(f"{len(lv)} levels: identifier mistaken for a category?")
        rows.append({"column": cn, "storage": storage, "n_levels": len(lv),
                     "levels": "|".join(lv[:8]), "reference": lv[0] if lv else None,
                     "hazards": " ;; ".join(hazards)})
    return rows


def crosstab_verify(original, recoded, declared: dict) -> dict:
    """Prove a recode with a before/after cross-tabulation; errors otherwise."""
    if len(original) != len(recoded):
        raise ValueError(f"morie_crosstab_verify: length mismatch ({len(original)} vs {len(recoded)}): "
                         "rows were lost or duplicated during the recode.")
    if [_is_na(a) for a in original] != [_is_na(b) for b in recoded]:
        raise ValueError("morie_crosstab_verify: missingness changed during the recode "
                         "(values silently became NA, or NAs were filled).")
    tab: dict = {}
    for a, b in zip(original, recoded):
        if _is_na(a):
            continue
        tab.setdefault(a, {}).setdefault(b, 0)
        tab[a][b] += 1
    fan = [old for old, d in tab.items() if len(d) > 1]
    if fan:
        raise ValueError("morie_crosstab_verify: original category mapped to MULTIPLE new categories: "
                         + ", ".join(_squote(o) for o in sorted(fan))
                         + ". The recode is not a function of the category label.")
    for old in sorted(tab):
        realized = next(iter(tab[old]))
        expected = declared.get(old, old)
        if realized != expected:
            raise ValueError("morie_crosstab_verify: " + _squote(old) + " was mapped to "
                             + _squote(realized) + " but the declared mapping says "
                             + _squote(expected) + ". THIS is how groups get swapped; fix "
                             "the recode before any model runs.")
    return {old: dict(d) for old, d in sorted(tab.items())}


def _permutations(labels):
    return [list(p) for p in itertools.permutations(labels)]


def marginals_verify(x, published: dict, tolerance: float = 0, strict: bool = True) -> dict:
    """Recoded counts per label must equal the published counts; names a permutation otherwise."""
    if not published or any(not k for k in published):
        raise ValueError("morie_marginals_verify: `published` must be a non-empty named mapping")
    labs = list(published)
    obs = {lab: sum(1 for v in x if not _is_na(v) and v == lab) for lab in labs}
    extra = [v for v in _uniq(x) if v not in labs]
    if extra:
        msg = ("morie_marginals_verify: labels present in the data but not in the "
               "published counts: " + ", ".join(_squote(e) for e in extra))
        if strict:
            raise ValueError(msg)
        return {"counts": obs, "published": {lab: published[lab] for lab in labs}, "ok": False,
                "permutation": None, "message": msg}
    ok = all(abs(obs[lab] - published[lab]) <= tolerance for lab in labs)
    perm = None
    if not ok and len(labs) <= 7:
        for p in _permutations(labs):
            relabelled = dict(zip(p, [obs[lab] for lab in labs]))
            if p != labs and all(abs(relabelled[lab] - published[lab]) <= tolerance for lab in labs):
                perm = dict(zip(labs, p))
                break
    out = {"counts": obs, "published": {lab: published[lab] for lab in labs}, "ok": ok,
           "permutation": perm, "message": None}
    if not ok:
        detail = "; ".join(f"{lab}: observed {_num(obs[lab])}, published {_num(published[lab])}" for lab in labs)
        hint = ""
        if perm is not None:
            moved = [lab for lab in labs if perm[lab] != lab]
            hint = (" The observed counts match the published ones if the labels are permuted ("
                    + ", ".join(f"{m} -> {perm[m]}" for m in moved)
                    + "): the labels are attached to the wrong groups.")
        out["message"] = ("morie_marginals_verify: recoded counts do not match the published counts. "
                          + detail + "." + hint)
        if strict:
            raise ValueError(out["message"])
    return out


def _num(v) -> str:
    """Render a count the way R's paste() does: 3 not 3.0."""
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def odds_ratio_check(counts: dict, reference: str, reported: dict, tolerance: float = 0.05) -> dict:
    """Check reported odds ratios against a labelled k-by-2 table under every relabelling.

    ``counts`` maps group label to ``[outcome_absent, outcome_present]``.
    """
    labs = list(counts)
    if any(len(v) != 2 for v in counts.values()) or not labs:
        raise ValueError("morie_odds_ratio_check: `counts` must be a k-by-2 matrix with row names "
                         "(groups) and columns outcome-absent, outcome-present.")
    if reference not in labs:
        raise ValueError("morie_odds_ratio_check: reference " + _squote(reference) + " is not a row of `counts`.")
    others = [lab for lab in labs if lab != reference]
    if any(o not in reported for o in others):
        raise ValueError("morie_odds_ratio_check: `reported` must be named by every non-reference row: "
                         + ", ".join(_squote(o) for o in others))
    rep = {o: float(reported[o]) for o in others}

    def or_of(m):
        ref_odds = m[reference][1] / m[reference][0]
        return {r: (m[r][1] / m[r][0]) / ref_odds for r in others}

    def close_to(a, b):
        return all(math.isfinite(a[k]) and math.isfinite(b[k]) for k in b) and \
            all(abs(a[k] - b[k]) <= tolerance * max(abs(b[k]), 2.220446049250313e-16) for k in b)

    computed = or_of({lab: [float(v[0]), float(v[1])] for lab, v in counts.items()})
    consistent = close_to(computed, rep)
    matches = []
    if len(labs) <= 7:
        for p in _permutations(labs):
            for swap_cols in (False, True):
                if p == labs and not swap_cols:
                    continue
                base = {lab: ([float(counts[lab][1]), float(counts[lab][0])] if swap_cols else [float(counts[lab][0]), float(counts[lab][1])]) for lab in labs}
                relabelled = dict(zip(p, [base[lab] for lab in labs]))
                m = {lab: relabelled[lab] for lab in labs}
                if close_to(or_of(m), rep):
                    moved = [(labs[i], p[i]) for i in range(len(labs)) if p[i] != labs[i]]
                    matches.append({"relabelling": ", ".join(f"{a} -> {b}" for a, b in moved) if moved else "none",
                                    "outcome_columns_swapped": swap_cols})
    if consistent:
        verdict = "the reported odds ratios follow from the table as labelled"
    elif matches:
        verdict = ("the reported odds ratios do NOT follow from the table as labelled; they are "
                   "reproduced under a relabelling (" + matches[0]["relabelling"]
                   + ("; outcome columns swapped" if matches[0]["outcome_columns_swapped"] else "")
                   + "): the groups were mislabelled, not the software")
    else:
        verdict = "the reported odds ratios follow from no relabelling of this table"
    return {"computed": computed, "reported": rep, "consistent": consistent, "matches": matches, "verdict": verdict}


def guard_binary_treatment(x, col: str) -> bool:
    """Refuse a categorical column where a numeric 0/1 treatment is required."""
    if any(isinstance(v, str) for v in x if not _is_na(v)):
        lv = _uniq(str(v) for v in x if not _is_na(v))
        raise ValueError("Column " + _squote(col) + " is categorical ("
                         + ", ".join(_squote(lab) for lab in lv[:4])
                         + "...). Refusing to coerce: as.numeric() on a factor returns level INDICES "
                         "(1, 2, ...), not your data, and a mis-ordered level silently relabels every "
                         "observation. Encode explicitly first, e.g. morie_safe_recode() + "
                         "as.integer(x == \"treated_label\").")
    ux = _uniq(x)
    if any(v not in (0, 1) for v in ux):
        raise ValueError("Column " + _squote(col) + " must be binary 0/1 (saw: "
                         + ", ".join(_num(v) for v in ux[:5]) + ").")
    return True


def safe_relabel(x, mapping, keep=()) -> dict:
    """Relabel by NAME only. A list of labels assigned by position is the
    mechanism behind the documented four-way rotation (OHRC correction,
    26 January 2023), so anything that is not a dict is refused."""
    if not isinstance(mapping, dict):
        raise ValueError("safe_relabel: `mapping` must be NAMED (old label = new label). "
                         "Assigning labels by POSITION is how four race codes were rotated "
                         "in a published analysis (OHRC correction, 26 January 2023): the "
                         "labels were in alphabetical order, the codes were not.")
    old = _uniq(str(v) for v in x if not _is_na(v))
    unmapped = [v for v in old if v not in mapping and v not in keep]
    if unmapped:
        raise ValueError("safe_relabel: level(s) with NO mapping: "
                         + ", ".join(_squote(v) for v in unmapped)
                         + ". Name every level or list it in `keep`.")
    out = safe_recode(x, mapping, keep=keep)
    levels = _uniq(list(mapping.values()) + list(keep))
    out["levels"] = [lv for lv in levels if lv in mapping.values() or lv in out["values"]]
    return out


def decode_labelled(codes, value_labels: dict | None = None) -> dict:
    """Decode imported codes by their value labels, BY CODE. `value_labels`
    maps code -> label (pyreadstat's value_labels[var] shape). Levels come
    out in code order, never alphabetical."""
    if value_labels is None:
        raise ValueError("decode_labelled: no value labels supplied; the codes alone are NOT "
                         "the categories")
    vl = {str(k): str(v) for k, v in value_labels.items()}
    out = safe_recode([None if _is_na(c) else str(int(c) if isinstance(c, float) and c == int(c) else c)
                       for c in codes], vl)
    def _key(k):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)
    out["levels"] = [vl[k] for k in sorted(vl, key=_key)]
    return out


def relabel_forensics(value_labels: dict, observed: dict, counts: dict | None = None) -> dict:
    """Which deterministic step, applied to the code book, reproduces the
    observed permutation? A match on a sort-based mechanism exonerates the
    transfer: no import routine sorts value labels onto codes."""
    L = [str(v) for v in value_labels.values()]
    k = len(L)
    if set(observed) != set(L) or set(observed.values()) != set(L):
        raise ValueError("relabel_forensics: `observed` must be a permutation of the labels "
                         "in `value_labels`")
    obs = [str(observed[lab]) for lab in L]
    mech = {
        "labels sorted alphabetically, assigned by code position": sorted(L),
        "labels sorted case-insensitively, assigned by code position": sorted(L, key=str.lower),
        "labels sorted in reverse, assigned by code position": sorted(L, reverse=True),
        "labels reversed": L[::-1],
        "codes sorted as strings, labels assigned in that order":
            [L[i] for i in sorted(range(k), key=lambda i: str(list(value_labels)[i]))],
    }
    for r in range(1, k):
        mech[f"rotation by {r} position(s)"] = [L[(i + r) % k] for i in range(k)]
    if counts is not None:
        c = [float(counts[lab]) for lab in L]
        idx = sorted(range(k), key=lambda i: (-c[i], i))
        mech["labels ordered by decreasing frequency, assigned by code position"] = [L[i] for i in idx]
        idx = sorted(range(k), key=lambda i: (c[i], i))
        mech["labels ordered by increasing frequency, assigned by code position"] = [L[i] for i in idx]
    rows = [{"mechanism": nm, "permutation": ", ".join(a + " -> " + b for a, b in zip(L, perm)),
             "matches": perm == obs} for nm, perm in mech.items()]
    if obs == L:
        for r in rows:
            r["matches"] = False
        return {"rows": rows, "matches": [],
                "verdict": "The labels are in place: every label was seen under itself, so "
                           "there is no permutation to explain."}
    hit = [r["mechanism"] for r in rows if r["matches"]]
    if hit:
        verdict = ("The observed permutation is reproduced EXACTLY by: " + "; ".join(hit)
                   + ". No import routine (haven, foreign, pandas, pyreadstat) reorders value "
                   "labels; each carries them keyed by code. A transfer fault does not select "
                   "the sort order of the labels. The step that did this was a positional "
                   "relabel in the analysis, and it is reproducible from the code book alone.")
    else:
        verdict = ("No positional or sort-based mechanism reproduces the observed permutation. "
                   "Look at merges/joins on the category column, manual edits, and the file "
                   "itself before blaming either program.")
    return {"rows": rows, "matches": hit, "verdict": verdict}


def transfer_verify(imported, source_counts: dict, value_labels: dict | None = None,
                    code_book: dict | None = None, tolerance: float = 0,
                    strict: bool = True) -> dict:
    """Verify a categorical variable that crossed from SPSS/Stata/SAS.
    `value_labels` is what arrived with the file (pyreadstat), `code_book`
    what the source program's variable view says; `source_counts` its
    frequency table. Any disagreement is an error naming the permutation."""
    code_book_ok = None
    if value_labels is not None and code_book is not None:
        got = {str(k): str(v) for k, v in value_labels.items()}
        want = {str(k): str(v) for k, v in code_book.items()}
        bad = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        code_book_ok = not bad
        if bad and strict:
            raise ValueError("transfer_verify: the value labels that arrived disagree with the "
                             "source code book at code(s) " + ", ".join(bad) + ": arrived "
                             + ", ".join(k + "=" + str(got.get(k)) for k in bad) + "; source "
                             + ", ".join(k + "=" + str(want.get(k)) for k in bad))
    vl = value_labels if value_labels is not None else code_book
    if vl is not None:
        decoded = decode_labelled(imported, vl)
        values = decoded["values"]
    else:
        decoded = {"values": [None if _is_na(v) else str(v) for v in imported],
                   "levels": list(source_counts)}
        values = decoded["values"]
    m = marginals_verify(values, source_counts, tolerance=tolerance, strict=strict)
    ok = bool(m["ok"]) and code_book_ok is not False
    reasons = []
    if code_book_ok is False:
        reasons.append("value labels disagree with the source code book")
    if not m["ok"]:
        reasons.append(m["message"])
    return {"ok": ok, "decoded": decoded, "marginals": m, "code_book_ok": code_book_ok,
            "reasons": reasons}
