"""SIU analysis surfaces -- turns the scraped SIU.csv / SIU_by_case.csv
into structured, RichResult-emitting analyses.

Each callable here loads the canonical SIU outputs from
data/datasets/vsr/SIU_by_case.csv (or accepts a DataFrame directly)
and emits a `RichResult` with table + warnings + interpretation.

Design rule: every analysis should be reproducible from the canonical
inputs and produce CSV/JSON output under data/manifest/outputs/siu/.
"""

from __future__ import annotations

import datetime
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from morie.fn import _frame_core as pd

from ..fn._richresult import RichResult


def _load(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the SIU case table: the given CSV, else the reviewed corpus (``morie pull siu``)."""
    if csv_path is None:
        from ..data import load_dataset

        return _clean(load_dataset("siu"))
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"SIU dataset not found at {p}. Without a path the reviewed corpus is used; "
            "a file comes from `morie pull siu --out FILE`."
        )
    return _clean(pd.read_csv(p))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """One spelling per police service and real incident dates (the SIU was created in 1990)."""
    from .native import to_iso_date

    if "police_service" in df.columns:
        # "Ontario Provincial Police (OPP)" and "Ontario Provincial Police" are one service
        df["police_service"] = [
            re.sub(r"\s*\([A-Z]{2,6}\)$", "", str(v)).strip() if isinstance(v, str) else v for v in df["police_service"]
        ]
    if "date_of_incident_iso" in df.columns:
        this_year = datetime.date.today().year
        fixed = []
        for v in df["date_of_incident_iso"]:
            d = v if isinstance(v, str) else ""
            if d and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                d = to_iso_date(d)  # "July 18, 2021" stored in an _iso column
            if d and not 1990 <= int(d[:4]) <= this_year:
                d = ""
            fixed.append(d)
        df["date_of_incident_iso"] = fixed
    return df


_FALSY_TEXT = re.compile(r"^(no|not|none|there are no)\b|not warranted|no (grounds|basis|criminal|charges)")


def _charge_flags(values) -> list:
    """charges_recommended as True / False / None: the corpus holds booleans and the Director's own
    words ("no criminal charges warranted", "No Charges to Issue")."""
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
            continue
        t = str(v).strip().lower() if v is not None and v == v else ""
        if t in ("true", "yes", "1", "t"):
            out.append(True)
        elif t in ("false", "no", "0", "f", "none") or (t and _FALSY_TEXT.search(t)):
            out.append(False)
        else:
            out.append(None)
    return out


def _iso_day(v):
    try:
        return datetime.date.fromisoformat(str(v)[:10]) if isinstance(v, str) and v else None
    except ValueError:
        return None


def by_police_service(csv_path: Path | str | None = None) -> RichResult:
    """Per-police-service tabulation: case counts, charges-recommended
    rate, common injuries.
    """
    df = _load(csv_path)
    svcs = [v if isinstance(v, str) and v.strip() else "(not recorded)" for v in df["police_service"]]
    flags = _charge_flags(df["charges_recommended"]) if "charges_recommended" in df.columns else [None] * len(svcs)
    n_cases = Counter(svcs)
    charges = Counter(s for s, f in zip(svcs, flags) if f is True)
    no_charges = Counter(s for s, f in zip(svcs, flags) if f is False)
    out_rows = []
    for svc, n in n_cases.items():
        c, nc = charges.get(svc, 0), no_charges.get(svc, 0)
        out_rows.append([svc[:50], n, c, nc, f"{100 * c / (c + nc):.1f}%" if (c + nc) else "--"])
    out_rows.sort(key=lambda r: -r[1])  # by case count

    return RichResult(
        title="SIU cases by police service",
        summary_lines=[
            ("Unique police services", len(n_cases)),
            ("Total cases", int(df.shape[0])),
            ("With charges_recommended True", sum(charges.values())),
            ("With charges_recommended False", sum(no_charges.values())),
        ],
        tables=[
            {
                "title": "By police service (top 30):",
                "headers": ["Police service", "Cases", "Charged", "No charges", "Charge rate"],
                "rows": out_rows[:30],
            }
        ],
        interpretation=(
            f"Top services by case count: "
            f"{', '.join(r[0] for r in out_rows[:5])}. "
            "Services with low charge rate may indicate either "
            "truly justified force or systematic under-charging -- "
            "context-dependent interpretation."
        ),
        payload={"counts": dict(n_cases), "charges": dict(charges), "no_charges": dict(no_charges)},
    )


def by_year(csv_path: Path | str | None = None) -> RichResult:
    """Year-over-year case volume + charges rate from date_of_incident."""
    df = _load(csv_path)
    years = [int(d[:4]) if isinstance(d, str) and d else None for d in df["date_of_incident_iso"]]
    flags = _charge_flags(df["charges_recommended"]) if "charges_recommended" in df.columns else [None] * len(years)
    n = Counter(y for y in years if y is not None)
    charged = Counter(y for y, f in zip(years, flags) if y is not None and f is True)
    no_charged = Counter(y for y, f in zip(years, flags) if y is not None and f is False)
    rows = []
    for year in sorted(n):
        c, nc = charged.get(year, 0), no_charged.get(year, 0)
        rows.append([year, n[year], c, nc, f"{100 * c / (c + nc):.1f}%" if (c + nc) else "--"])
    n_valid = sum(n.values())
    return RichResult(
        title="SIU cases by year",
        summary_lines=[
            ("Years covered", f"{min(n)}–{max(n)}" if n else "n/a"),
            ("Total cases with parseable date", n_valid),
            ("Cases with no parseable date", int(df.shape[0]) - n_valid),
        ],
        tables=[
            {
                "title": "By year:",
                "headers": ["Year", "Cases", "Charged", "No charges", "Charge rate"],
                "rows": rows,
            }
        ],
        payload={"by_year": {y: n[y] for y in sorted(n)}},
    )


def case_counts(csv_path: Path | str | None = None) -> RichResult:
    """Distribution of #SO, #WO, #CW per case."""
    df = _load(csv_path)
    rows = []
    for col, label in [
        ("number_of_subject_officials", "#Subject officials"),
        ("number_of_witness_officials", "#Witness officials"),
        ("number_of_civilian_witnesses", "#Civilian witnesses"),
        ("number_of_officers_involved", "#Officers involved"),
    ]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.size == 0:
            rows.append([label, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append(
            [
                label,
                int(vals.size),
                f"{vals.mean():.2f}",
                f"{vals.median():.0f}",
                f"{vals.min():.0f}",
                f"{vals.max():.0f}",
            ]
        )
    return RichResult(
        title="SIU case-team size distribution",
        summary_lines=[("Total cases", int(df.shape[0]))],
        tables=[
            {
                "title": "Per-case team / witness counts:",
                "headers": ["Field", "n parsed", "Mean", "Median", "Min", "Max"],
                "rows": rows,
            }
        ],
    )


def demographics(csv_path: Path | str | None = None) -> RichResult:
    """Sex/age distribution of affected persons."""
    df = _load(csv_path)
    sex = df["sex_gender_affected"].fillna("unknown").value_counts()
    sex_rows = [[k, int(v), f"{100 * v / sex.sum():.1f}%"] for k, v in sex.items()]
    age = pd.to_numeric(df["age_affected"], errors="coerce").dropna()
    # an "age" of 1985 is a birth year: ages outside 0-110 are left out and counted
    plausible = (age >= 0) & (age <= 110)
    n_implausible = int((~plausible).sum())
    age = age[plausible]
    return RichResult(
        title="Affected-person demographics",
        summary_lines=[
            ("Total cases", int(df.shape[0])),
            ("Cases with parseable age", int(age.size)),
            ("Ages outside 0-110 (left out)", n_implausible),
            ("Mean age", float(age.mean()) if age.size else float("nan")),
            ("Median age", float(age.median()) if age.size else float("nan")),
            ("Age range", f"{int(age.min())}–{int(age.max())}" if age.size else "n/a"),
        ],
        tables=[
            {
                "title": "By sex/gender:",
                "headers": ["Sex/gender", "Count", "Percent"],
                "rows": sex_rows,
            }
        ],
    )


def mental_health_race_indicators(csv_path: Path | str | None = None) -> RichResult:
    """Frequency of MH/race keyword indicators in narratives."""
    df = _load(csv_path)
    counts = Counter()
    nonempty = 0
    for sig in df["mental_health_or_race_indications"].dropna():
        s = str(sig).strip()
        if not s:
            continue
        nonempty += 1
        for kw in s.split(";"):
            kw = kw.strip()
            if kw:
                counts[kw] += 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    return RichResult(
        title="Mental-health / race indicators in SIU narratives",
        summary_lines=[
            ("Total cases", int(df.shape[0])),
            ("Cases with ≥1 indicator", nonempty),
            ("Distinct keywords matched", len(counts)),
        ],
        tables=[
            {
                "title": "Top keywords:",
                "headers": ["Keyword", "Cases mentioning"],
                "rows": [[k, v] for k, v in rows[:25]],
            }
        ],
        warnings=[
            "Keyword-presence is a SIGNAL not a verdict. A case mentioning "
            "'mental health' may discuss it briefly without being primarily "
            "about MH. Read narratives in `SIU_narratives.jsonl` for context."
        ],
        interpretation=(
            f"{nonempty}/{int(df.shape[0])} cases ({100 * nonempty / max(df.shape[0], 1):.1f}%) "
            "have at least one MH or race keyword in the narrative. The "
            "distribution by keyword is shown above; see also `by_police_service` "
            "for service-by-service patterns."
        ),
    )


def decision_timing(csv_path: Path | str | None = None) -> RichResult:
    """Distributions of intervals: incident -> notification -> director's decision."""
    df = _load(csv_path)
    inc = [_iso_day(v) for v in df["date_of_incident_iso"]]
    notif = [_iso_day(v) for v in df["date_siu_notified_iso"]]
    decision = [_iso_day(v) for v in df["date_of_director_decision_iso"]]

    def _days(a, b):
        # a later step dated before an earlier one is a recording error: left out
        return [(y - x).days for x, y in zip(a, b) if x is not None and y is not None and y >= x]

    inc_to_notif = _days(inc, notif)
    notif_to_decision = _days(notif, decision)
    inc_to_decision = _days(inc, decision)

    def _row(label, v):
        if not v:
            return [label, "n/a", "n/a", "n/a", "n/a", "n/a"]
        return [
            label,
            len(v),
            f"{statistics.fmean(v):.1f}",
            f"{statistics.median(v):.0f}",
            f"{min(v):.0f}",
            f"{max(v):.0f}",
        ]

    return RichResult(
        title="SIU decision timing (days)",
        summary_lines=[("Total cases", int(df.shape[0]))],
        tables=[
            {
                "title": "Interval distributions (days):",
                "headers": ["Interval", "n parsed", "Mean", "Median", "Min", "Max"],
                "rows": [
                    _row("Incident -> SIU notified", inc_to_notif),
                    _row("Notified -> director's decision", notif_to_decision),
                    _row("Incident -> decision (total)", inc_to_decision),
                ],
            }
        ],
    )


def all_analyses(csv_path: Path | str | None = None, out_dir: Path | None = None) -> dict:
    """Run every analysis; with *out_dir*, write each to its own .txt/.json file there.
    Returns a dict of name -> RichResult."""
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for name, fn in [
        ("by_police_service", by_police_service),
        ("by_year", by_year),
        ("case_counts", case_counts),
        ("demographics", demographics),
        ("mh_race_indicators", mental_health_race_indicators),
        ("decision_timing", decision_timing),
    ]:
        try:
            r = fn(csv_path)
            results[name] = r
            if out_dir is not None:
                (out_dir / f"siu_analysis_{name}.txt").write_text(str(r))
                (out_dir / f"siu_analysis_{name}.json").write_text(
                    json.dumps(r.payload, indent=2, default=str, ensure_ascii=False)
                )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(
                title=f"siu.{name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
