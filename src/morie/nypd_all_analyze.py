"""morie.nypd_all_analyze -- New York Police Department analysis aggregator.

Python parity with R's ``morie_nypd_all_analyses``. Loads NYPD arrests +
complaint data (bundled samples offline by default) and runs descriptive and
fairness surfaces. NYPD arrests carry ``perp_race`` and ``law_cat_cd``
(felony/misdemeanour), so a felony-charge disparate-impact by race is
computable directly from the data.

    analyze_<name>(...) -> RichResult
    analyze_all(*, out_dir=None) -> dict[str, RichResult]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .fairness.metrics import (
    fairness_demographic_parity,
    fairness_disparate_impact,
)
from .fn._richresult import RichResult

_DATA = Path(__file__).resolve().parent / "data"


def _load(which: str) -> pd.DataFrame | None:
    fname = {
        "arrests": "nypd_arrests_historic_sample.csv",
        "complaint": "nypd_complaint_historic_sample.csv",
    }[which]
    path = _DATA / fname
    if not path.exists():
        return None
    return pd.read_csv(path)


def _counts(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str)
    return s[s.str.len() > 0].value_counts()


def analyze_arrests_by_offense(
        arrests_df: pd.DataFrame | None = None) -> RichResult:
    """Arrest volume by NYPD offense description."""
    df = arrests_df if arrests_df is not None else _load("arrests")
    if df is None or "ofns_desc" not in df.columns:
        return RichResult(title="NYPD arrests by offense",
                          warnings=["missing column: ofns_desc"])
    counts = _counts(df["ofns_desc"])
    return RichResult(
        title="NYPD arrests by offense",
        summary_lines=[("distinct offenses", int(len(counts))),
                       ("records", int(counts.sum()))],
        tables=[{"title": "By offense (top 25)",
                 "headers": ["Offense", "Count"],
                 "rows": [[str(k), int(v)] for k, v in counts.head(25).items()]}],
        payload={"by_offense": {str(k): int(v)
                                for k, v in counts.head(50).items()}},
    )


def analyze_arrests_by_boro(
        arrests_df: pd.DataFrame | None = None) -> RichResult:
    """Arrest volume by borough."""
    df = arrests_df if arrests_df is not None else _load("arrests")
    if df is None or "arrest_boro" not in df.columns:
        return RichResult(title="NYPD arrests by borough",
                          warnings=["missing column: arrest_boro"])
    counts = _counts(df["arrest_boro"])
    return RichResult(
        title="NYPD arrests by borough",
        tables=[{"title": "By borough", "headers": ["Boro", "Count"],
                 "rows": [[str(k), int(v)] for k, v in counts.items()]}],
        payload={"by_boro": {str(k): int(v) for k, v in counts.items()}},
    )


def analyze_felony_race_disparity(
        arrests_df: pd.DataFrame | None = None) -> RichResult:
    """Disparate impact + demographic parity of felony (vs lesser) charging."""
    df = arrests_df if arrests_df is not None else _load("arrests")
    if df is None or not {"perp_race", "law_cat_cd"}.issubset(df.columns):
        return RichResult(title="NYPD felony-charge race disparity",
                          warnings=["missing column(s): perp_race, law_cat_cd"])
    sub = df[["perp_race", "law_cat_cd"]].dropna()
    race = sub["perp_race"].astype(str)
    felony = (sub["law_cat_cd"].astype(str).str.strip().str.upper() == "F").astype(int)
    keep = race.str.len() > 0
    race, felony = race[keep], felony[keep]
    di = fairness_disparate_impact(y_pred=felony.tolist(), group=race.tolist())
    dp = fairness_demographic_parity(y_pred=felony.tolist(), group=race.tolist())
    rates = felony.groupby(race).mean()
    return RichResult(
        title="NYPD felony-charge race disparity",
        summary_lines=[("groups", int(race.nunique())),
                       ("records", int(len(race)))],
        tables=[{"title": "Felony rate by race",
                 "headers": ["Race", "Felony rate"],
                 "rows": [[str(k), f"{float(v):.3f}"] for k, v in rates.items()]}],
        payload={"disparate_impact": _jsonable(di),
                 "demographic_parity": _jsonable(dp),
                 "felony_rate_by_race": {str(k): float(v)
                                         for k, v in rates.items()}},
    )


def analyze_complaints_by_race(
        complaint_df: pd.DataFrame | None = None) -> RichResult:
    """Complaint volume by recorded suspect race."""
    df = complaint_df if complaint_df is not None else _load("complaint")
    if df is None or "susp_race" not in df.columns:
        return RichResult(title="NYPD complaints: suspect race",
                          warnings=["missing column: susp_race"])
    counts = _counts(df["susp_race"])
    return RichResult(
        title="NYPD complaints: suspect race",
        tables=[{"title": "By suspect race", "headers": ["Race", "Count"],
                 "rows": [[str(k), int(v)] for k, v in counts.items()]}],
        payload={"by_race": {str(k): int(v) for k, v in counts.items()}},
    )


_SURFACES = {
    "arrests_by_offense": analyze_arrests_by_offense,
    "arrests_by_boro": analyze_arrests_by_boro,
    "felony_race_disparity": analyze_felony_race_disparity,
    "complaints_by_race": analyze_complaints_by_race,
}


def analyze_all(*, out_dir: str | None = None) -> dict[str, RichResult]:
    """Run every NYPD surface; returns a name -> RichResult mapping."""
    results: dict[str, RichResult] = {}
    for name, fn in _SURFACES.items():
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- surface-level isolation
            results[name] = RichResult(title=f"nypd.{name} (failed)",
                                       warnings=[f"{type(exc).__name__}: {exc}"])
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"nypd_{name}.json").write_text(
                json.dumps(_jsonable(results[name].payload), default=str))
    return results


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj
