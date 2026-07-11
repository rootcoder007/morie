"""morie.cpd_all_analyze -- Chicago Police Department analysis aggregator.

Python parity with R's ``morie_cpd_all_analyses``. Loads Chicago crime +
arrests data (bundled samples offline by default) and runs the applicable
descriptive, predictive-policing, and fairness surfaces, each emitting a
RichResult. Mirrors ``morie.otis_all_analyze``.

    analyze_<name>(...) -> RichResult
    analyze_all(*, out_dir=None) -> dict[str, RichResult]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .fairness.metrics import fairness_disparate_impact
from .fairness.predpol import predpol_aggregate_areas
from .fn._richresult import RichResult

_DATA = Path(__file__).resolve().parent / "data"


def _load(which: str) -> pd.DataFrame | None:
    fname = {
        "crime": "chicago_crime_synthetic.csv",
        "arrests": "chicago_arrests_dpt3_jri9_sample.csv",
    }[which]
    path = _DATA / fname
    if not path.exists():
        return None
    return pd.read_csv(path)


def analyze_crime_by_type(crime_df: pd.DataFrame | None = None) -> RichResult:
    """Incident volume by Chicago primary crime type."""
    df = crime_df if crime_df is not None else _load("crime")
    if df is None or "primary_type" not in df.columns:
        return RichResult(title="CPD crimes by primary type",
                          warnings=["missing column: primary_type"])
    counts = df["primary_type"].value_counts()
    rows = [[k, int(v)] for k, v in counts.items()]
    return RichResult(
        title="CPD crimes by primary type",
        summary_lines=[("distinct types", int(len(counts))),
                       ("total records", int(counts.sum()))],
        tables=[{"title": "By primary type", "headers": ["Type", "Count"],
                 "rows": rows}],
        payload={"by_type": {str(k): int(v) for k, v in counts.items()}},
    )


def analyze_arrests_by_area(crime_df: pd.DataFrame | None = None) -> RichResult:
    """Predpol area aggregation: arrest concentration by community area."""
    df = crime_df if crime_df is not None else _load("crime")
    if df is None or not {"community_area", "arrest"}.issubset(df.columns):
        return RichResult(title="CPD arrest concentration by community area",
                          warnings=["missing column(s): community_area, arrest"])
    area = df["community_area"].astype(str)
    arrest = df["arrest"].astype(str).str.lower().isin(
        ["true", "1", "t", "yes"]).astype(int)
    risk = arrest.groupby(area).transform("mean")
    agg = predpol_aggregate_areas(area=area.tolist(), risk=risk.tolist(),
                                  outcome=arrest.tolist())
    return RichResult(
        title="CPD arrest concentration by community area",
        summary_lines=[("areas", int(area.nunique()))],
        tables=[{"title": "Area aggregate (predpol)",
                 "headers": ["field", "value"],
                 "rows": [[k, str(v)[:60]] for k, v in dict(agg).items()]}],
        payload={"aggregate": _jsonable(agg)},
    )


def analyze_temporal(crime_df: pd.DataFrame | None = None) -> RichResult:
    """Incident counts by year."""
    df = crime_df if crime_df is not None else _load("crime")
    if df is None or "year" not in df.columns:
        return RichResult(title="CPD temporal trend",
                          warnings=["missing column: year"])
    counts = df["year"].value_counts().sort_index()
    return RichResult(
        title="CPD temporal trend",
        summary_lines=[("years", int(len(counts)))],
        tables=[{"title": "By year", "headers": ["Year", "Count"],
                 "rows": [[str(k), int(v)] for k, v in counts.items()]}],
        payload={"by_year": {str(k): int(v) for k, v in counts.items()}},
    )


def analyze_arrest_race_disparity(
        arrests_df: pd.DataFrame | None = None) -> RichResult:
    """Disparate-impact of arrest representation across race categories."""
    df = arrests_df if arrests_df is not None else _load("arrests")
    if df is None or "race" not in df.columns:
        return RichResult(title="CPD arrests: race disparity",
                          warnings=["missing column: race (arrests data)"])
    race = df["race"].dropna().astype(str)
    race = race[race.str.len() > 0]
    di = fairness_disparate_impact(y_pred=[1] * len(race), group=race.tolist())
    counts = race.value_counts()
    return RichResult(
        title="CPD arrests: race disparity",
        summary_lines=[("groups", int(race.nunique())),
                       ("records", int(len(race)))],
        tables=[{"title": "By race", "headers": ["Race", "Count"],
                 "rows": [[str(k), int(v)] for k, v in counts.items()]}],
        payload={"disparate_impact": _jsonable(di),
                 "counts": {str(k): int(v) for k, v in counts.items()}},
    )


_SURFACES = {
    "crime_by_type": analyze_crime_by_type,
    "arrests_by_area": analyze_arrests_by_area,
    "temporal": analyze_temporal,
    "arrest_race_disparity": analyze_arrest_race_disparity,
}


def analyze_all(*, out_dir: str | None = None) -> dict[str, RichResult]:
    """Run every CPD surface; returns a name -> RichResult mapping."""
    results: dict[str, RichResult] = {}
    for name, fn in _SURFACES.items():
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- surface-level isolation
            results[name] = RichResult(title=f"cpd.{name} (failed)",
                                       warnings=[f"{type(exc).__name__}: {exc}"])
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"cpd_{name}.json").write_text(
                json.dumps(_jsonable(results[name].payload), default=str))
    return results


def _jsonable(obj: Any) -> Any:
    """Best-effort convert numpy/pandas scalars + containers to JSON types."""
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
