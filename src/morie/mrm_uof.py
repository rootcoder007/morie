# SPDX-License-Identifier: AGPL-3.0-or-later
"""MRM-framework analyses on Use-of-Force (UOF) data.

Six rich-output callables for the MRM-UOF analytic suite. Each
function returns a :class:`morie.fn._richresult.RichResult` carrying a
multi-paragraph natural-language interpretation, a structured table /
section payload for programmatic access, and an explicit warnings
list. The functions mirror the conceptual structure of the R-side
``r-package/morie/R/mrm_uof.R`` and the Python-side ``mrm_otis``
family, but address the use-of-force domain: concentration of force
incidents across services, weapon-by-service contingency, year-on-year
change with optional change-point detection, region locality stability
across reporting periods, demographic disparity in outcome rates with
risk-ratio confidence intervals, and a CKAN-aware data-quality audit.

All callables are pure (no IO, no class state), accept ``pandas``
inputs, depend only on ``numpy`` + ``scipy.stats`` (with an optional
soft import of ``ruptures``), and produce results suitable for
inclusion in MRM empirical papers without further post-processing.

Functions
---------
mrm_uof_force_concentration
    Hill-MLE Pareto alpha + Gini + top-5 / top-10 share for incidents
    aggregated by force / service.
mrm_uof_weapon_diversity
    Weapon-by-service contingency: chi-square, Cramer's V, top-3
    standardised residuals.
mrm_uof_yoy_change
    Year-on-year percentage change with optional ``ruptures``-based
    change-point detection (PELT / rbf) or a manual largest-gap
    fallback.
mrm_uof_region_locality
    Region-at-time vs region-now contingency: diagonal share,
    chi-square and Cramer's V.
mrm_uof_demographic_disparity
    Per-category outcome rates with Wilson 95 percent intervals,
    risk-ratio relative to a baseline group, optional bootstrap
    percentile interval on the risk ratio.
mrm_uof_data_quality_audit
    Schema and null audit; CKAN-sidecar aware, optional duck-typed
    schema comparison, suspect-flag interpretation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from morie.fn._richresult import RichResult

__all__ = [
    "mrm_uof_force_concentration",
    "mrm_uof_weapon_diversity",
    "mrm_uof_yoy_change",
    "mrm_uof_region_locality",
    "mrm_uof_demographic_disparity",
    "mrm_uof_data_quality_audit",
]


# ─── internal helpers ────────────────────────────────────────────────


def _gini(x: np.ndarray) -> float:
    """Gini coefficient on a non-negative vector.

    Uses the standard sorted formulation
    ``(2 * sum(i * x_(i)) - (n + 1) * sum(x)) / (n * sum(x))``.
    Returns ``nan`` for empty input or zero-sum input.
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s == 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * idx.dot(x) - (n + 1) * s) / (n * s))


def _hill_alpha(x: np.ndarray, x_min: float = 1.0) -> float:
    """Hill maximum-likelihood Pareto tail exponent (Newman 2005).

    ``alpha_hat = 1 + n / sum_i log(x_i / x_min)``.

    Returns ``nan`` if fewer than 2 values exceed ``x_min``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= x_min]
    if x.size < 2:
        return float("nan")
    denom = np.log(x / x_min).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 + x.size / denom)


def _topk_share(x: np.ndarray, k: int) -> float:
    """Share of total mass held by the top-``k`` values."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if s == 0 or x.size == 0:
        return float("nan")
    k = min(k, x.size)
    return float(np.sort(x)[::-1][:k].sum() / s)


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95 percent confidence interval with continuity
    correction.

    Parameters
    ----------
    k : int
        Successes.
    n : int
        Trials.
    z : float
        Standard-normal critical value (default ``z_{0.975}``).

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds, clipped to ``[0, 1]``. Returns
        ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    # Continuity-corrected half-width
    try:
        half_lo = (z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1 - p) + (4 * p - 2)) + 1.0) / (2.0 * (n + z2))
        half_hi = (z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1 - p) - (4 * p - 2)) + 1.0) / (2.0 * (n + z2))
        lo = max(0.0, centre - half_lo) if n * p >= 1 else 0.0
        hi = min(1.0, centre + half_hi) if n * (1 - p) >= 1 else 1.0
    except ValueError:
        lo = float("nan")
        hi = float("nan")
    # Sanity: if continuity formula went pathological (sqrt of negative),
    # fall back to uncorrected Wilson.
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
        lo = max(0.0, centre - margin)
        hi = min(1.0, centre + margin)
    return (float(lo), float(hi))


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Cramer's V from a chi-square statistic on an ``r x c`` table.

    Returns ``nan`` for degenerate tables (single row or column, or
    empty ``n``).
    """
    k = min(r - 1, c - 1)
    if k <= 0 or n == 0:
        return float("nan")
    return float(math.sqrt(chi2 / (n * k)))


def _fmt_pct(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    return f"{100.0 * p:.2f}%"


# ─── 1. force concentration ─────────────────────────────────────────


def mrm_uof_force_concentration(
    df: pd.DataFrame,
    force_col: str,
    count_col: str | None = None,
) -> RichResult:
    """Concentration of UOF incidents across forces / services.

    Aggregates per-force incident counts and reports a Hill-MLE Pareto
    tail exponent, the Gini coefficient, and the top-5 / top-10
    concentration shares.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format UOF data with one row per incident (when
        ``count_col`` is ``None``) or one row per force-period with a
        numeric count column.
    force_col : str
        Column identifying the force / service / agency.
    count_col : str, optional
        If supplied, the per-row incident count to sum within each
        force; otherwise each row counts as one incident.

    Returns
    -------
    RichResult
        ``payload`` keys: ``n_forces``, ``n_incidents``,
        ``pareto_alpha_mle``, ``gini``, ``top5_share``, ``top10_share``,
        ``counts`` (list of (force, count) tuples sorted descending),
        ``n``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"force": ["A"] * 50 + ["B"] * 5})
    >>> r = mrm_uof_force_concentration(df, "force")
    >>> r.n_forces
    2
    """
    call = f"mrm_uof_force_concentration(df=<{len(df)} rows>, force_col={force_col!r}, count_col={count_col!r})"
    warnings: list[str] = []

    if force_col not in df.columns:
        return RichResult(
            title="MRM-UOF Force Concentration",
            call=call,
            warnings=[f"force_col {force_col!r} not in dataframe columns"],
            interpretation=(f"No analysis: column {force_col!r} is absent from the supplied dataframe."),
            payload={"n": 0, "n_forces": 0, "n_incidents": 0},
        )

    if count_col is None:
        counts_series = df.groupby(force_col).size()
    else:
        if count_col not in df.columns:
            return RichResult(
                title="MRM-UOF Force Concentration",
                call=call,
                warnings=[f"count_col {count_col!r} not in dataframe columns"],
                interpretation=(f"No analysis: count column {count_col!r} is absent."),
                payload={"n": 0, "n_forces": 0, "n_incidents": 0},
            )
        counts_series = df.groupby(force_col)[count_col].sum(min_count=1).fillna(0)

    counts_series = counts_series.sort_values(ascending=False)
    x = counts_series.to_numpy(dtype=np.float64)
    n_forces = int(x.size)
    n_incidents = int(x.sum())

    if n_incidents == 0:
        warnings.append("All counts are zero; concentration is undefined.")
        return RichResult(
            title="MRM-UOF Force Concentration",
            call=call,
            warnings=warnings,
            interpretation=("All per-force counts are zero, so no concentration statistics can be computed."),
            payload={
                "n": 0,
                "n_forces": n_forces,
                "n_incidents": 0,
                "pareto_alpha_mle": float("nan"),
                "gini": float("nan"),
                "top5_share": float("nan"),
                "top10_share": float("nan"),
                "counts": list(counts_series.items()),
            },
        )

    if n_forces < 10:
        warnings.append(
            f"Only {n_forces} force(s); concentration statistics with n<10 categories are descriptive at best."
        )

    pareto_alpha = _hill_alpha(x, x_min=1.0)
    gini_val = _gini(x)
    top5 = _topk_share(x, 5)
    top10 = _topk_share(x, 10)

    summary_lines: list[tuple[str, Any]] = [
        ("Forces (n)", n_forces),
        ("Incidents (n)", n_incidents),
        ("Pareto alpha (Hill MLE)", pareto_alpha),
        ("Gini coefficient", gini_val),
        ("Top-5 share", top5),
        ("Top-10 share", top10),
    ]

    # Per-force table (capped to top 25)
    head_rows: list[list[Any]] = [
        [str(name), int(c), float(c) / n_incidents] for name, c in counts_series.head(25).items()
    ]

    sections = [
        {
            "title": "Per-force incident counts (top 25)",
            "headers": ["force", "n", "share"],
            "table": head_rows,
        }
    ]

    # Interpretation paragraphs
    if math.isfinite(pareto_alpha):
        if pareto_alpha < 2.0:
            tail_text = (
                "An alpha below 2 indicates a very heavy upper tail: a "
                "small number of forces accounts for a disproportionate "
                "share of incidents, and the variance of incident counts "
                "across forces is theoretically unbounded."
            )
        elif pareto_alpha < 3.0:
            tail_text = (
                "An alpha between 2 and 3 indicates a heavy tail with "
                "finite mean but infinite variance in the underlying "
                "power-law model; concentration is substantial."
            )
        else:
            tail_text = (
                "An alpha above 3 indicates a relatively thin tail; "
                "incidents are spread comparatively evenly across forces "
                "given a power-law fit."
            )
    else:
        tail_text = (
            "Pareto alpha could not be estimated; this typically occurs "
            "when fewer than two forces have counts above the x_min "
            "threshold of 1."
        )

    if math.isfinite(gini_val):
        if gini_val < 0.30:
            gini_text = (
                "A Gini coefficient below 0.30 indicates an approximately even distribution of incidents across forces."
            )
        elif gini_val < 0.60:
            gini_text = "A Gini coefficient between 0.30 and 0.60 indicates moderate concentration."
        else:
            gini_text = (
                "A Gini coefficient at or above 0.60 indicates strong "
                "concentration: a few forces carry most of the volume."
            )
    else:
        gini_text = "Gini coefficient could not be computed."

    interpretation = (
        f"Across {n_forces} force(s) and {n_incidents} total incident(s), "
        f"the top-5 forces hold {_fmt_pct(top5)} of recorded volume and "
        f"the top-10 hold {_fmt_pct(top10)}. {gini_text} {tail_text}"
    )

    return RichResult(
        title="MRM-UOF Force Concentration",
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": n_forces,
            "n_forces": n_forces,
            "n_incidents": n_incidents,
            "pareto_alpha_mle": pareto_alpha,
            "gini": gini_val,
            "top5_share": top5,
            "top10_share": top10,
            "counts": [(str(k), int(v)) for k, v in counts_series.items()],
            "statistic": gini_val,
        },
    )


# ─── 2. weapon diversity ─────────────────────────────────────────────


def mrm_uof_weapon_diversity(
    df: pd.DataFrame,
    weapon_col: str,
    force_col: str,
) -> RichResult:
    """Weapon-by-force contingency test.

    Builds a weapon x force contingency table, runs a chi-square test
    of independence, computes Cramer's V, and reports the top-3
    (weapon, force) cells by standardised Pearson residual.
    """
    call = f"mrm_uof_weapon_diversity(df=<{len(df)} rows>, weapon_col={weapon_col!r}, force_col={force_col!r})"
    warnings: list[str] = []

    for col in (weapon_col, force_col):
        if col not in df.columns:
            return RichResult(
                title="MRM-UOF Weapon Diversity",
                call=call,
                warnings=[f"column {col!r} not in dataframe"],
                interpretation=(f"No analysis: required column {col!r} is missing."),
                payload={"n": 0},
            )

    table = pd.crosstab(df[weapon_col], df[force_col])
    n = int(table.values.sum())
    r, c = table.shape

    if n == 0 or r < 2 or c < 2:
        return RichResult(
            title="MRM-UOF Weapon Diversity",
            call=call,
            warnings=["Contingency table is degenerate (n=0 or single row/column)."],
            interpretation=(
                "No analysis: the weapon-by-force contingency table has "
                "fewer than two rows or columns, or contains no data."
            ),
            payload={"n": n, "table_shape": (r, c)},
        )

    chi2, pvalue, dof, expected = stats.chi2_contingency(table.values)
    chi2 = float(chi2)
    pvalue = float(pvalue)
    dof = int(dof)
    v = _cramers_v(chi2, n, r, c)

    low_expected = bool((expected < 5).any())
    n_low = int((expected < 5).sum())
    if low_expected:
        warnings.append(
            f"{n_low} of {expected.size} expected cell(s) are below 5; "
            f"chi-square approximation may be unreliable -- consider a "
            f"Fisher / Monte Carlo alternative."
        )

    # Standardised Pearson residuals
    obs = table.values.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_resid = (obs - expected) / np.sqrt(expected)
    abs_resid = np.abs(std_resid)
    flat_idx = np.argsort(abs_resid, axis=None)[::-1][:3]
    top_resid_rows: list[list[Any]] = []
    top_resid_payload: list[dict[str, Any]] = []
    for fi in flat_idx:
        ri, ci = np.unravel_index(fi, abs_resid.shape)
        w_label = str(table.index[ri])
        f_label = str(table.columns[ci])
        top_resid_rows.append(
            [
                w_label,
                f_label,
                int(obs[ri, ci]),
                float(expected[ri, ci]),
                float(std_resid[ri, ci]),
            ]
        )
        top_resid_payload.append(
            {
                "weapon": w_label,
                "force": f_label,
                "observed": int(obs[ri, ci]),
                "expected": float(expected[ri, ci]),
                "std_residual": float(std_resid[ri, ci]),
            }
        )

    summary_lines: list[tuple[str, Any]] = [
        ("N incidents", n),
        ("Table shape", f"{r} weapons x {c} forces"),
        ("Chi-square", chi2),
        ("Degrees of freedom", dof),
        ("p-value", pvalue),
        ("Cramer's V", v),
    ]

    sections = [
        {
            "title": "Top-3 standardised Pearson residuals",
            "headers": ["weapon", "force", "obs", "expected", "std_resid"],
            "table": top_resid_rows,
        }
    ]

    if math.isfinite(v):
        if v < 0.10:
            assoc_text = "Association is negligible (Cramer's V < 0.10)."
        elif v < 0.30:
            assoc_text = "Association is weak (0.10 <= V < 0.30)."
        elif v < 0.50:
            assoc_text = "Association is moderate (0.30 <= V < 0.50)."
        else:
            assoc_text = "Association is strong (V >= 0.50)."
    else:
        assoc_text = "Association strength could not be computed."

    sig_text = f"The chi-square test on a {r} x {c} table yields chi2={chi2:.4f} on {dof} df (p={pvalue:.4g})."

    interpretation = (
        f"{sig_text} {assoc_text} The strongest deviation from "
        f"independence sits at "
        f"({top_resid_payload[0]['weapon']}, "
        f"{top_resid_payload[0]['force']}) with a standardised residual "
        f"of {top_resid_payload[0]['std_residual']:+.2f}."
    )

    return RichResult(
        title="MRM-UOF Weapon Diversity",
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": n,
            "chi2": chi2,
            "pvalue": pvalue,
            "df": dof,
            "cramers_v": v,
            "table_shape": (r, c),
            "low_expected_count": low_expected,
            "top_residuals": top_resid_payload,
            "statistic": chi2,
        },
    )


# ─── 3. year-on-year change ──────────────────────────────────────────


def mrm_uof_yoy_change(
    dfs_by_year: Mapping[int, pd.DataFrame] | None = None,
    df: pd.DataFrame | None = None,
    year_col: str | None = None,
    count_col: str | None = None,
) -> RichResult:
    """Year-on-year change in incident counts with optional change-point.

    Either supply ``dfs_by_year`` (a mapping from integer year to a
    DataFrame) or ``df`` together with ``year_col``. When ``count_col``
    is ``None`` the incident count is the number of rows; otherwise it
    is the sum of ``count_col`` within each year.
    """
    call = (
        f"mrm_uof_yoy_change(dfs_by_year=<{'n/a' if dfs_by_year is None else len(dfs_by_year)} years>, "
        f"df=<{'None' if df is None else f'{len(df)} rows'}>, "
        f"year_col={year_col!r}, count_col={count_col!r})"
    )
    warnings: list[str] = []

    if dfs_by_year is None and df is None:
        return RichResult(
            title="MRM-UOF Year-on-Year Change",
            call=call,
            warnings=["Neither dfs_by_year nor df was supplied."],
            interpretation="No analysis: no input data supplied.",
            payload={"n": 0},
        )

    if dfs_by_year is not None:
        years_sorted = sorted(int(y) for y in dfs_by_year.keys())
        counts: list[int] = []
        for y in years_sorted:
            sub = dfs_by_year[y]
            if count_col is None:
                counts.append(int(len(sub)))
            else:
                if count_col not in sub.columns:
                    warnings.append(f"Year {y}: count_col {count_col!r} missing; treated as 0.")
                    counts.append(0)
                else:
                    counts.append(int(sub[count_col].sum()))
    else:
        assert df is not None
        if year_col is None or year_col not in df.columns:
            return RichResult(
                title="MRM-UOF Year-on-Year Change",
                call=call,
                warnings=[f"year_col {year_col!r} missing from dataframe."],
                interpretation=(f"No analysis: year column {year_col!r} is absent from the supplied dataframe."),
                payload={"n": 0},
            )
        if count_col is None:
            grouped = df.groupby(year_col).size().sort_index()
        else:
            if count_col not in df.columns:
                return RichResult(
                    title="MRM-UOF Year-on-Year Change",
                    call=call,
                    warnings=[f"count_col {count_col!r} missing from dataframe."],
                    interpretation=(f"No analysis: count column {count_col!r} is absent."),
                    payload={"n": 0},
                )
            grouped = df.groupby(year_col)[count_col].sum(min_count=1).fillna(0).sort_index()
        years_sorted = [int(y) for y in grouped.index.tolist()]
        counts = [int(c) for c in grouped.tolist()]

    n_years = len(years_sorted)

    if n_years == 0:
        return RichResult(
            title="MRM-UOF Year-on-Year Change",
            call=call,
            warnings=["No years present."],
            interpretation="No analysis: no years were found.",
            payload={"n": 0, "years": [], "counts": []},
        )

    counts_arr = np.asarray(counts, dtype=np.float64)
    yoy_pct: list[float | None] = [None]
    for i in range(1, n_years):
        prev = counts_arr[i - 1]
        if prev == 0:
            yoy_pct.append(None)
        else:
            yoy_pct.append(float((counts_arr[i] - prev) / prev * 100.0))

    # Change-point detection
    change_point_year: int | None = None
    change_point_method = "none"
    if n_years < 3:
        warnings.append(f"Only {n_years} year(s) supplied; too few for change-point detection.")
    else:
        try:
            import ruptures as rpt  # type: ignore

            algo = rpt.Pelt(model="rbf").fit(counts_arr.reshape(-1, 1))
            bkps = algo.predict(pen=10)
            # ruptures returns breakpoints as INDEX-AFTER positions, with
            # the final entry == n. Drop the trailing sentinel.
            real_bkps = [b for b in bkps if b < n_years]
            if real_bkps:
                change_point_year = int(years_sorted[real_bkps[0]])
                change_point_method = "ruptures.Pelt(model='rbf', pen=10)"
        except ImportError:
            diffs = np.abs(np.diff(counts_arr))
            if diffs.size > 0:
                idx = int(np.argmax(diffs)) + 1  # change at year[idx]
                change_point_year = int(years_sorted[idx])
                change_point_method = "largest-abs-diff fallback (ruptures not installed)"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"ruptures failed ({type(exc).__name__}); using fallback.")
            diffs = np.abs(np.diff(counts_arr))
            if diffs.size > 0:
                idx = int(np.argmax(diffs)) + 1
                change_point_year = int(years_sorted[idx])
                change_point_method = "largest-abs-diff fallback (ruptures errored)"

    finite_yoy = [v for v in yoy_pct if v is not None and math.isfinite(v)]
    mean_abs_yoy = float(np.mean([abs(v) for v in finite_yoy])) if finite_yoy else float("nan")

    summary_lines: list[tuple[str, Any]] = [
        ("Years (n)", n_years),
        ("Total incidents", int(counts_arr.sum())),
        ("Mean |YoY| %", mean_abs_yoy),
        ("Change-point year", change_point_year if change_point_year is not None else "none"),
        ("Change-point method", change_point_method),
    ]

    table_rows: list[list[Any]] = []
    for y, c, p in zip(years_sorted, counts, yoy_pct):
        table_rows.append([y, c, "—" if p is None else f"{p:+.2f}%"])

    sections = [
        {
            "title": "Per-year counts and YoY change",
            "headers": ["year", "count", "yoy_pct"],
            "table": table_rows,
        }
    ]

    if change_point_year is not None:
        cp_text = f"A structural break was identified at year {change_point_year} using {change_point_method}."
    else:
        cp_text = "No change-point could be identified."

    if math.isfinite(mean_abs_yoy):
        vol_text = f"Mean absolute year-on-year change across {len(finite_yoy)} transition(s) is {mean_abs_yoy:.2f}%."
    else:
        vol_text = "Year-on-year volatility is undefined (insufficient transitions)."

    interpretation = (
        f"Series spans {n_years} year(s) "
        f"({years_sorted[0]}-{years_sorted[-1]}) "
        f"with a total of {int(counts_arr.sum())} incident(s). "
        f"{vol_text} {cp_text}"
    )

    return RichResult(
        title="MRM-UOF Year-on-Year Change",
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": n_years,
            "years": years_sorted,
            "counts": counts,
            "yoy_pct": yoy_pct,
            "change_point_year": change_point_year,
            "change_point_method": change_point_method,
            "mean_abs_yoy_pct": mean_abs_yoy,
            "value": mean_abs_yoy,
        },
    )


# ─── 4. region locality ──────────────────────────────────────────────


def mrm_uof_region_locality(
    df: pd.DataFrame,
    region_at_col: str,
    region_now_col: str,
) -> RichResult:
    """Region-at-time vs region-now locality contingency."""
    call = (
        f"mrm_uof_region_locality(df=<{len(df)} rows>, "
        f"region_at_col={region_at_col!r}, region_now_col={region_now_col!r})"
    )
    warnings: list[str] = []

    for col in (region_at_col, region_now_col):
        if col not in df.columns:
            return RichResult(
                title="MRM-UOF Region Locality",
                call=call,
                warnings=[f"column {col!r} missing"],
                interpretation=(f"No analysis: required column {col!r} is absent."),
                payload={"n": 0},
            )

    pair = df[[region_at_col, region_now_col]].copy()
    n_pre = int(len(pair))
    pair = pair.dropna()
    n_dropped = n_pre - int(len(pair))
    if n_dropped > 0:
        warnings.append(f"Dropped {n_dropped} row(s) with NaN in region columns.")

    if len(pair) == 0:
        return RichResult(
            title="MRM-UOF Region Locality",
            call=call,
            warnings=warnings + ["Empty contingency after NaN drop."],
            interpretation=("No analysis: no rows remain after dropping NaN region values."),
            payload={"n": 0, "n_dropped": n_dropped},
        )

    table = pd.crosstab(pair[region_at_col], pair[region_now_col])
    all_labels = sorted(set(table.index) | set(table.columns), key=lambda s: str(s))
    table = table.reindex(index=all_labels, columns=all_labels, fill_value=0)

    obs = table.values.astype(np.float64)
    n = int(obs.sum())
    diag = float(np.trace(obs))
    diagonal_share = diag / n if n > 0 else float("nan")

    r, c = obs.shape
    if r >= 2 and c >= 2:
        chi2, pvalue, dof, expected = stats.chi2_contingency(obs)
        chi2 = float(chi2)
        pvalue = float(pvalue)
        dof = int(dof)
        v = _cramers_v(chi2, n, r, c)
        if (expected < 5).any():
            warnings.append("One or more expected cell counts below 5; chi-square approximation may be unreliable.")
    else:
        chi2 = float("nan")
        pvalue = float("nan")
        dof = 0
        v = float("nan")
        warnings.append("Contingency table too small for chi-square test.")

    if n < 30:
        warnings.append(f"Sample size n={n} is small; locality statistics are descriptive only.")

    summary_lines: list[tuple[str, Any]] = [
        ("N pairs", n),
        ("Categories", f"{r} x {c}"),
        ("Diagonal share", diagonal_share),
        ("Chi-square", chi2),
        ("Degrees of freedom", dof),
        ("p-value", pvalue),
        ("Cramer's V", v),
    ]

    table_rows: list[list[Any]] = [[str(lab)] + [int(v) for v in row] for lab, row in zip(table.index, obs)]
    sections = [
        {
            "title": "Region contingency (at-time x now)",
            "headers": ["region\\region"] + [str(c) for c in table.columns],
            "table": table_rows,
        }
    ]

    if math.isfinite(diagonal_share):
        if diagonal_share >= 0.75:
            loc_text = (
                f"Diagonal share is {_fmt_pct(diagonal_share)}, indicating "
                f"high locality stability -- most subjects remain in their "
                f"original region."
            )
        elif diagonal_share >= 0.50:
            loc_text = f"Diagonal share is {_fmt_pct(diagonal_share)}, indicating moderate locality stability."
        else:
            loc_text = f"Diagonal share is {_fmt_pct(diagonal_share)}, indicating substantial cross-regional movement."
    else:
        loc_text = "Diagonal share could not be computed."

    interpretation = (
        f"{loc_text} "
        f"Chi-square test on the {r} x {c} contingency table yields "
        f"chi2={chi2:.4f} on {dof} df (p={pvalue:.4g}); Cramer's V is "
        f"{v:.4f}."
    )

    return RichResult(
        title="MRM-UOF Region Locality",
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": n,
            "n_dropped": n_dropped,
            "diagonal_share": diagonal_share,
            "chi2": chi2,
            "pvalue": pvalue,
            "df": dof,
            "cramers_v": v,
            "table_shape": (r, c),
            "value": diagonal_share,
        },
    )


# ─── 5. demographic disparity ────────────────────────────────────────


def mrm_uof_demographic_disparity(
    df: pd.DataFrame,
    demo_col: str,
    outcome_col: str,
    baseline: str | None = None,
    bootstrap_reps: int = 0,
) -> RichResult:
    """Demographic disparity in outcome rates with risk-ratio CIs."""
    call = (
        f"mrm_uof_demographic_disparity(df=<{len(df)} rows>, "
        f"demo_col={demo_col!r}, outcome_col={outcome_col!r}, "
        f"baseline={baseline!r}, bootstrap_reps={bootstrap_reps})"
    )
    warnings: list[str] = []

    for col in (demo_col, outcome_col):
        if col not in df.columns:
            return RichResult(
                title="MRM-UOF Demographic Disparity",
                call=call,
                warnings=[f"column {col!r} missing"],
                interpretation=(f"No analysis: required column {col!r} is absent."),
                payload={"n": 0},
            )

    sub = df[[demo_col, outcome_col]].dropna().copy()
    if len(sub) == 0:
        return RichResult(
            title="MRM-UOF Demographic Disparity",
            call=call,
            warnings=["No non-null rows."],
            interpretation="No analysis: no non-null rows after dropping NaN.",
            payload={"n": 0},
        )

    y = sub[outcome_col]
    if y.dtype == bool:
        y_int = y.astype(np.int64)
    else:
        try:
            y_int = y.astype(np.int64)
        except (TypeError, ValueError):
            warnings.append(f"outcome_col {outcome_col!r} could not be coerced to int; treating non-zero as 1.")
            y_int = (y.astype(float) != 0).astype(np.int64)
    sub = sub.assign(**{outcome_col: y_int})

    grouped = sub.groupby(demo_col)[outcome_col].agg(["count", "sum"])
    grouped.columns = ["n", "k"]
    grouped["rate"] = grouped["k"] / grouped["n"]
    grouped = grouped.sort_values("rate", ascending=False)

    if baseline is None:
        baseline = grouped["n"].idxmax()
    if baseline not in grouped.index:
        return RichResult(
            title="MRM-UOF Demographic Disparity",
            call=call,
            warnings=[f"baseline {baseline!r} not present in demo column"],
            interpretation=(f"No analysis: baseline category {baseline!r} is absent."),
            payload={"n": int(len(sub))},
        )

    baseline_rate = float(grouped.loc[baseline, "rate"])
    if baseline_rate < 0.01:
        warnings.append(
            f"Baseline rate {baseline_rate:.4f} is below 1%; risk ratios "
            f"against this baseline are unstable -- treat with caution."
        )

    per_cat: list[dict[str, Any]] = []
    rr_table_rows: list[list[Any]] = []

    for cat, row in grouped.iterrows():
        n_i = int(row["n"])
        k_i = int(row["k"])
        rate = float(row["rate"])
        lo, hi = _wilson_ci(k_i, n_i)
        if baseline_rate > 0:
            rr = rate / baseline_rate
        else:
            rr = float("nan")
        rr_lo: float | None = None
        rr_hi: float | None = None

        if n_i < 30:
            warnings.append(f"Group {cat!r} has n={n_i} < 30; confidence intervals are wide.")

        if bootstrap_reps > 0 and cat != baseline:
            sub_cat = sub[sub[demo_col] == cat][outcome_col].to_numpy()
            sub_base = sub[sub[demo_col] == baseline][outcome_col].to_numpy()
            if sub_cat.size > 0 and sub_base.size > 0:
                rng = np.random.default_rng(0)
                rr_draws = np.empty(int(bootstrap_reps), dtype=np.float64)
                for b in range(int(bootstrap_reps)):
                    bi = rng.integers(0, sub_cat.size, sub_cat.size)
                    bj = rng.integers(0, sub_base.size, sub_base.size)
                    rate_i = sub_cat[bi].mean()
                    rate_j = sub_base[bj].mean()
                    rr_draws[b] = rate_i / rate_j if rate_j > 0 else np.nan
                rr_draws = rr_draws[np.isfinite(rr_draws)]
                if rr_draws.size >= 20:
                    rr_lo, rr_hi = (
                        float(np.percentile(rr_draws, 2.5)),
                        float(np.percentile(rr_draws, 97.5)),
                    )

        per_cat.append(
            {
                "category": str(cat),
                "n": n_i,
                "k": k_i,
                "rate": rate,
                "lo": lo,
                "hi": hi,
                "rr": rr,
                "rr_lo": rr_lo,
                "rr_hi": rr_hi,
                "baseline": cat == baseline,
            }
        )
        rr_table_rows.append(
            [
                str(cat) + (" (baseline)" if cat == baseline else ""),
                n_i,
                rate,
                lo,
                hi,
                rr,
                "—" if rr_lo is None else float(rr_lo),
                "—" if rr_hi is None else float(rr_hi),
            ]
        )

    summary_lines: list[tuple[str, Any]] = [
        ("N subjects", int(len(sub))),
        ("Categories", int(len(grouped))),
        ("Baseline", baseline),
        ("Baseline rate", baseline_rate),
    ]

    tables = [
        {
            "title": "Per-category outcome rates and risk ratios",
            "headers": ["category", "n", "rate", "lo", "hi", "RR", "RR_lo", "RR_hi"],
            "rows": rr_table_rows,
        }
    ]

    rr_values = [c["rr"] for c in per_cat if not c["baseline"] and math.isfinite(c["rr"])]
    if rr_values:
        max_rr = max(rr_values)
        max_rr_cat = next(c["category"] for c in per_cat if not c["baseline"] and c["rr"] == max_rr)
        rr_text = (
            f"The largest disparity is for group {max_rr_cat!r} with a "
            f"risk ratio of {max_rr:.3f} relative to the baseline "
            f"({baseline!r}, rate={baseline_rate:.4f})."
        )
    else:
        rr_text = "No non-baseline risk ratios could be computed."

    if bootstrap_reps > 0:
        boot_text = (
            f"Risk-ratio confidence intervals were obtained from "
            f"{int(bootstrap_reps)} non-parametric percentile bootstrap "
            f"replications (seed=0)."
        )
    else:
        boot_text = "Bootstrap was not requested (bootstrap_reps=0); only Wilson intervals for raw rates are reported."

    interpretation = f"{rr_text} {boot_text}"

    return RichResult(
        title="MRM-UOF Demographic Disparity",
        call=call,
        summary_lines=summary_lines,
        tables=tables,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": int(len(sub)),
            "baseline": str(baseline),
            "baseline_rate": baseline_rate,
            "per_category": per_cat,
            "risk_ratios": {c["category"]: c["rr"] for c in per_cat},
            "bootstrap_reps": int(bootstrap_reps),
            "value": float(max(rr_values)) if rr_values else float("nan"),
        },
    )


# ─── 6. data quality audit ───────────────────────────────────────────


def _schema_columns(schema: Any) -> list[tuple[str, str | None]] | None:
    """Duck-typed extraction of (name, dtype) from a schema object."""
    cols = getattr(schema, "columns", None)
    if cols is None:
        return None
    try:
        out: list[tuple[str, str | None]] = []
        for c in cols:
            name = getattr(c, "name", None)
            if name is None:
                return None
            dtype = getattr(c, "dtype", None)
            out.append((str(name), None if dtype is None else str(dtype)))
        return out
    except TypeError:
        return None


def mrm_uof_data_quality_audit(
    df: pd.DataFrame,
    sidecar: Mapping[str, Any] | None = None,
    expected_schema: Any = None,
) -> RichResult:
    """Schema + null + suspect-value audit of a UOF dataframe."""
    call = (
        f"mrm_uof_data_quality_audit(df=<{len(df)} rows x {len(df.columns)} cols>, "
        f"sidecar={'<dict>' if sidecar else 'None'}, "
        f"expected_schema={'<obj>' if expected_schema is not None else 'None'})"
    )
    warnings: list[str] = []

    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    per_column: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        pct_null = (n_null / n_rows) if n_rows > 0 else float("nan")
        pct_unique = (n_unique / n_rows) if n_rows > 0 else float("nan")
        dtype = str(s.dtype)
        entry: dict[str, Any] = {
            "column": str(col),
            "dtype": dtype,
            "n_null": n_null,
            "pct_null": pct_null,
            "n_unique": n_unique,
            "pct_unique": pct_unique,
        }
        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if len(non_null) > 0:
                entry["min"] = float(non_null.min())
                entry["max"] = float(non_null.max())
            else:
                entry["min"] = float("nan")
                entry["max"] = float("nan")
        else:
            try:
                mode = s.mode(dropna=True)
                entry["mode"] = None if mode.empty else mode.iloc[0]
            except Exception:
                entry["mode"] = None
        per_column.append(entry)

    expected_cols: list[tuple[str, str | None]] | None = None
    if sidecar is not None:
        fields = sidecar.get("fields") if isinstance(sidecar, Mapping) else None
        if isinstance(fields, list):
            try:
                expected_cols = [(str(f["id"]), str(f.get("type")) if f.get("type") else None) for f in fields]
            except (KeyError, TypeError):
                warnings.append(
                    "Sidecar 'fields' present but did not duck-type as CKAN; expected schema comparison skipped."
                )
        else:
            warnings.append("Sidecar is not a CKAN-shaped mapping with a 'fields' list; ignored for schema comparison.")
    if expected_cols is None and expected_schema is not None:
        expected_cols = _schema_columns(expected_schema)
        if expected_cols is None:
            warnings.append(
                "expected_schema did not duck-type (needs .columns of objects with .name); schema comparison skipped."
            )

    missing_columns: list[str] = []
    extra_columns: list[str] = []
    dtype_mismatches: list[dict[str, str]] = []

    if expected_cols is not None:
        actual = {str(c): str(df[c].dtype) for c in df.columns}
        expected_names = [c[0] for c in expected_cols]
        missing_columns = [c for c in expected_names if c not in actual]
        extra_columns = [c for c in actual if c not in expected_names]
        for name, exp_dt in expected_cols:
            if name in actual and exp_dt is not None:
                if exp_dt.lower() not in actual[name].lower() and actual[name].lower() not in exp_dt.lower():
                    dtype_mismatches.append({"column": name, "expected": exp_dt, "actual": actual[name]})

    suspect_flags: list[str] = []
    for entry in per_column:
        if entry["pct_null"] > 0.50:
            suspect_flags.append(f"{entry['column']}: {_fmt_pct(entry['pct_null'])} null")
        if pd.api.types.is_numeric_dtype(df[entry["column"]]):
            if entry.get("min") == entry.get("max") and n_rows > 1:
                if math.isfinite(entry.get("min", float("nan"))):
                    suspect_flags.append(f"{entry['column']}: constant value ({entry['min']})")
        else:
            if entry["n_unique"] == n_rows and n_rows > 1:
                suspect_flags.append(f"{entry['column']}: every value unique -- possible identifier")

    summary_lines: list[tuple[str, Any]] = [
        ("Rows", n_rows),
        ("Columns", n_cols),
        ("Missing columns", len(missing_columns)),
        ("Extra columns", len(extra_columns)),
        ("Dtype mismatches", len(dtype_mismatches)),
        ("Suspect flags", len(suspect_flags)),
    ]

    top_null_sorted = sorted(per_column, key=lambda e: e["pct_null"], reverse=True)[:3]
    top_null_rows: list[list[Any]] = [[e["column"], e["dtype"], e["n_null"], e["pct_null"]] for e in top_null_sorted]

    sections: list[dict[str, Any]] = [
        {
            "title": "Top-3 columns by null share",
            "headers": ["column", "dtype", "n_null", "pct_null"],
            "table": top_null_rows,
        }
    ]

    if missing_columns:
        sections.append(
            {
                "title": "Missing columns (in expected schema, absent in df)",
                "headers": ["column"],
                "table": [[c] for c in missing_columns],
            }
        )
    if extra_columns:
        sections.append(
            {
                "title": "Extra columns (in df, absent in expected schema)",
                "headers": ["column"],
                "table": [[c] for c in extra_columns],
            }
        )
    if dtype_mismatches:
        sections.append(
            {
                "title": "Dtype mismatches",
                "headers": ["column", "expected", "actual"],
                "table": [[m["column"], m["expected"], m["actual"]] for m in dtype_mismatches],
            }
        )
    if suspect_flags:
        sections.append(
            {
                "title": "Suspect flags",
                "headers": ["flag"],
                "table": [[f] for f in suspect_flags],
            }
        )

    flag_paragraphs: list[str] = []
    if missing_columns:
        flag_paragraphs.append(
            f"{len(missing_columns)} expected column(s) missing -- "
            f"these are declared by the schema but absent in the actual "
            f"dataframe."
        )
    if extra_columns:
        flag_paragraphs.append(
            f"{len(extra_columns)} extra column(s) present that the "
            f"schema does not declare; review whether to add to the "
            f"schema or drop from the frame."
        )
    if dtype_mismatches:
        flag_paragraphs.append(
            f"{len(dtype_mismatches)} dtype mismatch(es) detected -- "
            f"loose string comparison flagged a discrepancy between "
            f"the declared and observed types."
        )
    if suspect_flags:
        flag_paragraphs.append(
            f"{len(suspect_flags)} suspect-value flag(s) raised: high "
            f"null share, constant numeric values, or full-uniqueness "
            f"in non-numeric columns (typical of identifiers)."
        )
    if not flag_paragraphs:
        flag_paragraphs.append(
            "No structural or content flags were raised: schema (if any) "
            "matches and no column tripped a high-null, constant, or "
            "all-unique heuristic."
        )

    interpretation = " ".join(flag_paragraphs)

    return RichResult(
        title="MRM-UOF Data Quality Audit",
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "n": n_rows,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "per_column": per_column,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "dtype_mismatches": dtype_mismatches,
            "suspect_flags": suspect_flags,
            "value": float(len(suspect_flags)),
        },
    )


# ─── demo ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(0)
    df = pd.DataFrame(
        {
            "force": np.random.choice(["TPS", "OPP", "HRPS", "PRPS", "YRP"], 1000),
            "weapon": np.random.choice(["Firearm", "Taser", "Baton", "OC", "None"], 1000),
            "region_now": np.random.choice(["Central", "Eastern", "Northern", "Toronto", "Western"], 1000),
            "region_at": np.random.choice(["Central", "Eastern", "Northern", "Toronto", "Western"], 1000),
            "demo": np.random.choice(["A", "B", "C"], 1000),
            "outcome": np.random.binomial(1, 0.3, 1000),
        }
    )
    print(mrm_uof_force_concentration(df, "force"))
    print(mrm_uof_weapon_diversity(df, "weapon", "force"))
    print(mrm_uof_region_locality(df, "region_at", "region_now"))
    print(mrm_uof_demographic_disparity(df, "demo", "outcome"))
    print(mrm_uof_data_quality_audit(df))
    print(
        mrm_uof_yoy_change(
            {
                2020: df,
                2021: df.sample(900),
                2022: df.sample(1100),
                2023: df.sample(950),
                2024: df.sample(1050),
            }
        )
    )
