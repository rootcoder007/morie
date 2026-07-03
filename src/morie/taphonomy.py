# SPDX-License-Identifier: AGPL-3.0-or-later
"""Taphonomic preservation as a causal-inference problem.

A thin domain layer that recasts "is this body's preservation natural or
anomalous?" as a treatment-effect estimate over MORIE's existing causal
estimators. It adds no new statistics: it documents the taphonomy variable set,
dispatches to :func:`morie.estimate_irm` / :func:`morie.estimate_cate`, and
attaches an E-value (:func:`morie.sensitivity.e_value_d`) that quantifies how
strong an *unmeasured* cause would have to be to explain away the natural
preservation effect -- i.e. it turns a claim of "incorruptibility" into a
bounded statistical one.

**This module ships no data.** Forensic-taphonomy comparanda (documented
lime/desiccation burials with preservation outcomes) and any non-invasive
readings of a specific case (CT/micro-CT density, pXRF elemental signatures,
hyperspectral surface composition) must be supplied by the caller from real
sources. :func:`taphonomy_schema` returns a typed, zero-row frame describing the
expected columns; it never fabricates rows.

**What it can and cannot do.** With comparanda it estimates the average
preservation effect attributable to burial *processing* (e.g. quicklime
desiccation) and reports the E-value residual. It cannot falsify a miracle: a
small natural-preservation probability yields a *delta*, not a disproof
(Chernozhukov et al. 2018 give the effect; VanderWeele & Ding 2017 bound the
unmeasured confounding needed to nullify it).

R parity: ``r-morie-oss`` ``R/taphonomy.R``
(``morie_taphonomy_schema`` / ``morie_taphonomy_preservation_delta``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from .causal import estimate_cate, estimate_irm
from .sensitivity import e_value_d

# Canonical taphonomy variable set, grouped by causal role. One source of truth
# for both the schema and the estimator.
_TREATMENT = {"lime_treatment": "Int64"}  # 1 = interred with quicklime, 0 = not
_COVARIATES = {
    "temp_c": "float64",             # mean interment temperature (deg C)
    "humidity_pct": "float64",       # mean relative humidity (%)
    "arid": "Int64",                 # 1 = cool/arid microclimate
    "casket_sealed": "Int64",        # 1 = sealed casket / low air exchange
    "reinterment_count": "Int64",    # times exhumed / moved
    "exposure_days": "float64",      # days of pre-burial exposure
    "decades_elapsed": "float64",    # time since death (decades)
}
_MEASUREMENTS = {
    "ct_density_hu": "float64",       # CT/micro-CT mean tissue density (HU)
    "ct_void_fraction": "float64",    # internal void fraction (0-1)
    "pxrf_ca_ppm": "float64",         # pXRF residual surface calcium (ppm)
    "hyperspectral_resin": "Int64",   # 1 = applied resin/wax detected
}
_OUTCOME = {"preservation_score": "float64"}  # 0 (decayed) .. 1 (intact)

_ROLES = (
    [("treatment", c) for c in _TREATMENT]
    + [("covariate", c) for c in _COVARIATES]
    + [("measurement", c) for c in _MEASUREMENTS]
    + [("outcome", c) for c in _OUTCOME]
)


def taphonomy_schema() -> pd.DataFrame:
    """Return an empty, typed DataFrame documenting the expected columns.

    No rows are fabricated -- the caller fills it with real comparanda. The
    ``role`` of each column (``treatment`` / ``covariate`` / ``measurement`` /
    ``outcome``) is stored in ``df.attrs['role']``.
    """
    dtypes = {**_TREATMENT, **_COVARIATES, **_MEASUREMENTS, **_OUTCOME}
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    df.attrs["role"] = {c: role for role, c in _ROLES}
    return df


def taphonomy_preservation_delta(
    data: pd.DataFrame,
    *,
    treatment: str = "lime_treatment",
    outcome: str = "preservation_score",
    covariates: list[str] | None = None,
    estimator: str = "irm",
    se_method: str = "none",
    n_boot: int = 199,
    boot_seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate the natural preservation "delta" and its E-value.

    Recasts a taphonomy question as a treatment-effect estimate: how much of the
    observed preservation is attributable to burial *processing* (the
    ``treatment``, e.g. quicklime desiccation), holding environment and handling
    fixed, and how strong an unmeasured cause (a "miracle") would need to be to
    explain the rest. A thin dispatch over :func:`morie.estimate_irm` (default)
    or :func:`morie.estimate_cate` plus :func:`morie.sensitivity.e_value_d`.
    Adds no new statistics.

    Parameters
    ----------
    data : pandas.DataFrame
        Real comparanda (see :func:`taphonomy_schema`). Must be non-empty.
    treatment : str
        Binary treatment column (default ``"lime_treatment"``).
    outcome : str
        Continuous preservation outcome (default ``"preservation_score"``).
    covariates : list[str], optional
        Confounder/measurement columns. Defaults to every schema covariate +
        measurement present in ``data``.
    estimator : str
        ``"irm"`` (DoubleML, default) or ``"cate"``.
    se_method : str
        Inference for the ``"cate"`` path only: ``"none"`` (default) reports the
        point estimate + ``cate_sd`` dispersion with no SE/CI/p-value;
        ``"bootstrap"`` resamples rows and refits the CATE procedure ``n_boot``
        times for a *valid* SE, percentile CI, and p-value. (``sd(tau)/sqrt(n)``
        is deliberately not offered -- the per-unit effects are correlated
        fitted predictions, so it understates the SE; use ``cate_sd`` for
        heterogeneity or bootstrap for inference.) Ignored for ``"irm"``.
    n_boot : int
        Bootstrap resamples when ``se_method="bootstrap"`` (default 199).
    boot_seed : int
        RNG seed for the bootstrap (default 42).
    **kwargs
        Passed to the chosen estimator.

    Returns
    -------
    dict
        All estimates are ``float`` (double); ``n`` is an ``int`` count.
        Keys: ``value`` (preservation delta = ATE / mean CATE), ``se``,
        ``p_value`` (Wald; ``None`` on the CATE path -- no valid SE),
        ``ci_lower``, ``ci_upper``, ``n``, ``e_value``, ``e_value_ci``,
        ``estimator``, ``cate_per_unit`` and ``cate_sd`` (``None`` unless
        ``estimator="cate"``), ``warnings``, ``interpretation``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    if len(data) == 0:
        raise ValueError(
            "`data` is empty. Fill taphonomy_schema() with real comparanda; "
            "this module never fabricates burial rows."
        )
    for col in (treatment, outcome):
        if col not in data.columns:
            raise KeyError(f"column '{col}' not found in `data`")

    if covariates is None:
        wanted = list(_COVARIATES) + list(_MEASUREMENTS)
        covariates = [c for c in wanted if c in data.columns]
    covariates = [c for c in covariates if c not in (treatment, outcome)]
    if not covariates:
        raise ValueError(
            "no covariates available; supply environment/handling/measurement "
            "columns (see taphonomy_schema())"
        )

    warnings: list[str] = []
    if data[treatment].dropna().nunique() < 2:
        warnings.append(
            "treatment has no contrast (all treated or all control) -- "
            "the effect is not identified."
        )

    # Normalise each estimator's output to a common summary. estimate_irm
    # returns a dict (cross-fit ATE + orthogonal SE); estimate_cate returns a
    # per-unit Series, summarised here to its mean (point estimate) + dispersion.
    cate_per_unit: pd.Series | None = None
    cate_sd: float | None = None
    if estimator == "irm":
        est = estimate_irm(
            data, treatment=treatment, outcome=outcome,
            covariates=covariates, **kwargs,
        )
        ate = float(est["ate"])
        se = est.get("se")
        se = float(se) if se is not None else None
        ci_lower, ci_upper = est.get("ci_lower"), est.get("ci_upper")
        n = est.get("n", len(data))
        method = est.get("method", "IRM (DoubleML)")
    elif estimator == "cate":
        tau = estimate_cate(
            data, treatment=treatment, outcome=outcome,
            covariates=covariates, **kwargs,
        ).dropna()
        if len(tau) == 0:
            raise ValueError("CATE estimation returned no finite effects")
        cate_per_unit = tau
        n = int(len(tau))
        ate = float(tau.mean())
        cate_sd = float(tau.std(ddof=1)) if n > 1 else None  # effect heterogeneity
        method = "CATE (meta-learner, mean of per-unit effects)"
        if se_method == "bootstrap":
            # Valid inference: resample rows and refit the whole CATE procedure,
            # so the SE reflects sampling + estimation uncertainty of the mean
            # effect (not the correlated per-unit dispersion).
            import numpy as np

            rng = np.random.default_rng(boot_seed)
            nrow = len(data)
            boot = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, nrow, size=nrow)
                tb = estimate_cate(
                    data.iloc[idx], treatment=treatment, outcome=outcome,
                    covariates=covariates, **kwargs,
                ).dropna()
                if len(tb):
                    boot.append(float(tb.mean()))
            boot_arr = np.asarray(boot, dtype=float)
            se = float(boot_arr.std(ddof=1))
            ci_lower, ci_upper = (float(x) for x in np.quantile(boot_arr, [0.025, 0.975]))
            method = f"CATE (meta-learner, mean; {len(boot)}-boot SE/CI)"
        elif se_method == "none":
            # Point summary only. sd(tau)/sqrt(n) is NOT reported as an SE: the
            # per-unit effects are correlated fitted predictions, so it would
            # understate the true uncertainty. Use se_method="bootstrap" for a
            # valid SE, or cate_sd for heterogeneity.
            se = None
            ci_lower = ci_upper = None
            _sd = cate_sd if cate_sd is not None else float("nan")
            warnings.append(
                "CATE point summary: no SE reported (per-unit effects are "
                "correlated fitted predictions; "
                f"cate_sd={_sd:.3f} is heterogeneity, range "
                f"[{float(tau.min()):.3f}, {float(tau.max()):.3f}]). Pass "
                "se_method='bootstrap' for a valid SE + CI + p-value, or use "
                "estimator='irm'."
            )
        else:
            raise ValueError(
                f"unknown se_method {se_method!r}; choose 'none' or 'bootstrap'"
            )
    else:
        raise ValueError(
            f"unknown estimator {estimator!r}; choose 'irm' or 'cate'"
        )

    # Wald p-value from a valid SE only (double). IRM's SE is a cross-fit
    # orthogonal SE; the CATE path leaves se = None, so p stays None.
    p_value: float | None = None
    if se is not None and se > 0:
        p_value = float(2.0 * norm.sf(abs(ate / se)))

    out_sd = float(data[outcome].std(skipna=True))
    e_point = e_ci = None
    if out_sd and out_sd > 0:
        d = ate / out_sd
        se_d = (se / out_sd) if (se is not None and se > 0) else None
        ev = e_value_d(d, se=se_d)
        e_point, e_ci = ev.e_value_point, ev.e_value_ci
    else:
        warnings.append("outcome has zero/undefined SD; E-value skipped.")

    lo = float("nan") if ci_lower is None else ci_lower
    hi = float("nan") if ci_upper is None else ci_upper
    interpretation = (
        f"Preservation delta (ATE of {treatment} on {outcome}) = {ate:.3f} "
        f"[{lo:.3f}, {hi:.3f}], n={n}, via {method}. "
    )
    if e_point is not None:
        interpretation += (
            f"An unmeasured cause would need an E-value of {e_point:.2f} "
            "(association with both treatment and outcome, on the risk-ratio "
            "scale) to fully explain it away. This bounds -- it does not falsify "
            "-- any 'incorruptibility' claim: a natural mechanism (e.g. quicklime "
            "desiccation) of this strength is sufficient, but sufficiency is not "
            "proof of exclusivity."
        )
    else:
        interpretation += "E-value unavailable."

    return {
        "value": ate,
        "se": se,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "e_value": e_point,
        "e_value_ci": e_ci,
        "estimator": method,
        "cate_per_unit": cate_per_unit,
        "cate_sd": cate_sd,
        "warnings": warnings,
        "interpretation": interpretation,
    }
