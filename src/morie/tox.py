# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forensic toxicology: matrix-aware quantitation and antemortem inference.

A thin domain layer for postmortem forensic toxicology, aimed at the hard case
where peripheral blood is compromised (hemodilution after water submersion, or
decompositional change) and the analyst must reason from alternative matrices
(liver, vitreous humour, brain, gastric contents). Like :mod:`morie.taphonomy`,
it adds no new statistics: it supplies typed, zero-row schemas and dispatches
quantitation to least squares and the antemortem-vs-artefact question to MORIE's
existing likelihood-ratio machinery (:func:`taphonomy_likelihood_ratio`).

**This module ships no data.** Calibrator responses, case measurements, and
matrix concentrations must be supplied by the caller from real analytical runs.
:func:`tox_matrix_schema` returns a typed zero-row frame; it never fabricates
rows.

**What it can and cannot do.** It fits an internal-standard calibration curve
and inverse-predicts an unknown concentration with LOD/LOQ, flags postmortem
redistribution from a central:peripheral ratio, and quantifies support for
antemortem ingestion versus postmortem microbial production as a likelihood
ratio. It reports bounded statistical support, never proof or a posterior
probability, and it does not establish cause of death.

R parity: ``rmorie`` ``R/tox.R`` (``morie_tox_*``).

References
----------
Dinis-Oliveira RJ, et al. (2010). Collection of biological samples in forensic
toxicology. *Toxicology Mechanisms and Methods* 20(7), 363-414.
Pounder DJ, Jones GR (1990). Post-mortem drug redistribution. *Forensic Science
International* 45(3), 253-263.
Kugelberg FC, Jones AW (2007). Interpreting results of ethanol analysis in
postmortem specimens. *Forensic Science International* 165(1), 10-29.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from morie.fn import _frame_core as pd

from .taphonomy import taphonomy_preservation_lr

# Recognised specimen matrices, ordered by resistance to dilution/putrefaction.
_TOX_MATRICES = (
    "peripheral_blood", "central_blood", "liver", "vitreous_humour", "brain",
    "gastric", "urine", "bile", "muscle", "kidney",
)

# Column dtypes and roles for the sample schema.
_SCHEMA: dict[str, str] = {
    "case_id": "object", "analyte": "object", "matrix": "object",
    "conc": "float64", "conc_units": "object", "lod": "float64",
    "loq": "float64", "decomp_stage": "Int64", "submersion_days": "float64",
    "censored": "Int64",
}
_SCHEMA_ROLES = {
    "case_id": "identifier", "analyte": "identifier", "matrix": "matrix",
    "conc": "measurement", "conc_units": "measurement", "lod": "quality",
    "loq": "quality", "decomp_stage": "matrix", "submersion_days": "matrix",
    "censored": "quality",
}

# Documented resistance-to-degradation ordering (Dinis-Oliveira et al. 2010), a
# transparent heuristic, NOT an empirical prior. `dilutes` = susceptible to
# submersion hemodilution.
_MATRIX_RELIABILITY: dict[str, dict[str, Any]] = {
    "vitreous_humour": {"base": 0.95, "dilutes": False},
    "brain": {"base": 0.85, "dilutes": False},
    "liver": {"base": 0.80, "dilutes": False},
    "bile": {"base": 0.70, "dilutes": False},
    "muscle": {"base": 0.65, "dilutes": False},
    "kidney": {"base": 0.65, "dilutes": False},
    "gastric": {"base": 0.55, "dilutes": False},
    "urine": {"base": 0.60, "dilutes": True},
    "peripheral_blood": {"base": 0.55, "dilutes": True},
    "central_blood": {"base": 0.35, "dilutes": True},
}


def tox_matrix_schema() -> pd.DataFrame:
    """Return an empty, typed DataFrame for a forensic-toxicology sample table.

    No rows are fabricated. Each column's ``role`` (``identifier`` / ``matrix``
    / ``measurement`` / ``quality``) is stored in ``df.attrs['role']``.
    R parity: ``morie_tox_matrix_schema``.
    """
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in _SCHEMA.items()})
    df.attrs["role"] = dict(_SCHEMA_ROLES)
    return df


def tox_calibration(
    conc,
    response,
    weights: str | np.ndarray = "1/x^2",
    response_unknown: float | None = None,
) -> dict[str, Any]:
    """Fit an internal-standard calibration curve and inverse-predict an unknown.

    Weighted linear least squares of instrument response on known calibrator
    concentration, the basis for GC-MS / LC-MS/MS quantitation. LOD/LOQ follow
    the ICH Q2 residual-standard-error convention (LOD = 3.3 s/b, LOQ = 10 s/b).
    With ``response_unknown``, inverse-predicts and flags against LOD/LOQ.
    R parity: ``morie_tox_calibration``.
    """
    conc = np.asarray(conc, dtype=float)
    response = np.asarray(response, dtype=float)
    if conc.shape != response.shape:
        raise ValueError("`conc` and `response` must be the same length")
    if conc.size < 3:
        raise ValueError("need at least 3 calibrators to fit a curve")
    if not (np.isfinite(conc).all() and np.isfinite(response).all()):
        raise ValueError("`conc`/`response` have non-finite values")
    if (conc <= 0).any():
        raise ValueError("calibrator `conc` must be > 0")

    if isinstance(weights, str):
        if weights == "1/x^2":
            w = 1.0 / conc**2
        elif weights == "1/x":
            w = 1.0 / conc
        elif weights == "none":
            w = np.ones_like(conc)
        else:
            raise ValueError("`weights` must be '1/x^2', '1/x', 'none', or array")
        weights_label = weights
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != conc.shape:
            raise ValueError("array `weights` must match `conc` length")
        weights_label = "array"

    # Weighted least squares for y = a + b x.
    sw = w.sum()
    xm = (w * conc).sum() / sw
    ym = (w * response).sum() / sw
    sxx = (w * (conc - xm) ** 2).sum()
    sxy = (w * (conc - xm) * (response - ym)).sum()
    if sxx == 0:
        raise ValueError("degenerate calibration (zero/undefined slope)")
    slope = sxy / sxx
    intercept = ym - slope * xm
    if not np.isfinite(slope) or slope == 0:
        raise ValueError("degenerate calibration (zero/undefined slope)")

    pred = intercept + slope * conc
    resid = response - pred
    dof = conc.size - 2
    s_resid = float(np.sqrt((w * resid**2).sum() / dof)) if dof > 0 else float("nan")
    # Weighted R^2.
    ss_tot = (w * (response - ym) ** 2).sum()
    r_squared = float(1 - (w * resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    lod = 3.3 * s_resid / abs(slope)
    loq = 10.0 * s_resid / abs(slope)

    out: dict[str, Any] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "lod": float(lod),
        "loq": float(loq),
        "weights": weights_label,
    }
    if response_unknown is not None:
        ru = float(response_unknown)
        if not np.isfinite(ru):
            raise ValueError("`response_unknown` must be a finite scalar")
        conc_hat = (ru - intercept) / slope
        out["conc_hat"] = float(conc_hat)
        out["flag"] = (
            "below_lod" if conc_hat < lod
            else "below_loq" if conc_hat < loq
            else "quantifiable"
        )
    return out


def tox_pmr_ratio(central: float, peripheral: float) -> dict[str, Any]:
    """Flag postmortem redistribution from a central:peripheral (C/P) ratio.

    PMR inflates central-blood drug concentrations relative to peripheral after
    death. C/P near 1 is stable; larger ratios indicate redistribution and
    unreliable central quantitation. R parity: ``morie_tox_pmr_ratio``.
    """
    central = float(central)
    peripheral = float(peripheral)
    if not (np.isfinite(central) and np.isfinite(peripheral)):
        raise ValueError("`central`/`peripheral` must be finite")
    if peripheral <= 0 or central <= 0:
        raise ValueError("concentrations must be > 0")
    cp = central / peripheral
    flag = "minimal" if cp <= 1 else "modest" if cp <= 2 else "significant"
    note = {
        "minimal": "Central and peripheral agree; central quantitation is a "
                   "reasonable antemortem proxy.",
        "modest": "Some redistribution; prefer the peripheral (femoral) value.",
        "significant": "Marked redistribution; the central value likely "
                       "overstates the antemortem concentration -- interpret "
                       "from peripheral blood or an alternative matrix.",
    }[flag]
    return {
        "cp_ratio": cp,
        "redistribution": flag,
        "interpretation": f"C/P ratio = {cp:.2f} ({flag} redistribution). {note}",
    }


def tox_antemortem_lr(
    marker, antemortem: dict[str, Any], postmortem: dict[str, Any]
) -> dict[str, Any]:
    """Likelihood ratio for antemortem ingestion versus postmortem artefact.

    The caller supplies an observed marker (typically a secondary metabolite a
    living liver produces, e.g. ethyl glucuronide, or a congener/matrix ratio)
    and Gaussian models ``{"mean":.., "sd":..}`` for the two hypotheses. Thin
    wrapper over :func:`taphonomy_preservation_lr`.
    R parity: ``morie_tox_antemortem_lr``.
    """
    out = taphonomy_preservation_lr(marker, antemortem, postmortem)
    lr = out["lr"]
    if np.isfinite(lr) and lr >= 1:
        factor = f"{lr:.4g} times"
    elif np.isfinite(lr):
        factor = f"{1 / lr:.4g} times less"
    else:
        factor = "infinitely"
    out["interpretation"] = (
        f"LR = {lr:.4g} (log10 = {out['log10_lr']:.3f}): the evidence is "
        f"{out['verbal']} for antemortem presence. The observed marker is "
        f"{factor} more probable under H1 (antemortem ingestion) than under H2 "
        "(postmortem artefact). This is bounded support, not proof and not a "
        "posterior probability."
    )
    return out


def tox_matrix_reliability(
    matrix: list[str] | None = None,
    submersion_days: float = 0.0,
    decomp_stage: float = 0.0,
) -> pd.DataFrame:
    """Rank specimen matrices by reliability under submersion and decomposition.

    Encodes the documented resistance-to-degradation ordering (vitreous humour
    and brain are protected; the liver concentrates; blood-based matrices
    dilute) as a transparent heuristic, penalised by days submerged and
    decomposition stage. A documented ordering, not an empirical prior.
    Returns a DataFrame of ``matrix`` / ``reliability`` (0-1) / ``rank``.
    R parity: ``morie_tox_matrix_reliability``.
    """
    submersion_days = float(submersion_days)
    decomp_stage = float(decomp_stage)
    if not np.isfinite(submersion_days) or submersion_days < 0:
        raise ValueError("`submersion_days` must be a finite scalar >= 0")
    if not np.isfinite(decomp_stage) or decomp_stage < 0:
        raise ValueError("`decomp_stage` must be a finite scalar >= 0")
    mats = list(matrix) if matrix is not None else list(_MATRIX_RELIABILITY)
    unknown = [m for m in mats if m not in _MATRIX_RELIABILITY]
    if unknown:
        raise ValueError("unknown matrix: " + ", ".join(unknown))
    dil_pen = 1 - min(submersion_days / 30, 0.8)
    dec_pen = 1 - min(decomp_stage / 10, 0.6)
    rows = []
    for m in mats:
        spec = _MATRIX_RELIABILITY[m]
        r = spec["base"] * dec_pen
        if spec["dilutes"]:
            r *= dil_pen
        rows.append((m, round(r, 3)))
    rows.sort(key=lambda t: t[1], reverse=True)
    return pd.DataFrame(
        {"matrix": [r[0] for r in rows],
         "reliability": [r[1] for r in rows],
         "rank": list(range(1, len(rows) + 1))}
    )


def tox_left_censor_impute(values, lod: float, method: str = "half") -> dict[str, Any]:
    """Impute left-censored (below-LOD) toxicology values.

    Applies a documented simple-substitution rule to censored entries (``NaN``
    or below ``lod``). For regression on censored data prefer
    ``morie.horowitz_censored_regression``. R parity:
    ``morie_tox_left_censor_impute``.
    """
    values = np.asarray(values, dtype=float)
    lod = float(lod)
    if not np.isfinite(lod) or lod <= 0:
        raise ValueError("`lod` must be a finite scalar > 0")
    if method == "half":
        sub = lod / 2
    elif method == "sqrt2":
        sub = lod / np.sqrt(2)
    elif method == "lod":
        sub = lod
    else:
        raise ValueError("`method` must be 'half', 'sqrt2', or 'lod'")
    censored = np.isnan(values) | (values < lod)
    imputed = values.copy()
    imputed[censored] = sub
    return {
        "imputed": imputed,
        "censored": censored,
        "fraction_censored": float(censored.mean()),
    }


def tox_ethanol_congeners(
    ethanol: float,
    n_propanol: float = 0.0,
    n_butanol: float = 0.0,
    etg: float = float("nan"),
    congener_threshold: float = 0.0,
) -> dict[str, Any]:
    """Adjudicate ethanol as antemortem ingestion versus postmortem production.

    Higher alcohols (n-propanol, n-butanol) are fermentation by-products, so
    their presence points to postmortem microbial production; ethyl glucuronide
    (EtG) is produced only by living metabolism, so its presence points to
    antemortem intake (Kugelberg & Jones 2007). For a quantified LR, feed EtG to
    :func:`tox_antemortem_lr`. R parity: ``morie_tox_ethanol_congeners``.
    """
    ethanol = float(ethanol)
    if not np.isfinite(ethanol) or ethanol < 0:
        raise ValueError("`ethanol` must be a finite scalar >= 0")
    higher = max(float(n_propanol), float(n_butanol))
    ferment = np.isfinite(higher) and higher > congener_threshold
    antemortem = np.isfinite(float(etg)) and float(etg) > 0

    if antemortem:
        verdict = "antemortem"
        interpretation = (
            f"EtG present ({float(etg):.3g}); ethyl glucuronide requires living "
            "metabolism, supporting antemortem ingestion despite decomposition."
        )
    elif ferment:
        verdict = "postmortem_production"
        interpretation = (
            f"Higher alcohols present ({higher:.3g} > {congener_threshold:.3g}); "
            "n-propanol/n-butanol are fermentation by-products, so the ethanol "
            f"({ethanol:.3g}) is consistent with postmortem microbial production."
        )
    else:
        verdict = "indeterminate"
        interpretation = (
            f"No fermentation congeners and no EtG measured; ethanol "
            f"({ethanol:.3g}) cannot be adjudicated antemortem vs postmortem "
            "from these signals alone -- add EtG/EtS or a vitreous:blood "
            "comparison."
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "higher_alcohol": higher,
        "etg": float(etg),
    }
