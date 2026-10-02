# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P5: what can be certified about the fairness of a risk score
(``research/lean/P5Fairness.lean``, ``P5Compare.lean``, ``P5Rescale.lean``, ``P5Ranking.lean``, ``P5Hazard.lean``).

* ``Research.P5.Table.chouldechova`` / ``impossibility``
* ``Research.P5.true_base_rate_bounds`` / ``lower_bound_attained`` / ``upper_bound_attained`` / ``trueRate_observedRate``
* ``Research.P5.compare_decided`` / ``compare_undecided``
* ``Research.P5.reduced_coefficient`` / ``rescale_lt_one`` / ``rescale_eq_one_iff`` / ``ratio_is_rescaling``
* ``Research.P5.rank_reversal_exists`` / ``rank_stable_of_gap`` / ``identified_scores_le``
* ``Research.P5.survivor_hazard_mono`` / ``hr2_gt_one_of_depletion`` / ``hr2_witness``

R parity: ``rmorie`` ``R/fairness_bounds.R``.
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = [
    "fairness_rates",
    "fairness_implied_fpr",
    "fairness_base_rate_bounds",
    "fairness_true_rate",
    "fairness_compare_groups",
    "logit_rescale",
    "ranking_resolution",
    "hazard_selection",
]


def _pos(v, name):
    v = float(v)
    if v != v or v <= 0:
        raise ValueError(f"{name} must be single positive numbers")
    return v


def fairness_rates(tp, fp, fn, tn) -> dict:
    """Group-wise rates from a confusion table.

    Examples
    --------
    >>> r = fairness_rates(tp=120, fp=60, fn=40, tn=280)
    >>> (round(r["p"], 12), round(r["ppv"], 12), round(r["fpr"], 12), r["fnr"])
    (0.32, 0.666666666667, 0.176470588235, 0.25)
    """
    tp = _pos(tp, "tp, fp, fn and tn")
    fp = _pos(fp, "tp, fp, fn and tn")
    fn = _pos(fn, "tp, fp, fn and tn")
    tn = _pos(tn, "tp, fp, fn and tn")
    n = tp + fp + fn + tn
    return {"p": (tp + fn) / n, "ppv": tp / (tp + fp), "fpr": fp / (fp + tn), "fnr": fn / (tp + fn), "n": n}


def fairness_implied_fpr(p, ppv, fnr) -> float:
    """Chouldechova's identity ``fpr = p/(1-p) (1-ppv)/ppv (1-fnr)`` (``Research.P5.Table.chouldechova``).

    Examples
    --------
    >>> r = fairness_rates(120, 60, 40, 280)
    >>> round(fairness_implied_fpr(r["p"], r["ppv"], r["fnr"]) - r["fpr"], 15)
    0.0
    """
    p = float(p)
    ppv = float(ppv)
    fnr = float(fnr)
    if p != p or p <= 0 or p >= 1:
        raise ValueError("p must be a single number in (0, 1)")
    if ppv != ppv or ppv <= 0 or ppv > 1:
        raise ValueError("ppv must be a single number in (0, 1]")
    if fnr != fnr or fnr < 0 or fnr >= 1:
        raise ValueError("fnr must be a single number in [0, 1)")
    return p / (1 - p) * ((1 - ppv) / ppv) * (1 - fnr)


def fairness_base_rate_bounds(p_obs, alpha_max, beta_max):
    """Sharp bounds on a true base rate from a noisy recorded one (``Research.P5.true_base_rate_bounds``).

    Examples
    --------
    >>> b = fairness_base_rate_bounds([0.35, 0.55], alpha_max=0.10, beta_max=0.20)
    >>> [round(v, 12) for v in b["upper"]]
    [0.4375, 0.6875]
    """
    p = np.atleast_1d(np.asarray(p_obs, dtype=float))
    if np.any(np.isnan(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p_obs must be numeric in [0, 1]")
    a = float(alpha_max)
    b = float(beta_max)
    if a != a or b != b or a < 0 or b < 0:
        raise ValueError("alpha_max and beta_max must be single non-negative numbers")
    if a + b >= 1:
        raise ValueError("alpha_max + beta_max must be below 1")
    lower = np.maximum(0.0, (p - a) / (1 - a))
    upper = np.minimum(1.0, p / (1 - b))
    return pd.DataFrame({"p_obs": p, "lower": lower, "upper": upper, "width": upper - lower})


def fairness_true_rate(p_obs, alpha, beta):
    """Recover a true base rate when the noise rates are known (``Research.P5.trueRate_observedRate``).

    Examples
    --------
    >>> round(float(fairness_true_rate(0.4 * 0.8 + 0.6 * 0.1, alpha=0.1, beta=0.2)[0]), 12)
    0.4
    """
    p = np.atleast_1d(np.asarray(p_obs, dtype=float))
    alpha = float(alpha)
    beta = float(beta)
    if alpha != alpha or beta != beta or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("alpha and beta must be non-negative with alpha + beta < 1")
    return (p - alpha) / (1 - alpha - beta)


def fairness_compare_groups(p_obs_a, p_obs_b, alpha_max, beta_max) -> dict:
    """Can two groups' true base rates be ordered under label noise? (``compare_decided``, ``compare_undecided``)

    Examples
    --------
    >>> fairness_compare_groups(0.35, 0.55, alpha_max=0.05, beta_max=0.20)["order"]
    'a < b'
    >>> fairness_compare_groups(0.35, 0.55, alpha_max=0.05, beta_max=0.40)["order"]
    'undecided'
    """
    ba = fairness_base_rate_bounds(p_obs_a, alpha_max, beta_max)
    bb = fairness_base_rate_bounds(p_obs_b, alpha_max, beta_max)
    la, ua = float(ba["lower"][0]), float(ba["upper"][0])
    lb, ub = float(bb["lower"][0]), float(bb["upper"][0])
    order = "a < b" if ua < lb else ("b < a" if ub < la else "undecided")
    pa = float(p_obs_a)
    pb = float(p_obs_b)
    am = float(alpha_max)
    breakdown = float("nan")
    lo_b = (pb - am) / (1 - am)
    lo_a = (pa - am) / (1 - am)
    if pa < pb and lo_b > 0:
        b = 1 - pa / lo_b
        if b > 0:
            breakdown = min(b, 1 - am - 1e-12)
    elif pb < pa and lo_a > 0:
        b = 1 - pb / lo_a
        if b > 0:
            breakdown = min(b, 1 - am - 1e-12)
    return {
        "decided": order != "undecided",
        "order": order,
        "interval_a": {"lower": la, "upper": ua},
        "interval_b": {"lower": lb, "upper": ub},
        "breakdown_beta_max": breakdown,
        "theorem": "Research.P5.compare_decided" if order != "undecided" else "Research.P5.compare_undecided",
    }


def logit_rescale(beta, omitted_var, error_var=math.pi**2 / 3) -> dict:
    """Rescaling of logit coefficients across nested models (``Research.P5.reduced_coefficient`` and kin).

    Examples
    --------
    >>> r = logit_rescale(beta=0.8, omitted_var=1)
    >>> round(r["rescale"], 12)
    0.875724044222
    """
    error_var = float(error_var)
    scalar = isinstance(beta, int | float) and isinstance(omitted_var, int | float)
    beta = np.asarray(beta, dtype=float)
    ov = np.asarray(omitted_var, dtype=float)
    if np.any(ov < 0) or error_var <= 0:
        raise ValueError("omitted_var must be non-negative and error_var positive")
    c = np.sqrt(error_var / (error_var + ov))
    fin = (lambda a: float(a)) if scalar else (lambda a: a)
    return {
        "rescale": fin(c),
        "beta_reduced": fin(beta * c),
        "odds_ratio_full": fin(np.exp(beta)),
        "odds_ratio_reduced": fin(np.exp(beta * c)),
        "apparent_change": fin(np.exp((c - 1) * beta)),
        "theorems": [
            "Research.P5.rescale_lt_one",
            "Research.P5.rescale_eq_one_iff",
            "Research.P5.reduced_coefficient",
            "Research.P5.ratio_is_rescaling",
        ],
    }


def ranking_resolution(estimate, half_width) -> dict:
    """Resolution of a ranking built from noisy risk scores (``rank_reversal_exists``, ``rank_stable_of_gap``).

    Examples
    --------
    >>> r = ranking_resolution([0.2, 0.35, 0.8], half_width=[0.1, 0.1, 0.05])
    >>> (r["n_pairs"], round(r["share_unidentified"], 12), r["resolution"])
    (3, 0.333333333333, 0.2)
    """
    e = np.asarray(estimate, dtype=float)
    n = e.shape[0]
    if n < 2:
        raise ValueError("need at least two scores")
    hw = np.atleast_1d(np.asarray(half_width, dtype=float))
    if hw.shape[0] == 1:
        hw = np.full(n, float(hw[0]))
    if hw.shape[0] != n:
        hw = np.array([float(hw[i % hw.shape[0]]) for i in range(n)])
    if np.any(hw < 0):
        raise ValueError("half_width must be non-negative")
    gap = np.abs(np.outer(e, np.ones(n)) - np.outer(np.ones(n), e))
    tol = np.outer(hw, np.ones(n)) + np.outer(np.ones(n), hw)
    ident = gap > tol
    pairs = [bool(ident[i, j]) for i in range(n) for j in range(i + 1, n)]
    return {
        "n_pairs": len(pairs),
        "identified_pairs": ident,
        "share_unidentified": sum(1 for p in pairs if not p) / len(pairs),
        "resolution": 2 * float(np.max(hw)),
        "theorems": [
            "Research.P5.rank_reversal_exists",
            "Research.P5.rank_stable_of_gap",
            "Research.P5.identified_scores_le",
        ],
    }


def hazard_selection(s, h, l, survive_high, survive_low) -> dict:  # noqa: E741
    """Built-in selection in period-by-period hazard ratios (``survivor_hazard_mono``, ``hr2_gt_one_of_depletion``).

    Examples
    --------
    >>> r = hazard_selection(s=0.5, h=0.5, l=0.1, survive_high=(0.5, 0.8), survive_low=(0.9, 0.9))
    >>> round(r["period2_hazard_ratio"], 12)
    1.186851211073
    """
    s = float(s)
    h = float(h)
    l = float(l)  # noqa: E741
    if s <= 0 or s >= 1:
        raise ValueError("s must lie in (0, 1)")
    if not (l < h) or l < 0 or h > 1:
        raise ValueError("need 0 <= l < h <= 1")
    sh = [float(v) for v in survive_high]
    sl = [float(v) for v in survive_low]
    if len(sh) != 2 or len(sl) != 2 or any(v <= 0 for v in sh + sl):
        raise ValueError("survival probabilities must be length-2 positive vectors")
    w = [s * sh[i] / (s * sh[i] + (1 - s) * sl[i]) for i in range(2)]
    hz = [w[i] * h + (1 - w[i]) * l for i in range(2)]
    return {
        "surviving_high_share": {"control": w[0], "treated": w[1]},
        "period2_hazard": {"control": hz[0], "treated": hz[1]},
        "period2_hazard_ratio": hz[1] / hz[0],
        "theorems": [
            "Research.P5.survivor_hazard_mono",
            "Research.P5.hr2_gt_one_of_depletion",
            "Research.P5.hr2_witness",
        ],
    }
