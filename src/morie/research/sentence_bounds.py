# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P11: sentencing effects as intervals
(``research/lean/P11Bounds.lean``, ``P11Contaminated.lean``, ``P11Monotone.lean``, ``P11Coverage.lean``;
Manski, Identification for Prediction and Decision; Manski & Nagin 1998; Imbens & Manski 2004).

* ``Research.P11.Pop.outcome_bounds`` / ``ate_width_one`` / ``ate_contains_zero``
* ``Research.P11.clean_bounds`` / ``clean_width`` / ``clean_informative``
* ``Research.P11.Pop.mtr_lower`` / ``mtr_upper`` / ``mtr_upper_attained``
* ``Research.P11.region_coverage_le`` / ``im_cutoff_antitone`` / ``im_cutoff_between`` / ``two_sided_overcovers``
* ``Research.P11.Pop.mts_mean_b_le`` / ``mts_mean_a_ge`` / ``mts_ate_le_naive`` / ``mtr_mts_bounds`` (``P11Selection.lean``)

R parity: ``rmorie`` ``R/sentence_bounds.R``, ``R/bounds_confidence.R``.
"""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._sci_core import brentq
from morie.fn._stats_core import norm

__all__ = ["sentence_effect_bounds", "contaminated_bounds", "sentence_effect_mtr", "sentence_effect_mts", "bounds_confidence"]


def sentence_effect_bounds(y, z, weights=None, contrast=None) -> dict:
    """Worst-case identification bounds for a binary outcome under two sentences.

    ``P(y=1, z=t) <= P[y(t)=1] <= P(y=1, z=t) + P(z != t)``, both ends attained
    (``Research.P11.Pop.outcome_bounds``); the contrast interval has width one
    and contains zero (``ate_width_one``, ``ate_contains_zero``).

    Examples
    --------
    >>> b = sentence_effect_bounds([1, 0, 1, 0, 1, 1], ["a", "a", "a", "b", "b", "b"])
    >>> (round(b["ate_width"], 12), round(b["ate_bounds"]["lower"], 12), round(b["ate_bounds"]["upper"], 12))
    (1.0, -0.5, 0.5)
    """
    y = np.asarray(y, dtype=float); z = [str(v) for v in z]
    n = y.shape[0]
    if len(z) != n:
        raise ValueError("y and z must have equal length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be 0/1")
    lv = sorted(set(z))
    if len(lv) != 2:
        raise ValueError("z must take exactly two distinct values")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    w = w / float(w.sum())
    if contrast is None:
        trt = lv[1]
    else:
        trt = str(contrast)
        if trt not in lv:
            raise ValueError("contrast must be one of the sentence values")
    ctl = lv[0] if lv[1] == trt else lv[1]
    zz = np.array([v == trt for v in z])
    joint = {ctl: float(np.sum(w[~zz] * y[~zz])), trt: float(np.sum(w[zz] * y[zz]))}
    pz = {ctl: float(np.sum(w[~zz])), trt: float(np.sum(w[zz]))}
    ob = {k: {"lower": joint[k], "upper": joint[k] + (1 - pz[k])} for k in (ctl, trt)}
    ate = {"lower": ob[trt]["lower"] - ob[ctl]["upper"], "upper": ob[trt]["upper"] - ob[ctl]["lower"]}
    return {
        "levels": {"comparison": ctl, "treatment": trt}, "joint": joint, "pz": pz,
        "outcome_bounds": ob, "ate_bounds": ate, "ate_width": ate["upper"] - ate["lower"],
        "naive_difference": joint[trt] / pz[trt] - joint[ctl] / pz[ctl],
        "theorems": ["Research.P11.Pop.outcome_bounds", "Research.P11.Pop.lower_attained",
                     "Research.P11.Pop.upper_attained", "Research.P11.Pop.ate_width_one",
                     "Research.P11.Pop.ate_contains_zero"],
    }


def contaminated_bounds(q, p):
    """Contaminated-sample bounds for a recorded proportion
    (``Research.P11.clean_bounds``, ``clean_width``, ``clean_informative``).

    Examples
    --------
    >>> b = contaminated_bounds([0.05, 0.5, 0.97], 0.1)
    >>> [round(v, 12) for v in b["lower"]]
    [0.0, 0.444444444444, 0.966666666667]
    """
    p = float(p)
    if p != p or p < 0 or p >= 1:
        raise ValueError("p must be a single number in [0, 1)")
    qq = np.atleast_1d(np.asarray(q, dtype=float))
    if np.any(np.isnan(qq)) or np.any((qq < 0) | (qq > 1)):
        raise ValueError("q must lie in [0, 1]")
    lower = np.maximum(0.0, (qq - p) / (1 - p))
    upper = np.minimum(1.0, qq / (1 - p))
    out = pd.DataFrame({"q": qq, "lower": lower, "upper": upper, "width": np.full(qq.shape[0], p / (1 - p)),
                        "informative": (p < qq) | (p < 1 - qq)})
    out.attrs["theorems"] = ["Research.P11.clean_bounds", "Research.P11.clean_lower_attained",
                             "Research.P11.clean_upper_attained", "Research.P11.clean_width",
                             "Research.P11.clean_informative"]
    return out


def sentence_effect_mtr(y, z, weights=None, contrast=None, direction="non-decreasing") -> dict:
    """Monotone-treatment-response bounds (``Research.P11.Pop.mtr_lower``, ``mtr_upper``, ``mtr_upper_attained``).

    Examples
    --------
    >>> r = sentence_effect_mtr([1, 0, 1, 0, 1, 1], ["a", "a", "a", "b", "b", "b"])
    >>> (r["bounds"]["lower"], round(r["bounds"]["upper"], 12))
    (0, 0.5)
    """
    if direction not in ("non-decreasing", "non-increasing"):
        raise ValueError("direction must be 'non-decreasing' or 'non-increasing'")
    b = sentence_effect_bounds(y, z, weights, contrast)
    trt = b["levels"]["treatment"]; ctl = b["levels"]["comparison"]
    up = b["joint"][trt] + (b["pz"][ctl] - b["joint"][ctl])
    if direction == "non-decreasing":
        bounds = {"lower": 0, "upper": up}
    else:
        bounds = {"lower": -((b["pz"][trt] - b["joint"][trt]) + b["joint"][ctl]), "upper": 0}
    return {"levels": b["levels"], "direction": direction, "bounds": bounds,
            "width": bounds["upper"] - bounds["lower"], "naive_difference": b["naive_difference"],
            "theorems": ["Research.P11.Pop.mtr_lower", "Research.P11.Pop.mtr_upper", "Research.P11.Pop.mtr_upper_attained"]}


def bounds_confidence(lower, upper, se_lower, se_upper, level=0.95) -> dict:
    """Imbens-Manski confidence interval for a partially identified parameter.

    The cutoff ``c`` solves ``Phi(c + Delta) - Phi(-c) = 1 - alpha`` with
    ``Delta = (upper - lower)/max(se)``; it is non-increasing in Delta
    (``Research.P11.im_cutoff_antitone``), lies between the one- and two-sided
    quantiles (``im_cutoff_between``), and the two-sided quantile over-covers
    any region of positive width (``two_sided_overcovers``).

    Examples
    --------
    >>> r = bounds_confidence(lower=0.10, upper=0.35, se_lower=0.03, se_upper=0.04)
    >>> round(r["cutoff"], 6)
    1.644854
    """
    vals = [float(v) for v in (lower, upper, se_lower, se_upper, level)]
    if any(v != v for v in vals):
        raise ValueError("all arguments must be single numbers")
    lower, upper, se_lower, se_upper, level = vals
    if upper < lower:
        raise ValueError("upper must be at least lower")
    if se_lower <= 0 or se_upper <= 0:
        raise ValueError("standard errors must be positive")
    if level <= 0 or level >= 1:
        raise ValueError("level must lie in (0, 1)")
    alpha = 1 - level
    sigma = max(se_lower, se_upper)
    delta = (upper - lower) / sigma
    z1 = float(norm.ppf(1 - alpha)); z2 = float(norm.ppf(1 - alpha / 2))

    def f(c):
        return float(norm.cdf(c + delta)) - float(norm.cdf(-c)) - (1 - alpha)

    cutoff = float(brentq(f, z1 - 1e-9, z2 + 1e-9, xtol=1e-12))
    return {
        "cutoff": cutoff, "delta": delta, "level": level,
        "interval": {"lower": lower - cutoff * se_lower, "upper": upper + cutoff * se_upper},
        "region_interval": {"lower": lower - z2 * se_lower, "upper": upper + z2 * se_upper},
        "z_one_sided": z1, "z_two_sided": z2,
        "coverage_two_sided": float(norm.cdf(z2 + delta)) - float(norm.cdf(-z2)),
        "theorems": ["Research.P11.region_coverage_le", "Research.P11.coverage_strictMono_c",
                     "Research.P11.im_cutoff_antitone", "Research.P11.im_cutoff_between",
                     "Research.P11.two_sided_overcovers"],
    }


def sentence_effect_mts(y, z, weights=None, contrast=None) -> dict:
    """Monotone-treatment-selection bounds for a sentencing contrast (Manski & Pepper 2000).

    The observed mean of the harsher group bounds ``E[y(b)]`` from above and the
    lighter group's bounds ``E[y(a)]`` from below (``Research.P11.Pop.mts_mean_b_le``,
    ``mts_mean_a_ge``); the naive difference overstates the effect
    (``mts_ate_le_naive``); with MTR as well the contrast lies in ``[0, naive]``
    (``mtr_mts_bounds``).

    Examples
    --------
    >>> r = sentence_effect_mts([1, 0, 1, 0, 1, 1], ["a", "a", "a", "b", "b", "b"])
    >>> ({k: round(v, 12) for k, v in r["ate_bounds_mts"].items()}, r["ate_bounds_mtr_mts"])
    ({'lower': -0.5, 'upper': 0.0}, {'lower': 0, 'upper': 0.0})
    """
    b = sentence_effect_bounds(y, z, weights, contrast)
    trt = b["levels"]["treatment"]; ctl = b["levels"]["comparison"]
    m_b = b["joint"][trt] / b["pz"][trt]; m_a = b["joint"][ctl] / b["pz"][ctl]
    pzb = b["pz"][trt]; pza = b["pz"][ctl]
    mean_b = {"lower": pzb * m_b, "upper": m_b}
    mean_a = {"lower": m_a, "upper": pza * m_a + pzb}
    naive = m_b - m_a
    return {
        "levels": b["levels"], "observed_means": {"comparison": m_a, "treatment": m_b}, "pz": b["pz"],
        "mean_b_bounds": mean_b, "mean_a_bounds": mean_a,
        "ate_bounds_mts": {"lower": mean_b["lower"] - mean_a["upper"], "upper": naive},
        "ate_bounds_mtr_mts": {"lower": 0, "upper": max(0.0, naive)},
        "naive_difference": naive,
        "theorems": ["Research.P11.Pop.mts_mean_b_le", "Research.P11.Pop.mts_mean_a_ge",
                     "Research.P11.Pop.mts_ate_le_naive", "Research.P11.Pop.mtr_mts_bounds"],
    }
