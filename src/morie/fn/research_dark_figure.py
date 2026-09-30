# morie.fn -- function file (rootcoder007/morie)
"""Research P1: the dark figure of crime as a partial-identification problem.

Python twin of ``R/dark_figure.R``; machine-checked in
``research/lean/P1DarkFigure.lean`` (building on ``P5Fairness.lean``):

- ``Research.P1.TwoSource.petersen_identity`` / ``lincoln_petersen`` / ``petersen_bounds``:
  N = theta n1 n2 / m, and theta in [1/kappa, kappa] bounds N sharply
- ``Research.P1.true_rate_bounds`` / ``dark_figure_bounds``:
  max(r, (v_obs - a)/(1 - a)) <= v <= v_obs/(1 - b)
- ``Research.P1.conclusion_holds_below_breakdown`` / ``conclusion_fails_above_breakdown``
- ``Research.P1.three_list_saturated_fits`` / ``missing_cell_unconstrained``
- ``Research.P1.offence_count_bounds`` / ``offence_count_eq`` / ``category_not_identified``

The theorems are about expected counts and rates; list independence and
the misreporting boxes are claims about the world, taken as arguments.
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = [
    "dark_figure_two_source",
    "dark_figure_bounds",
    "dark_figure_breakdown",
    "dark_figure_three_list",
    "dark_figure_hierarchy",
]

_CELLS = ("100", "010", "001", "110", "101", "011", "111")


def _num1(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def _vec(v):
    return [float(v)] if isinstance(v, (int, float)) else [float(a) for a in v]


def dark_figure_two_source(n1, n2, m, kappa=1.0, chapman=False):
    """Two-source (capture-recapture) count with a dependence box.

    Two lists of the same events of sizes ``n1`` and ``n2`` share ``m``
    events. Under independence the true count is the Lincoln-Petersen
    ratio ``n1 n2 / m``; with the dependence factor only known to lie in
    ``[1/kappa, kappa]`` the true count lies in
    ``[n1 n2 / (kappa m), kappa n1 n2 / m]``, both ends attainable.

    Parameters
    ----------
    n1, n2 : float
        Sizes of the two lists; positive.
    m : float
        Events on both lists; positive and at most ``min(n1, n2)``.
    kappa : float
        Largest credible dependence factor, at least 1.
    chapman : bool
        Also return Chapman's ``(n1+1)(n2+1)/(m+1) - 1``.

    Returns
    -------
    RichResult
        ``point``, ``lower``, ``upper``, ``kappa``, ``theorem`` and
        ``chapman`` when requested.

    Examples
    --------
    >>> r = dark_figure_two_source(400, 250, 80, kappa=2, chapman=True)
    >>> r.point, r.lower, r.upper, round(r.chapman, 9)
    (1250.0, 625.0, 2500.0, 1241.604938272)
    """
    for v in (n1, n2, m, kappa):
        if not _num1(v):
            raise ValueError("n1, n2, m and kappa must be single non-missing numbers")
    if n1 <= 0 or n2 <= 0 or m <= 0:
        raise ValueError("n1, n2 and m must be positive")
    if m > min(n1, n2):
        raise ValueError("m cannot exceed the smaller list")
    if kappa < 1:
        raise ValueError("kappa must be at least 1")
    point = n1 * n2 / m
    out = {
        "point": float(point),
        "lower": point / kappa,
        "upper": kappa * point,
        "kappa": kappa,
        "theorem": "Research.P1.TwoSource.lincoln_petersen" if kappa == 1 else "Research.P1.TwoSource.petersen_bounds",
    }
    if chapman:
        out["chapman"] = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    return RichResult(title="Two-source dark-figure count", payload=out)


def dark_figure_bounds(v_obs, r, alpha_max, beta_max):
    """Sharp bounds on a true victimisation rate and its dark figure.

    A recorded rate ``r`` never exceeds the true rate ``v``; a survey rate
    ``v_obs = v (1 - beta) + (1 - v) alpha`` is a noisy proxy. With only
    ``alpha in [0, a]``, ``beta in [0, b]``, ``a + b < 1`` credible,
    ``max(r, (v_obs - a)/(1 - a)) <= v <= v_obs/(1 - b)`` and the dark
    figure ``v - r`` lies in that interval shifted by ``r``.

    Parameters
    ----------
    v_obs : float or sequence of float
        Survey victimisation rate(s) in [0, 1].
    r : float or sequence of float
        Recorded rate(s) in [0, 1], length 1 or the length of ``v_obs``.
    alpha_max, beta_max : float
        Largest credible over- and under-reporting rates, sum below 1.

    Returns
    -------
    dict
        Columns ``v_obs``, ``r``, ``v_lower``, ``v_upper``, ``dark_lower``,
        ``dark_upper`` and ``ratio_upper`` (``nan`` when ``r = 0``).

    Examples
    --------
    >>> b = dark_figure_bounds(0.06, 0.02, alpha_max=0.01, beta_max=0.30)
    >>> [round(b[k][0], 12) for k in ("v_lower", "v_upper", "dark_lower", "dark_upper", "ratio_upper")]
    [0.050505050505, 0.085714285714, 0.030505050505, 0.065714285714, 4.285714285714]
    """
    v = _vec(v_obs)
    if any(math.isnan(a) or a < 0 or a > 1 for a in v):
        raise ValueError("v_obs must be numeric in [0, 1]")
    rr = _vec(r)
    if any(math.isnan(a) or a < 0 or a > 1 for a in rr):
        raise ValueError("r must be numeric in [0, 1]")
    if len(rr) != 1 and len(rr) != len(v):
        raise ValueError("r must have length 1 or the length of v_obs")
    for a in (alpha_max, beta_max):
        if not _num1(a) or a < 0:
            raise ValueError("alpha_max and beta_max must be single non-negative numbers")
    if alpha_max + beta_max >= 1:
        raise ValueError("alpha_max + beta_max must be below 1")
    if len(rr) == 1:
        rr = rr * len(v)
    lo = [max(ri, (vi - alpha_max) / (1 - alpha_max)) for vi, ri in zip(v, rr)]
    up = [max(min(1.0, vi / (1 - beta_max)), li) for vi, li in zip(v, lo)]
    return {
        "v_obs": v,
        "r": rr,
        "v_lower": lo,
        "v_upper": up,
        "dark_lower": [a - b for a, b in zip(lo, rr)],
        "dark_upper": [a - b for a, b in zip(up, rr)],
        "ratio_upper": [a / b if b > 0 else math.nan for a, b in zip(up, rr)],
    }


def dark_figure_breakdown(v_obs, threshold):
    """Breakdown analysis for a dark-figure conclusion.

    "The true rate is below ``threshold``" survives every admissible noise
    pair while the under-reporting box stays below ``1 - v_obs/threshold``
    and fails at or above it.

    Parameters
    ----------
    v_obs : float
        Survey victimisation rate in (0, 1].
    threshold : float
        The rate the conclusion claims is not reached; above ``v_obs``.

    Returns
    -------
    RichResult
        ``breakdown_beta_max``, ``ratio`` (threshold / v_obs) and ``theorems``.

    Examples
    --------
    >>> b = dark_figure_breakdown(0.06, threshold=0.10)
    >>> round(b.breakdown_beta_max, 12), round(b.ratio, 12)
    (0.4, 1.666666666667)
    """
    if not _num1(v_obs) or v_obs <= 0 or v_obs > 1:
        raise ValueError("v_obs must be a single number in (0, 1]")
    if not _num1(threshold) or threshold <= v_obs:
        raise ValueError("threshold must be a single number above v_obs")
    return RichResult(
        title="Dark-figure breakdown",
        payload={
            "breakdown_beta_max": 1 - v_obs / threshold,
            "ratio": threshold / v_obs,
            "theorems": [
                "Research.P1.conclusion_holds_below_breakdown",
                "Research.P1.conclusion_fails_above_breakdown",
            ],
        },
    )


def dark_figure_three_list(counts, candidate_missing=None):
    """Three-list capture-recapture: what is and is not identified.

    The saturated log-linear model reproduces any positive eight-cell table,
    so the unobserved cell is unconstrained by the seven observed ones.
    Fixing the three-way interaction at zero gives
    ``m000 = m111 m100 m010 m001 / (m110 m101 m011)``. Also reported: the
    floor (units seen), the pairwise Petersen and Chapman estimates (each
    at least its pair's floor), and the three-way interaction implied by
    any candidate missing count.

    Parameters
    ----------
    counts : dict
        The seven observed cells keyed ``"100"``, ``"010"``, ``"001"``,
        ``"110"``, ``"101"``, ``"011"``, ``"111"`` (on list 1, 2, 3); all positive.
    candidate_missing : sequence of float, optional
        Positive candidate values of the missing cell.

    Returns
    -------
    RichResult
        ``observed``, ``missing_no_three_way``, ``N_no_three_way``,
        ``pairwise`` (list of dicts per pair of lists), ``implied_three_way``
        (dict of columns, or ``None``) and ``theorems``.

    Examples
    --------
    >>> c = {"100": 120, "010": 90, "001": 70, "110": 40, "101": 30, "011": 25, "111": 15}
    >>> t = dark_figure_three_list(c, candidate_missing=[100, 200])
    >>> t.observed, round(t.missing_no_three_way, 9), round(t.N_no_three_way, 9)
    (390.0, 378.0, 768.0)
    >>> [round(p["petersen"], 6) for p in t.pairwise]
    [633.636364, 637.777778, 595.0]
    >>> [round(v, 12) for v in t.implied_three_way["three_way"]]
    [1.329724009631, 0.636576829072]
    """
    if not isinstance(counts, dict) or not all(k in counts for k in _CELLS):
        raise ValueError("counts must be named by the seven cells 100, 010, 001, 110, 101, 011, 111")
    m = {k: float(counts[k]) for k in _CELLS}
    if any(math.isnan(v) or v <= 0 for v in m.values()):
        raise ValueError("all seven observed cells must be positive")
    observed = math.fsum(m.values())
    m000 = m["111"] * m["100"] * m["010"] * m["001"] / (m["110"] * m["101"] * m["011"])
    pairwise = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        n1 = math.fsum(m[k] for k in _CELLS if k[a] == "1")
        n2 = math.fsum(m[k] for k in _CELLS if k[b] == "1")
        mm = math.fsum(m[k] for k in _CELLS if k[a] == "1" and k[b] == "1")
        pairwise.append(
            {
                "lists": f"{a + 1}-{b + 1}",
                "n1": n1,
                "n2": n2,
                "m": mm,
                "floor": n1 + n2 - mm,
                "petersen": n1 * n2 / mm,
                "chapman": (n1 + 1) * (n2 + 1) / (mm + 1) - 1,
            }
        )
    implied = None
    if candidate_missing is not None:
        cm = _vec(candidate_missing)
        if any(v <= 0 for v in cm):
            raise ValueError("candidate_missing must be positive")
        lm = {k: math.log(v) for k, v in m.items()}
        base = lm["111"] - lm["110"] - lm["101"] - lm["011"] + lm["100"] + lm["010"] + lm["001"]
        implied = {
            "missing": cm,
            "N": [observed + v for v in cm],
            "three_way": [base - math.log(v) for v in cm],
        }
    return RichResult(
        title="Three-list capture-recapture",
        payload={
            "observed": observed,
            "missing_no_three_way": m000,
            "N_no_three_way": observed + m000,
            "pairwise": pairwise,
            "implied_three_way": implied,
            "theorems": [
                "Research.P1.three_list_saturated_fits",
                "Research.P1.missing_cell_unconstrained",
                "Research.P1.petersen_ge_floor",
                "Research.P1.chapman_ge_floor",
            ],
        },
    )


def dark_figure_hierarchy(offences_per_incident):
    """Incident versus offence counting: the hierarchy-rule arithmetic.

    With ``N`` incidents carrying ``k_i >= 1`` offences each, bounded by
    ``K``, the offence count lies in ``[N, K N]`` and equals
    ``N (1 + mean extra offences)``; the incident count alone does not
    identify the offence count, nor any category's share of it.

    Parameters
    ----------
    offences_per_incident : sequence of int
        Offences carried by each incident, each at least 1.

    Returns
    -------
    RichResult
        ``incidents``, ``offences``, ``ratio``, ``mean_extra``, ``bounds``
        (``lower``, ``upper``) and ``theorems``.

    Examples
    --------
    >>> h = dark_figure_hierarchy([1, 1, 2, 1, 3, 1, 1, 2])
    >>> h.incidents, h.offences, h.ratio, h.mean_extra, h.bounds
    (8, 12, 1.5, 0.5, {'lower': 8, 'upper': 24})
    """
    try:
        k = [int(v) for v in offences_per_incident]
    except (TypeError, ValueError):
        raise ValueError("every incident must carry at least one offence") from None
    if not k or any(v < 1 for v in k):
        raise ValueError("every incident must carry at least one offence")
    N = len(k)
    tot = sum(k)
    return RichResult(
        title="Hierarchy-rule arithmetic",
        payload={
            "incidents": N,
            "offences": tot,
            "ratio": tot / N,
            "mean_extra": (tot - N) / N,
            "bounds": {"lower": N, "upper": max(k) * N},
            "theorems": [
                "Research.P1.offence_count_bounds",
                "Research.P1.offence_count_eq",
                "Research.P1.category_not_identified",
            ],
        },
    )


def cheatsheet() -> str:
    return (
        "dark_figure_two_source(n1, n2, m, kappa=1) -> Lincoln-Petersen N with dependence box (Research P1)\n"
        "dark_figure_bounds(v_obs, r, alpha_max, beta_max) -> sharp true-rate and dark-figure interval\n"
        "dark_figure_breakdown(v_obs, threshold) -> under-reporting that overturns 'v < threshold'\n"
        "dark_figure_three_list(counts) -> no-three-way missing cell, pairwise Petersen/Chapman, floors\n"
        "dark_figure_hierarchy(k) -> offence count bounds [N, K N] under a hierarchy rule"
    )
