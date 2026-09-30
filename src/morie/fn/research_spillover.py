# morie.fn -- function file (rootcoder007/morie)
"""Research P3: policing effects under interference.

Python twin of ``R/spillover.R`` and of its C++ exposure kernel
(``src/morie_spillover.cpp``). Partial interference through a stated
exposure mapping: 2 = treated, 1 = untreated with a treated neighbour,
0 = untreated and isolated. Machine-checked in
``research/lean/P3Interference.lean``, ``P3HorvitzThompson.lean``,
``P3Variance.lean`` and ``P3SYG.lean``:

- ``Research.P3.Model.exposure_adjustment``: E[Y(l)] = sum_k m_k mean(Y | stratum k, exposure l)
- ``Research.P3.Model.direct_spillover_decomposition`` / ``misspecified_exposure_bias``
- ``Research.P3.Design.ht_unbiased`` / ``ht_contrast_unbiased``
- ``Research.P3.Design.ht_variance`` / ``ht_variance_estimator_unbiased``
- ``Research.P3.SYG.syg_eq_ht`` / ``syg_zero_of_const``
"""

from __future__ import annotations

import itertools
import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "spillover_exposure",
    "spillover_effects",
    "spillover_ht",
    "spillover_ht_variance",
    "spillover_exposure_probs",
]


def _exposure_kernel(treated, edges):
    """Pure-Python twin of morie_spillover_exposure_cpp (0-based edges)."""
    n = len(treated)
    nb = [0] * n
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            raise ValueError("edge index out of range")
        if a == b:
            continue
        if treated[b] == 1:
            nb[a] += 1
        if treated[a] == 1:
            nb[b] += 1
    exp = [2 if treated[i] == 1 else (1 if nb[i] > 0 else 0) for i in range(n)]
    return exp, nb


def _edges(edges, n):
    out = []
    for e in edges:
        if len(e) != 2:
            raise ValueError("edges must have two columns")
        a, b = e
        if a != int(a) or b != int(b) or a < 1 or b < 1 or a > n or b > n:
            raise ValueError("edges must index places 1..length(treated)")
        out.append((int(a) - 1, int(b) - 1))
    return out


def spillover_exposure(treated, edges):
    """Three-level exposure from a treatment on a place network.

    2 if treated, 1 if untreated with at least one treated neighbour, 0 if
    untreated and isolated from treatment. Pass the adjacency of the ring
    you are willing to assume spillover stops at.

    Parameters
    ----------
    treated : sequence of bool or int
        One entry per place.
    edges : sequence of (int, int)
        Undirected adjacency as 1-based place index pairs.

    Returns
    -------
    dict
        Columns ``place``, ``treated``, ``treated_neighbours``, ``exposure``.

    Examples
    --------
    >>> e = spillover_exposure([1, 0, 0, 0], [(1, 2), (2, 3), (3, 4)])
    >>> e["exposure"], e["treated_neighbours"]
    ([2, 1, 0, 0], [0, 1, 0, 0])
    """
    if any(t is None or (isinstance(t, float) and math.isnan(t)) for t in treated):
        raise ValueError("treated must not contain NA")
    tr = [1 if t else 0 for t in treated]
    exp, nb = _exposure_kernel(tr, _edges(edges, len(tr)))
    return {"place": list(range(1, len(tr) + 1)), "treated": tr, "treated_neighbours": nb, "exposure": exp}


def spillover_effects(y, exposure, stratum, weights=None):
    """Direct, spillover and net effects under a stated exposure mapping.

    Within each stratum the outcome is averaged by exposure level and the
    stratum means are combined with the stratum weights. Under positivity
    and exposure ignorability given the stratum these are ``E[Y(0)]``,
    ``E[Y(1)]``, ``E[Y(2)]``; the total effect splits exactly into spillover
    and direct parts. The pooling bias is how far pooling exposures 0 and 1
    as "control" sits from ``E[Y(0)]``.

    Parameters
    ----------
    y : sequence of float
        Outcome per place.
    exposure : sequence of int
        Exposure level per place (0, 1, 2).
    stratum : sequence
        Covariate stratum per place.
    weights : sequence of float, optional
        Non-negative place weights.

    Returns
    -------
    RichResult
        ``means`` (``Y0``, ``Y1``, ``Y2``), ``spillover``, ``direct``,
        ``total``, ``pooling_bias`` (``by_stratum``, ``overall``),
        ``cell_means``, ``stratum_weights``, ``positivity`` and ``theorems``.
        Cells without data are ``nan``, as are the means they feed.

    Examples
    --------
    >>> y = [5.0, 4.6, 4.0, 6.1, 5.5, 5.0, 4.9, 4.5]
    >>> ex = [0, 1, 2, 0, 1, 2, 0, 2]
    >>> st = ["core", "core", "core", "edge", "edge", "edge", "core", "edge"]
    >>> r = spillover_effects(y, ex, st)
    >>> round(r.spillover, 12), round(r.direct, 12), round(r.total, 12)
    (-0.475, -0.675, -1.15)
    """
    n = len(y)
    if len(exposure) != n or len(stratum) != n:
        raise ValueError("y, exposure and stratum must have equal length")
    if not all(e in (0, 1, 2) for e in exposure):
        raise ValueError("exposure must take values 0, 1, 2")
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n or any(w < 0 for w in weights) or math.fsum(weights) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    tw = math.fsum(weights)
    w = [v / tw for v in weights]
    strata = list(dict.fromkeys(stratum))

    def cmean(idx):
        if not idx:
            return math.nan
        return math.fsum(w[i] * y[i] for i in idx) / math.fsum(w[i] for i in idx)

    cells = {
        s: [cmean([i for i in range(n) if stratum[i] == s and exposure[i] == lv]) for lv in (0, 1, 2)] for s in strata
    }
    m_k = {s: math.fsum(w[i] for i in range(n) if stratum[i] == s) for s in strata}
    positivity = not any(math.isnan(v) for s in strata for v in cells[s])
    means = [math.fsum(m_k[s] * cells[s][lv] for s in strata) for lv in (0, 1, 2)]
    means = [math.nan if any(math.isnan(cells[s][lv]) for s in strata) else v for lv, v in enumerate(means)]
    pooled = {s: cmean([i for i in range(n) if stratum[i] == s and exposure[i] in (0, 1)]) for s in strata}
    bias_k = {s: pooled[s] - cells[s][0] for s in strata}
    overall = math.nan if any(math.isnan(v) for v in bias_k.values()) else math.fsum(m_k[s] * bias_k[s] for s in strata)
    return RichResult(
        title="Direct and spillover effects under an exposure mapping",
        payload={
            "means": {"Y0": means[0], "Y1": means[1], "Y2": means[2]},
            "spillover": means[1] - means[0],
            "direct": means[2] - means[1],
            "total": means[2] - means[0],
            "pooling_bias": {"by_stratum": bias_k, "overall": overall},
            "cell_means": cells,
            "stratum_weights": m_k,
            "positivity": positivity,
            "theorems": [
                "Research.P3.Model.exposure_adjustment",
                "Research.P3.Model.direct_spillover_decomposition",
                "Research.P3.Model.misspecified_exposure_bias",
            ],
        },
    )


def spillover_ht(y, exposure, probs, joint=None):
    """Horvitz-Thompson totals and contrasts under a randomised deployment.

    ``T(l) = sum_i 1{E_i = l} y_i / pi_i(l)`` is unbiased for the total of
    ``Y(l)`` whenever every ``pi_i(l) > 0``, and so is any contrast. With
    joint probabilities ``pi_ij(l)`` the Horvitz-Thompson variance
    estimator is unbiased when every ``pi_ij(l) > 0``; a realised pair with
    ``pi_ij(l) = 0`` leaves the variance unidentified (``nan``).

    Parameters
    ----------
    y : sequence of float
        Observed outcome per place.
    exposure : sequence of int
        Realised exposure per place (0, 1, 2).
    probs : sequence of (p0, p1, p2)
        Exposure probabilities per place, all positive.
    joint : list of three n-by-n matrices, optional
        Joint exposure probabilities per level, as returned by
        ``spillover_exposure_probs(..., joint=True)["joint"]``.

    Returns
    -------
    RichResult
        ``totals`` and ``means`` (``T0``, ``T1``, ``T2``), ``spillover``,
        ``direct``, ``total_effect`` and, with ``joint``, ``variance``,
        ``se`` and ``variance_identified``; plus ``theorems``.

    Examples
    --------
    >>> pr = spillover_exposure_probs(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 2, joint=True)
    >>> ex = spillover_exposure([0, 1, 0, 0, 1, 0], [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])["exposure"]
    >>> y = [4.5 if e == 1 else 4.0 if e == 2 else 5.0 for e in ex]
    >>> r = spillover_ht(y, ex, pr["marginal"], pr["joint"])
    >>> round(r.spillover, 9), round(r.direct, 9), [round(v, 9) for v in r.se.values()]
    (8.839285714, -4.839285714, [0.0, 18.617902017, 0.0])
    """
    n = len(y)
    if len(exposure) != n or len(probs) != n or any(len(p) != 3 for p in probs):
        raise ValueError("y, exposure and probs (n x 3) must describe the same places")
    if any(v <= 0 for p in probs for v in p):
        raise ValueError("every exposure probability must be positive (positivity)")
    if not all(e in (0, 1, 2) for e in exposure):
        raise ValueError("exposure must take values 0, 1, 2")
    totals = [math.fsum(y[i] / probs[i][lv] for i in range(n) if exposure[i] == lv) for lv in (0, 1, 2)]
    means = [t / n for t in totals]
    out = {
        "totals": {"T0": totals[0], "T1": totals[1], "T2": totals[2]},
        "means": {"T0": means[0], "T1": means[1], "T2": means[2]},
        "spillover": means[1] - means[0],
        "direct": means[2] - means[1],
        "total_effect": means[2] - means[0],
    }
    theorems = ["Research.P3.Design.ht_unbiased", "Research.P3.Design.ht_contrast_unbiased"]
    if joint is not None:
        if len(joint) != 3 or any(len(J) != n or any(len(r) != n for r in J) for J in joint):
            raise ValueError("joint must be an n x n x 3 array of joint exposure probabilities")
        var = []
        for lv in (0, 1, 2):
            J = joint[lv]
            idx = [i for i in range(n) if exposure[i] == lv]
            if any(J[i][j] <= 0 for i in idx for j in idx):
                var.append(math.nan)
                continue
            c = [y[i] / probs[i][lv] for i in range(n)]
            terms = []
            for i in idx:
                for j in idx:
                    pij = J[i][j]
                    terms.append((pij - probs[i][lv] * probs[j][lv]) / pij * c[i] * c[j])
            var.append(math.fsum(terms))
        keys = ("T0", "T1", "T2")
        out["variance"] = dict(zip(keys, var))
        out["se"] = {k: (math.sqrt(max(v, 0.0)) if not math.isnan(v) else math.nan) for k, v in zip(keys, var)}
        out["variance_identified"] = [all(v > 0 for r in joint[lv] for v in r) for lv in (0, 1, 2)]
        theorems += ["Research.P3.Design.ht_variance", "Research.P3.Design.ht_variance_estimator_unbiased"]
    out["theorems"] = theorems
    return RichResult(title="Horvitz-Thompson spillover contrasts", payload=out)


def spillover_ht_variance(y_pot, pi, pij, form="ht"):
    """Design variance of the Horvitz-Thompson total (population formula).

    ``sum_ij (pi_ij - pi_i pi_j) Y_i Y_j / (pi_i pi_j)`` for a known
    potential-outcome vector; the Sen-Yates-Grundy form
    ``1/2 sum_ik (pi_i pi_k - pi_ik)(c_i - c_k)^2``, ``c_i = Y_i/pi_i``,
    equals it on a fixed-size design (``sum_k pi_ik = m pi_i``).

    Parameters
    ----------
    y_pot : sequence of float
        Potential outcome per place at the level of interest.
    pi : sequence of float
        Marginal exposure probability per place, positive.
    pij : n-by-n matrix
        Joint exposure probabilities.
    form : {"ht", "syg", "both"}

    Returns
    -------
    float or RichResult
        The variance, or for ``"both"`` ``ht``, ``syg``, ``fixed_size``,
        ``level_count`` and ``theorems``.

    Examples
    --------
    >>> pr = spillover_exposure_probs(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 2, joint=True)
    >>> round(spillover_ht_variance([1.0] * 6, [p[2] for p in pr["marginal"]], pr["joint"][2]), 12)
    0.0
    >>> v = spillover_ht_variance([1.0, 2, 3, 4, 5, 6], [p[1] for p in pr["marginal"]], pr["joint"][1], "both")
    >>> round(v.ht, 9), round(v.syg, 9), v.fixed_size
    (99.153061224, -3.933673469, False)
    """
    if form not in ("ht", "syg", "both"):
        raise ValueError("form must be 'ht', 'syg' or 'both'")
    n = len(y_pot)
    if len(pi) != n or len(pij) != n or any(len(r) != n for r in pij):
        raise ValueError("y_pot, pi and pij must describe the same places")
    if any(v <= 0 for v in pi):
        raise ValueError("every marginal probability must be positive")
    c = [a / b for a, b in zip(y_pot, pi)]
    ht = math.fsum((pij[i][j] - pi[i] * pi[j]) * (c[i] * c[j]) for i in range(n) for j in range(n))
    if form == "ht":
        return ht
    syg = 0.5 * math.fsum((pi[i] * pi[j] - pij[i][j]) * (c[i] - c[j]) ** 2 for i in range(n) for j in range(n))
    if form == "syg":
        return syg
    m = math.fsum(pi)
    fixed = max(abs(math.fsum(pij[i]) - m * pi[i]) for i in range(n)) < 1e-10
    return RichResult(
        title="Design variance of the Horvitz-Thompson total",
        payload={
            "ht": ht,
            "syg": syg,
            "fixed_size": fixed,
            "level_count": m,
            "theorems": [
                "Research.P3.Design.ht_variance",
                "Research.P3.SYG.syg_eq_ht",
                "Research.P3.SYG.syg_zero_of_const",
            ],
        },
    )


def spillover_exposure_probs(n, edges, n_treated, n_draws=5000, exact_max=20000, joint=False, seed=0):
    """Exposure probabilities of a completely randomised deployment.

    Share of assignments of ``n_treated`` treated places out of ``n`` in
    which each place sits at each exposure level. All ``C(n, n_treated)``
    assignments are enumerated when that is at most ``exact_max``;
    otherwise ``n_draws`` assignments are drawn from the Philox stream (draw
    ``k`` treats the ``n_treated`` places with the smallest of
    ``n`` uniforms ``k n + 1 .. (k + 1) n`` from ``seed``, stream 0),
    identical to the R arm.

    Parameters
    ----------
    n : int
        Number of places.
    edges : sequence of (int, int)
        Adjacency as in :func:`spillover_exposure`.
    n_treated : int
        Treated places per assignment, in 1..n-1.
    n_draws : int
        Monte Carlo draws when enumeration is too large.
    exact_max : int
        Enumeration threshold.
    joint : bool
        Also return joint probabilities per level.
    seed : int
        Philox seed for the Monte Carlo branch.

    Returns
    -------
    list or dict
        ``n`` rows ``[p0, p1, p2]``; with ``joint=True`` a dict with
        ``marginal`` and ``joint`` (three n-by-n matrices, one per level).

    Examples
    --------
    >>> pr = spillover_exposure_probs(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 2)
    >>> [[round(v, 9) for v in row] for row in pr[:2]]
    [[0.4, 0.266666667, 0.333333333], [0.2, 0.466666667, 0.333333333]]
    """
    n = int(n)
    n_treated = int(n_treated)
    if n_treated < 1 or n_treated >= n:
        raise ValueError("n_treated must lie in 1..n-1")
    e = _edges(edges, n)
    counts = [[0.0, 0.0, 0.0] for _ in range(n)]
    jc = [[[0.0] * n for _ in range(n)] for _ in range(3)] if joint else None

    def tally(trt):
        tr = [0] * n
        for i in trt:
            tr[i] = 1
        ex, _ = _exposure_kernel(tr, e)
        for i in range(n):
            counts[i][ex[i]] += 1
        if joint:
            for lv in (0, 1, 2):
                idx = [i for i in range(n) if ex[i] == lv]
                J = jc[lv]
                for i in idx:
                    row = J[i]
                    for j in idx:
                        row[j] += 1

    if math.comb(n, n_treated) <= exact_max:
        combos = list(itertools.combinations(range(n), n_treated))
        for trt in combos:
            tally(trt)
        m = len(combos)
    else:
        m = int(n_draws)
        u = random_uniform(n * m, seed=seed, stream=0)
        for k in range(m):
            uk = u[k * n : (k + 1) * n]
            tally(sorted(range(n), key=lambda i: uk[i])[:n_treated])
    marginal = [[v / m for v in row] for row in counts]
    if joint:
        return {"marginal": marginal, "joint": [[[v / m for v in row] for row in J] for J in jc]}
    return marginal


def cheatsheet() -> str:
    return (
        "spillover_exposure(treated, edges) -> exposure 2/1/0 (Research P3)\n"
        "spillover_effects(y, exposure, stratum) -> adjusted E[Y(l)], spillover, direct, pooling bias\n"
        "spillover_ht(y, exposure, probs, joint) -> Horvitz-Thompson totals, contrasts, variance\n"
        "spillover_ht_variance(y_pot, pi, pij, form) -> HT / SYG design variance\n"
        "spillover_exposure_probs(n, edges, n_treated, joint=False, seed=0) -> exposure probabilities"
    )
