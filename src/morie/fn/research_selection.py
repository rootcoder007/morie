# morie.fn -- function file (rootcoder007/morie)
"""Research P2, P6, P8: selection, aggregation and deterrence identification.

Python twin of ``R/research_p268.R``; machine-checked in
``research/lean/P2Selection.lean``, ``P2Benchmark.lean``,
``P6AgeCrime.lean`` and ``P8Deterrence.lean``:

- ``Research.P2.rate_bounds`` / ``disparity_bounds`` / ``disparity_sign_identified``:
  exposure within a factor gamma of the proxy bounds rates by gamma and ratios by gamma^2
- ``Research.P2.benchmark_product`` / ``benchmark_not_additive`` / ``offset_shift``
- ``Research.P2.or_eq_rr_mul`` / ``rr_between`` / ``or_overstates``
- ``Research.P2.collider_or_eq_background``: conditioning on arrest induces an odds ratio
- ``Research.P6.aggregate_not_identifying``: any mixture's aggregate curve is a one-type curve
- ``Research.P8.constant_dimension_not_identified`` / ``certainty_monotone`` /
  ``necessity_bounds``
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = [
    "disparity_exposure_bounds",
    "age_crime_aggregate",
    "deterrence_design_check",
    "disparity_benchmark",
    "relative_risk_from_or",
    "deterrence_response",
    "interracial_rates",
    "probability_of_necessity",
    "collider_arrest",
]


def disparity_exposure_bounds(y, m, gamma=1.0, reference=None):
    """Bounds on a police-outcome disparity when exposure is measured by a proxy.

    Recorded outcomes per group ``y`` are rates per unit of true exposure,
    measured only by a proxy ``m`` known to be within a factor ``gamma`` of
    it. The true rate then lies in ``[y/m / gamma, gamma y/m]`` and the
    ratio to the reference group within a factor ``gamma**2`` of the proxy
    ratio; the direction is identified when the proxy ratio lies outside
    ``[1/gamma**2, gamma**2]``.

    Parameters
    ----------
    y, m : dict
        Outcomes and proxy exposures keyed by group; positive, same groups.
    gamma : float
        Largest credible proxy error factor, at least 1.
    reference : str, optional
        Reference group (default the first key of ``y``).

    Returns
    -------
    dict
        Columns ``group``, ``y``, ``m``, ``proxy_rate``, ``rate_lower``,
        ``rate_upper``, ``ratio_proxy``, ``ratio_lower``, ``ratio_upper``,
        ``direction_identified``.

    Examples
    --------
    >>> d = disparity_exposure_bounds({"A": 300, "B": 100}, {"A": 1000, "B": 1000}, gamma=1.5, reference="B")
    >>> [round(v, 12) for v in d["ratio_proxy"]], [round(v, 12) for v in d["ratio_lower"]], d["direction_identified"]
    ([3.0, 1.0], [1.333333333333, 0.444444444444], [True, False])
    """
    if not isinstance(y, dict) or not isinstance(m, dict) or set(y) != set(m):
        raise ValueError("y and m must be named vectors over the same groups")
    if any(v <= 0 for v in y.values()) or any(v <= 0 for v in m.values()):
        raise ValueError("y and m must be positive")
    if not isinstance(gamma, (int, float)) or math.isnan(gamma) or gamma < 1:
        raise ValueError("gamma must be a single number >= 1")
    g = list(y)
    if reference is None:
        reference = g[0]
    if reference not in y:
        raise ValueError("reference must be one of the groups")
    pr = [y[k] / m[k] for k in g]
    ref = y[reference] / m[reference]
    ratio = [v / ref for v in pr]
    g2 = gamma**2
    return {
        "group": g,
        "y": [float(y[k]) for k in g],
        "m": [float(m[k]) for k in g],
        "proxy_rate": pr,
        "rate_lower": [v / gamma for v in pr],
        "rate_upper": [gamma * v for v in pr],
        "ratio_proxy": ratio,
        "ratio_lower": [v / g2 for v in ratio],
        "ratio_upper": [g2 * v for v in ratio],
        "direction_identified": [v > g2 or v < 1 / g2 for v in ratio],
    }


def age_crime_aggregate(shares, curves, ages=None):
    """Aggregate age-crime curve of a mixture of latent types.

    Returns ``A(a) = sum_g pi_g lambda_g(a)``. The aggregate is itself a
    valid one-type curve, so the aggregate shape cannot tell a single
    invariant curve from a mixture of differently shaped ones.

    Parameters
    ----------
    shares : sequence of float
        Type shares, non-negative, summing to 1.
    curves : dict or sequence of sequences
        One age curve per type: a dict ``{type: curve}`` or a list of curves.
    ages : sequence, optional
        Age labels (default 1..number of ages).

    Returns
    -------
    dict
        ``age``, ``aggregate``, one column per type, ``equivalent_single_type``
        and ``theorems``.

    Examples
    --------
    >>> a = age_crime_aggregate([0.25, 0.75], {"early": [4, 8, 2, 1], "late": [0, 2, 6, 3]}, ages=[15, 20, 25, 30])
    >>> a["aggregate"]
    [1.0, 3.5, 5.0, 2.5]
    """
    if isinstance(curves, dict):
        names = list(curves)
        cols = [[float(v) for v in curves[k]] for k in names]
    else:
        cols = [[float(v) for v in c] for c in curves]
        names = [f"type{j + 1}" for j in range(len(cols))]
    shares = [float(v) for v in shares]
    if len(shares) != len(cols):
        raise ValueError("one share per type column")
    if any(v < 0 for v in shares) or abs(math.fsum(shares) - 1) > 1e-8:
        raise ValueError("shares must be non-negative and sum to 1")
    na = len(cols[0])
    if ages is None:
        ages = list(range(1, na + 1))
    agg = [math.fsum(shares[j] * cols[j][i] for j in range(len(cols))) for i in range(na)]
    out = {"age": list(ages), "aggregate": agg}
    for k, c in zip(names, cols):
        out[k] = c
    out["equivalent_single_type"] = list(agg)
    out["theorems"] = ["Research.P6.aggregate_not_identifying", "Research.P6.invariance_not_necessary"]
    return out


def _rank(cols, tol=1e-7):
    """Numerical rank with R's qr() criterion: a column whose residual norm after
    projecting out the kept columns falls below tol times its own norm is deficient."""
    basis = []
    for col in cols:
        v = list(col)
        nrm0 = math.sqrt(math.fsum(a * a for a in v))
        for q in basis:
            d = math.fsum(a * b for a, b in zip(v, q))
            v = [a - d * b for a, b in zip(v, q)]
        nrm = math.sqrt(math.fsum(a * a for a in v))
        if nrm0 > 0 and nrm > tol * nrm0:
            basis.append([a / nrm for a in v])
    return len(basis)


def deterrence_design_check(p, s, c):
    """Which deterrence partial effects a design can identify.

    For the linear response ``y = b0 + bp p + bs s + bc c`` over a set of
    sanction regimes, the coefficients are identified exactly when the
    design matrix ``[1, p, s, c]`` has full column rank; a dimension held
    constant in the design has an unidentified partial effect. In a
    rank-deficient design a dimension is identified iff dropping its column
    lowers the rank.

    Parameters
    ----------
    p, s, c : sequence of float
        Certainty, severity and celerity per regime.

    Returns
    -------
    RichResult
        ``rank``, ``identified`` and ``constant`` (dicts over
        ``certainty``, ``severity``, ``celerity``), ``n_regimes`` and ``theorem``.

    Examples
    --------
    >>> d = deterrence_design_check(p=[0.1, 0.3, 0.5], s=[2, 2, 2], c=[30, 30, 30])
    >>> d.rank, d.identified
    (2, {'certainty': True, 'severity': False, 'celerity': False})
    >>> deterrence_design_check([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]).rank
    4
    """
    n = len(p)
    if len(s) != n or len(c) != n:
        raise ValueError("p, s and c must have equal length")
    X = [[1.0] * n, [float(v) for v in p], [float(v) for v in s], [float(v) for v in c]]
    rk = _rank(X)
    full = rk == 4
    keys = ("certainty", "severity", "celerity")
    identified = dict.fromkeys(keys, full)
    if not full:
        for j in (1, 2, 3):
            identified[keys[j - 1]] = _rank([X[i] for i in range(4) if i != j]) < rk
    return RichResult(
        title="Deterrence design identification",
        payload={
            "rank": rk,
            "identified": identified,
            "n_regimes": n,
            "constant": {k: len(set(v)) == 1 for k, v in zip(keys, (p, s, c))},
            "theorem": "Research.P8.constant_dimension_not_identified",
        },
    )


def disparity_benchmark(pop, contact, force, reference, exposure_error_factor=None):
    """Disparity benchmarks: the product identity and the exposure-offset shift.

    The resident-benchmarked disparity is the product (not the sum) of the
    contact disparity and the force-given-contact disparity. A group-specific
    exposure measurement error by factor ``k_g`` shifts a log-rate offset by
    ``-log k_g`` and the resident disparity by ``k_ref / k_g``.

    Parameters
    ----------
    pop, contact, force : dict
        Counts keyed by group, same keys in the same order; positive.
    reference : str
        Reference group.
    exposure_error_factor : dict, optional
        Positive exposure error factor per group.

    Returns
    -------
    dict
        Columns ``group``, ``contact_disparity``,
        ``force_given_contact_disparity``, ``resident_disparity``,
        ``additive_claim``, ``product_check`` (zero up to rounding), and with
        ``exposure_error_factor`` ``log_shift`` and
        ``resident_disparity_corrected``; plus ``theorems``.

    Examples
    --------
    >>> b = disparity_benchmark({"A": 100, "B": 100}, {"A": 30, "B": 10}, {"A": 12, "B": 2}, reference="B")
    >>> [round(v, 12) for v in b["contact_disparity"]], b["force_given_contact_disparity"], b["resident_disparity"], b["additive_claim"]
    ([3.0, 1.0], [2.0, 1.0], [6.0, 1.0], [5.0, 2.0])
    """
    g = list(pop)
    if not g or list(contact) != g or list(force) != g:
        raise ValueError("pop, contact and force must share the same group names")
    if any(v <= 0 for d in (pop, contact, force) for v in d.values()):
        raise ValueError("all counts must be positive")
    if reference not in g:
        raise ValueError("reference must be one of the group names")
    r = reference
    cd = [(contact[k] / pop[k]) / (contact[r] / pop[r]) for k in g]
    fd = [(force[k] / contact[k]) / (force[r] / contact[r]) for k in g]
    rd = [(force[k] / pop[k]) / (force[r] / pop[r]) for k in g]
    out = {
        "group": g,
        "contact_disparity": cd,
        "force_given_contact_disparity": fd,
        "resident_disparity": rd,
        "additive_claim": [a + b for a, b in zip(cd, fd)],
        "product_check": [c - a * b for a, b, c in zip(cd, fd, rd)],
    }
    if exposure_error_factor is not None:
        k = [exposure_error_factor.get(v, math.nan) for v in g]
        if any(math.isnan(v) or v <= 0 for v in k):
            raise ValueError("exposure_error_factor must be positive for every group")
        kr = exposure_error_factor[r]
        out["log_shift"] = [-math.log(v) for v in k]
        out["resident_disparity_corrected"] = [a * (kr / v) for a, v in zip(rd, k)]
    out["theorems"] = [
        "Research.P2.benchmark_product",
        "Research.P2.benchmark_not_additive",
        "Research.P2.offset_shift",
        "Research.P2.disparity_ratio_shift",
    ]
    return out


def _solve_unexposed(odds_ratio, base_rate, exposed_share):
    """Root b of s a(b) + (1 - s) b = q, a(b) = OR b / (1 - b + OR b), by bisection
    to machine precision (the left side is strictly increasing in b)."""

    def f(b):
        return exposed_share * (odds_ratio * b / (1 - b + odds_ratio * b)) + (1 - exposed_share) * b - base_rate

    lo, hi = 1e-12, 1 - 1e-12
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return lo if abs(f(lo)) <= abs(f(hi)) else hi


def relative_risk_from_or(odds_ratio, base_rate=None, exposed_share=None):
    """Relative risk from an odds ratio: the interval arrest-only data allow.

    Case-control and arrest-only samples identify the odds ratio, not the
    relative risk; without a base rate the relative risk lies between 1 and
    the odds ratio, which always overstates it. With the population base
    rate and the exposed share supplied, both risks and the relative risk
    are point identified.

    Parameters
    ----------
    odds_ratio : float
        Positive odds ratio.
    base_rate, exposed_share : float, optional
        Population outcome rate and exposed share, each in (0, 1).

    Returns
    -------
    RichResult
        ``rr_bounds`` (``lower``, ``upper``), ``theorems`` and, with both
        optional arguments, ``risks`` (``exposed``, ``unexposed``,
        ``relative_risk``) and ``overstatement_factor``.

    Examples
    --------
    >>> relative_risk_from_or(3).rr_bounds
    {'lower': 1, 'upper': 3}
    >>> r = relative_risk_from_or(3, base_rate=0.2, exposed_share=0.3)
    >>> {k: round(v, 9) for k, v in r.risks.items()}, round(r.overstatement_factor, 9)
    ({'exposed': 0.333333333, 'unexposed': 0.142857143, 'relative_risk': 2.333333333}, 1.285714286)
    """
    if not isinstance(odds_ratio, (int, float)) or math.isnan(odds_ratio) or odds_ratio <= 0:
        raise ValueError("odds_ratio must be a single positive number")
    out = {
        "rr_bounds": {"lower": min(1, odds_ratio), "upper": max(1, odds_ratio)},
        "theorems": ["Research.P2.or_eq_rr_mul", "Research.P2.rr_between", "Research.P2.or_overstates"],
    }
    if base_rate is not None and exposed_share is not None:
        if base_rate <= 0 or base_rate >= 1 or exposed_share <= 0 or exposed_share >= 1:
            raise ValueError("base_rate and exposed_share must lie in (0, 1)")
        b = _solve_unexposed(odds_ratio, base_rate, exposed_share)
        a = odds_ratio * b / (1 - b + odds_ratio * b)
        out["risks"] = {"exposed": a, "unexposed": b, "relative_risk": a / b}
        out["overstatement_factor"] = odds_ratio / (a / b)
    return RichResult(title="Relative risk from an odds ratio", payload=out)


def deterrence_response(x, benefit, sanction, p):
    """Direction of deterrence without convexity.

    An offender picks the level ``x`` from a finite menu to maximise
    ``benefit(x) - p sanction(x)`` with sanction strictly increasing in
    ``x``. The largest optimal level never rises with the certainty ``p``
    (or with a uniform increase in severity), whatever the shape of the
    benefit.

    Parameters
    ----------
    x, benefit, sanction : sequence of float
        Menu of levels with their benefits and sanctions.
    p : float or sequence of float
        Certainty values, non-negative.

    Returns
    -------
    dict
        Columns ``p``, ``x_opt`` (largest maximiser), ``value`` and ``theorems``.

    Examples
    --------
    >>> xs = list(range(11))
    >>> ben = [3 * v**0.5 - (v % 3 == 0) for v in xs]
    >>> r = deterrence_response(xs, ben, [v**1.5 for v in xs], [0.1, 0.3, 0.5, 1])
    >>> r["x_opt"]
    [10, 4, 2, 1]
    """
    n = len(x)
    if len(benefit) != n or len(sanction) != n:
        raise ValueError("x, benefit and sanction must have equal length")
    o = sorted(range(n), key=lambda i: x[i])
    xs = [x[i] for i in o]
    ben = [float(benefit[i]) for i in o]
    san = [float(sanction[i]) for i in o]
    if any(san[i + 1] - san[i] <= 0 for i in range(n - 1)):
        raise ValueError("sanction must be strictly increasing in x")
    ps = [p] if isinstance(p, (int, float)) else list(p)
    if any(v < 0 for v in ps):
        raise ValueError("p must be non-negative")
    rows = {"p": [], "x_opt": [], "value": []}
    for pp in ps:
        u = [b - pp * s for b, s in zip(ben, san)]
        mu = max(u)
        rows["p"].append(pp)
        rows["x_opt"].append(max(xs[i] for i in range(n) if u[i] >= mu - 1e-12))
        rows["value"].append(mu)
    rows["theorems"] = [
        "Research.P8.certainty_monotone",
        "Research.P8.severity_monotone",
        "Research.P8.aggregate_monotone",
    ]
    return rows


def interracial_rates(offences, population):
    """Interracial offending rates against the random-mixing null.

    Under random mixing the expected number of A-on-B offences is
    proportional to the pair exposure ``p_A p_B N``, so per-offender-group
    rates rise with the victim group's population share even when the
    rate per pair exposure is constant. Reports both rates and the ratio of
    each dyad's pair-exposure rate to the pooled rate.

    Parameters
    ----------
    offences : dict
        Counts keyed ``"<offender>_on_<victim>"``.
    population : dict
        Population per group; positive.

    Returns
    -------
    dict
        Columns ``offender``, ``victim``, ``count``, ``rate_per_offender_group``,
        ``rate_per_pair_exposure``, ``null_rate_per_offender_group``,
        ``ratio_to_null`` and ``theorems``.

    Examples
    --------
    >>> r = interracial_rates({"A_on_B": 120, "B_on_A": 200, "A_on_A": 900, "B_on_B": 300}, {"A": 80000, "B": 20000})
    >>> [round(v, 9) for v in r["ratio_to_null"]]
    [0.493421053, 0.822368421, 0.925164474, 4.934210526]
    """
    off, vic = [], []
    for k in offences:
        parts = k.split("_on_")
        if len(parts) != 2:
            raise ValueError("offence names must be of the form offender_on_victim")
        off.append(parts[0])
        vic.append(parts[1])
    if not all(v in population for v in off + vic):
        raise ValueError("every group in offences must appear in population")
    if any(v <= 0 for v in population.values()):
        raise ValueError("populations must be positive")
    N = math.fsum(population.values())
    p = {k: v / N for k, v in population.items()}
    cnt = [float(v) for v in offences.values()]
    pair = [p[a] * p[b] * N for a, b in zip(off, vic)]
    k_hat = math.fsum(cnt) / math.fsum(pair)
    return {
        "offender": off,
        "victim": vic,
        "count": cnt,
        "rate_per_offender_group": [c / population[a] for c, a in zip(cnt, off)],
        "rate_per_pair_exposure": [c / q for c, q in zip(cnt, pair)],
        "null_rate_per_offender_group": [k_hat * p[b] for b in vic],
        "ratio_to_null": [(c / q) / k_hat for c, q in zip(cnt, pair)],
        "theorems": [
            "Research.P2.rate_per_offender_group",
            "Research.P2.null_slope_positive",
            "Research.P2.dyad_ratio",
            "Research.P2.pair_exposure_rate_constant",
        ],
    }


def _all_equal(a, b, tol=1.5e-8):
    """R's isTRUE(all.equal(a, b)) for two numbers."""
    xn = abs(a)
    d = abs(a - b)
    return (d / xn if math.isfinite(xn) and xn > tol else d) < tol


def probability_of_necessity(p_treated, p_control):
    """Probability of necessity: bounds for attributing an outcome to a sanction.

    With outcome probabilities ``a`` under the sanction and ``b`` without,
    the share of the population for whom the sanction was necessary lies in
    ``[max(0, a - b), min(a, 1 - b)]`` and the probability of necessity in
    that interval divided by ``a``; under monotonicity it is ``(a - b)/a``.

    Parameters
    ----------
    p_treated, p_control : float
        Outcome probabilities in [0, 1].

    Returns
    -------
    RichResult
        ``necessary_share_bounds``, ``pn_bounds``, ``pn_monotone``,
        ``identified`` and ``theorems``.

    Examples
    --------
    >>> r = probability_of_necessity(0.6, 0.4)
    >>> {k: round(v, 12) for k, v in r.pn_bounds.items()}, round(r.pn_monotone, 12), r.identified
    ({'lower': 0.333333333333, 'upper': 1.0}, 0.333333333333, False)
    """
    a, b = p_treated, p_control
    if not all(isinstance(v, (int, float)) for v in (a, b)) or any(v < 0 or v > 1 for v in (a, b)):
        raise ValueError("probabilities must be single numbers in [0, 1]")
    lo = max(0, a - b)
    hi = min(a, 1 - b)
    return RichResult(
        title="Probability of necessity",
        payload={
            "necessary_share_bounds": {"lower": lo, "upper": hi},
            "pn_bounds": {"lower": lo / a, "upper": hi / a} if a > 0 else {"lower": math.nan, "upper": math.nan},
            "pn_monotone": (a - b) / a if a > 0 else math.nan,
            "identified": _all_equal(lo, hi),
            "theorems": [
                "Research.P8.necessity_bounds",
                "Research.P8.necessity_of_monotone",
                "Research.P8.necessity_of_disjoint",
            ],
        },
    )


def collider_arrest(a, e, pi_bg):
    """Collider bias from conditioning on arrest.

    Two independent risk factors with prevalences ``a`` and ``e``; anyone
    with either factor is arrested, others with background probability
    ``pi_bg``. Among arrestees the odds ratio of the two factors is
    ``pi_bg`` (below one unless ``pi_bg = 1``) although it is one in the
    population (Hernan and Robins, Fine Point 8.2).

    Parameters
    ----------
    a, e : float
        Prevalences in (0, 1).
    pi_bg : float
        Background arrest probability in [0, 1].

    Returns
    -------
    RichResult
        ``population_or``, ``arrestee_or``, ``arrestee_log_or``,
        ``cells_among_arrestees`` and ``theorems``.

    Examples
    --------
    >>> r = collider_arrest(0.3, 0.2, 0.1)
    >>> round(r.arrestee_or, 12), round(r.arrestee_log_or, 12)
    (0.1, -2.302585092994)
    """
    if a <= 0 or a >= 1 or e <= 0 or e >= 1 or pi_bg < 0 or pi_bg > 1:
        raise ValueError("a and e must lie in (0, 1) and pi_bg in [0, 1]")
    cells = {"A1E1": a * e, "A1E0": a * (1 - e), "A0E1": (1 - a) * e, "A0E0": (1 - a) * (1 - e) * pi_bg}
    oddsr = (cells["A1E1"] * cells["A0E0"]) / (cells["A1E0"] * cells["A0E1"])
    tot = math.fsum(cells.values())
    return RichResult(
        title="Collider bias from conditioning on arrest",
        payload={
            "population_or": 1,
            "arrestee_or": oddsr,
            "arrestee_log_or": math.log(oddsr) if oddsr > 0 else -math.inf,
            "cells_among_arrestees": {k: v / tot for k, v in cells.items()},
            "theorems": [
                "Research.P2.population_or_one",
                "Research.P2.collider_or_eq_background",
                "Research.P2.collider_or_lt_one",
            ],
        },
    )


def cheatsheet() -> str:
    return (
        "disparity_exposure_bounds(y, m, gamma) -> rate and ratio bounds under proxy exposure (Research P2)\n"
        "age_crime_aggregate(shares, curves) -> mixture age curve (Research P6)\n"
        "deterrence_design_check(p, s, c) -> identified deterrence dimensions (Research P8)\n"
        "disparity_benchmark(pop, contact, force, reference) -> product identity of benchmarks\n"
        "relative_risk_from_or(OR, base_rate, exposed_share) -> RR interval / point\n"
        "deterrence_response(x, benefit, sanction, p) -> optimal offending level per certainty\n"
        "interracial_rates(offences, population) -> dyad rates vs random-mixing null\n"
        "probability_of_necessity(a, b) -> PN bounds\n"
        "collider_arrest(a, e, pi_bg) -> odds ratio among arrestees"
    )
