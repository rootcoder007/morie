# morie.fn -- function file (rootcoder007/morie)
"""Political polarization and party-system measures, and roll-call matrix utilities: dispersion, Esteban-Ray,
Gini mean difference, bimodality, earth mover's distance, issue constraint, trends, Rice cohesion, Rae
fractionalization, party divergence and overlap, roll-call coding, summaries and filtering."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "polarization_indices",
    "esteban_ray_index",
    "earth_movers_distance",
    "issue_constraint",
    "polarization_trend",
    "party_system_indices",
    "party_divergence",
    "rollcall_matrix",
    "rollcall_summary",
    "rollcall_filter",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def polarization_indices(positions, weights=None) -> RichResult:
    r"""Distributional polarization of ideal points or opinions on one dimension.

    Variance (divisor ``n - 1``) and standard deviation; Gini mean
    difference ``sum_i sum_j w_i w_j |x_i - x_j| / (sum w)^2`` (equal
    weights by default); sample skewness ``G1`` and excess kurtosis ``G2``
    (bias-corrected, as SAS/SPSS); and the bimodality coefficient ``b = (G1^2
    + 1)/(G2 + 3 (n - 1)^2/((n - 2)(n - 3)))`` -- values above ``5/9`` point to
    bimodality (SAS Institute 1990; Pfister et al. 2013). Negative excess
    kurtosis is the kurtosis-based polarization signal (DiMaggio, Evans and
    Bryson 1996).

    References
    ----------
    DiMaggio, P., Evans, J. and Bryson, B. (1996). Have Americans' social
    attitudes become more polarized? *American Journal of Sociology*,
    102(3), 690-755.
    Pfister, R., Schwarz, K. A., Janczyk, M., Dale, R. and Freeman, J. B.
    (2013). Good things peak in pairs: a note on the bimodality coefficient.
    *Frontiers in Psychology*, 4, 700.

    Examples
    --------
    >>> r = polarization_indices([-1.0, -1.0, 1.0, 1.0, 0.0])
    >>> round(r.variance, 6), round(r.gini_mean_difference, 6)
    (1.0, 0.96)
    """
    x = _vec(positions)
    n = len(x)
    w = [1.0] * n if weights is None else _vec(weights)
    m = ssum(x) / n
    d = [v - m for v in x]
    var = ssum(v * v for v in d) / (n - 1)
    sw = ssum(w)
    gmd = ssum(w[i] * w[j] * abs(x[i] - x[j]) for i in range(n) for j in range(n)) / sw**2
    m2 = ssum(v * v for v in d) / n
    m3 = ssum(v**3 for v in d) / n
    m4 = ssum(v**4 for v in d) / n
    g1, g2 = m3 / m2**1.5, m4 / m2**2 - 3
    G1 = g1 * math.sqrt(n * (n - 1)) / (n - 2) if n > 2 else float("nan")
    G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6) if n > 3 else float("nan")
    bc = (G1 * G1 + 1) / (G2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))) if n > 3 else float("nan")
    return RichResult(
        payload={
            "variance": var,
            "sd": math.sqrt(var),
            "gini_mean_difference": gmd,
            "skewness": G1,
            "excess_kurtosis": G2,
            "bimodality_coefficient": bc,
        }
    )


def esteban_ray_index(positions, shares=None, *, alpha: float = 1.0, k: float = 1.0) -> float:
    r"""Esteban and Ray (1994) polarization ``P = K sum_i sum_j pi_i^{1 + alpha} pi_j |y_i - y_j|``.

    ``positions`` are group positions ``y_i`` with population shares
    ``pi_i`` (equal by default, normalised to sum 1); ``alpha`` in ``(0,
    1.6]`` is the identification sensitivity (``alpha = 0`` gives half the
    Gini mean difference times two).

    References
    ----------
    Esteban, J.-M. and Ray, D. (1994). On the measurement of polarization.
    *Econometrica*, 62(4), 819-851.

    Examples
    --------
    >>> round(esteban_ray_index([0.0, 1.0], [0.5, 0.5], alpha=1.0), 6)
    0.25
    """
    y = _vec(positions)
    p = [1.0] * len(y) if shares is None else _vec(shares)
    s = ssum(p)
    p = [v / s for v in p]
    return k * ssum(p[i] ** (1 + alpha) * p[j] * abs(y[i] - y[j]) for i in range(len(y)) for j in range(len(y)))


def earth_movers_distance(a, b) -> float:
    r"""One-dimensional earth mover's (Wasserstein-1) distance ``int |F_a(x) - F_b(x)| dx`` between two samples.

    Examples
    --------
    >>> earth_movers_distance([0.0, 1.0], [2.0, 3.0])
    2.0
    """
    A, B = sorted(_vec(a)), sorted(_vec(b))
    pts = sorted(set(A) | set(B))
    na, nb = len(A), len(B)
    ia = ib = 0
    tot = 0.0
    for t in range(len(pts) - 1):
        while ia < na and A[ia] <= pts[t]:
            ia += 1
        while ib < nb and B[ib] <= pts[t]:
            ib += 1
        tot += abs(ia / na - ib / nb) * (pts[t + 1] - pts[t])
    return tot


def issue_constraint(X) -> RichResult:
    r"""Issue constraint (cross-dimensional polarization): mean absolute Pearson correlation between issue columns.

    ``X`` is respondents x issues; the pairwise correlation matrix and its
    mean absolute off-diagonal entry are returned (Converse 1964;
    Baldassarri and Gelman 2008).

    References
    ----------
    Baldassarri, D. and Gelman, A. (2008). Partisans without constraint:
    political polarization and trends in American public opinion. *American
    Journal of Sociology*, 114(2), 408-446.

    Examples
    --------
    >>> round(issue_constraint([[1, 2], [2, 4], [3, 5]]).constraint, 6)
    0.981981
    """
    M = [_vec(r) for r in X]
    p = len(M[0])
    cols = [[r[j] for r in M] for j in range(p)]

    def cor(a, b):
        ma, mb = ssum(a) / len(a), ssum(b) / len(b)
        sab = ssum((u - ma) * (v - mb) for u, v in zip(a, b))
        return sab / math.sqrt(ssum((u - ma) ** 2 for u in a) * ssum((v - mb) ** 2 for v in b))

    R = [[1.0 if i == j else cor(cols[i], cols[j]) for j in range(p)] for i in range(p)]
    off = [abs(R[i][j]) for i in range(p) for j in range(i + 1, p)]
    return RichResult(payload={"correlation": R, "constraint": ssum(off) / len(off)})


def polarization_trend(time, values) -> RichResult:
    r"""Linear trend of a polarization series: OLS slope, intercept, slope standard error and t statistic.

    Examples
    --------
    >>> r = polarization_trend([1, 2, 3, 4], [0.5, 0.6, 0.8, 0.9])
    >>> round(r.slope, 6), round(r.intercept, 6)
    (0.14, 0.35)
    """
    t, y = _vec(time), _vec(values)
    n = len(t)
    mt, my = ssum(t) / n, ssum(y) / n
    sxx = ssum((a - mt) ** 2 for a in t)
    b = ssum((a - mt) * (c - my) for a, c in zip(t, y)) / sxx
    a0 = my - b * mt
    rss = ssum((c - a0 - b * a) ** 2 for a, c in zip(t, y))
    se = math.sqrt(rss / (n - 2) / sxx) if n > 2 else float("nan")
    return RichResult(payload={"slope": b, "intercept": a0, "se": se, "t": b / se if se > 0 else float("inf")})


def party_system_indices(shares) -> RichResult:
    r"""Rae (1967) fractionalization ``1 - sum s_i^2``, Laakso-Taagepera effective number ``1/sum s_i^2`` and Golosov ``N_P``.

    Shares are normalised to sum 1. Golosov (2010): ``N_P = sum_i s_i /
    (s_i + s_1^2 - s_i^2)`` with ``s_1`` the largest share.

    References
    ----------
    Rae, D. W. (1967). *The Political Consequences of Electoral Laws*. Yale
    University Press.
    Golosov, G. V. (2010). The effective number of parties: a new approach.
    *Party Politics*, 16(2), 171-192.

    Examples
    --------
    >>> r = party_system_indices([50, 30, 20])
    >>> round(r.fractionalization, 6), round(r.effective_number, 6), round(r.golosov, 6)
    (0.62, 2.631579, 2.139979)
    """
    s = _vec(shares)
    t = ssum(s)
    s = [v / t for v in s]
    h = ssum(v * v for v in s)
    s1 = max(s)
    return RichResult(
        payload={
            "fractionalization": 1 - h,
            "effective_number": 1 / h,
            "golosov": ssum(v / (v + s1 * s1 - v * v) for v in s),
        }
    )


def party_divergence(positions, party, left, right, *, votes=None) -> RichResult:
    r"""Party divergence and overlap of ideal points (e.g. first-dimension NOMINATE scores).

    Distance between party means and between party medians (``right -
    left``); overlap = legislators lying between the most left-wing member of
    ``right`` and the most right-wing member of ``left`` (0 when the parties
    are separated); with ``votes`` (legislators x roll calls, 1 yea, 0 nay,
    ``nan`` missing) the mean Rice (1925) cohesion ``|yea - nay|/(yea +
    nay)`` of each party.

    References
    ----------
    McCarty, N., Poole, K. T. and Rosenthal, H. (2006). *Polarized America*.
    MIT Press.
    Rice, S. A. (1925). The behavior of legislative groups: a method of
    measurement. *Political Science Quarterly*, 40(1), 60-72.

    Examples
    --------
    >>> r = party_divergence([-0.8, -0.2, 0.1, 0.3, 0.9], ["D", "D", "D", "R", "R"], "D", "R")
    >>> round(r.mean_difference, 6), r.overlap
    (0.9, 0)
    """
    x = _vec(positions)
    g = list(party)
    L = [v for v, q in zip(x, g) if q == left]
    R = [v for v, q in zip(x, g) if q == right]

    def med(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    lo, hi = min(R), max(L)
    ov = sum(1 for v in x if lo <= v <= hi) if lo <= hi else 0
    out = {
        "mean_difference": ssum(R) / len(R) - ssum(L) / len(L),
        "median_difference": med(R) - med(L),
        "overlap": ov,
        "overlap_share": ov / len(x),
    }
    if votes is not None:
        V = [_vec(r) for r in votes]
        for name, key in ((left, "rice_left"), (right, "rice_right")):
            rows = [V[i] for i in range(len(V)) if g[i] == name]
            rc = []
            for j in range(len(V[0])):
                col = [r[j] for r in rows if r[j] == r[j]]
                if col:
                    yea = sum(1 for v in col if v == 1)
                    rc.append(abs(2 * yea - len(col)) / len(col))
            out[key] = ssum(rc) / len(rc)
    return RichResult(payload=out)


def rollcall_matrix(legislator, vote, choice, *, yea=(1, 2, 3), nay=(4, 5, 6)) -> RichResult:
    r"""Legislator x roll-call matrix from long records, coded 1 (yea), 0 (nay), ``nan`` (other).

    The default codes follow Voteview/``pscl::rollcall``: 1-3 yea, 4-6 nay,
    0 and 7-9 not voting or absent. Rows and columns are the sorted unique
    legislator and vote identifiers.

    References
    ----------
    Poole, K. T. and Rosenthal, H. (1997). *Congress: A Political-Economic
    History of Roll Call Voting*. Oxford University Press.

    Examples
    --------
    >>> r = rollcall_matrix(["a", "a", "b", "b"], [1, 2, 1, 2], [1, 6, 4, 9])
    >>> r.matrix
    [[1.0, 0.0], [0.0, nan]]
    """
    legs = sorted(set(legislator), key=str)
    vs = sorted(set(vote), key=str)
    li = {v: i for i, v in enumerate(legs)}
    vi = {v: i for i, v in enumerate(vs)}
    M = [[float("nan")] * len(vs) for _ in legs]
    Y, N = set(yea), set(nay)
    for a, b, c in zip(legislator, vote, choice):
        M[li[a]][vi[b]] = 1.0 if c in Y else 0.0 if c in N else float("nan")
    return RichResult(payload={"matrix": M, "legislators": legs, "votes": vs})


def rollcall_summary(matrix) -> RichResult:
    r"""Per-vote yeas, nays, minority share and margin; per-legislator participation.

    Examples
    --------
    >>> r = rollcall_summary([[1, 0], [1, 1], [0, None]])
    >>> r.yeas, r.nays, [round(v, 6) for v in r.minority_share]
    ([2, 1], [1, 1], [0.333333, 0.5])
    """
    M = [[float(v) if v is not None else float("nan") for v in r] for r in matrix]
    p = len(M[0])
    ye = [sum(1 for r in M if r[j] == 1) for j in range(p)]
    na = [sum(1 for r in M if r[j] == 0) for j in range(p)]
    ms = [min(a, b) / (a + b) if a + b else float("nan") for a, b in zip(ye, na)]
    return RichResult(
        payload={
            "yeas": ye,
            "nays": na,
            "minority_share": ms,
            "margin": [abs(a - b) for a, b in zip(ye, na)],
            "participation": [sum(1 for v in r if v == v) for r in M],
        }
    )


def rollcall_filter(matrix, *, lop: float = 0.025, minvotes: int = 20) -> RichResult:
    r"""Drop lopsided roll calls and low-participation legislators (``wnominate``/``pscl`` defaults).

    Votes whose minority share is below ``lop`` (or with no yea/nay) are
    dropped first; then legislators voting on fewer than ``minvotes`` of the
    retained roll calls. Returns the kept indices (0-based) and the filtered
    matrix.

    References
    ----------
    Poole, K., Lewis, J., Lo, J. and Carroll, R. (2011). Scaling roll call
    votes with wnominate in R. *Journal of Statistical Software*, 42(14), 1-21.

    Examples
    --------
    >>> r = rollcall_filter([[1, 1, 0], [1, 0, 1], [1, 1, 1]], lop=0.2, minvotes=1)
    >>> r.votes, r.legislators
    ([1, 2], [0, 1, 2])
    """
    M = [[float(v) if v is not None else float("nan") for v in r] for r in matrix]
    s = rollcall_summary(M)
    keep_v = [j for j, v in enumerate(s["minority_share"]) if v == v and v >= lop]
    keep_l = [i for i, r in enumerate(M) if sum(1 for j in keep_v if r[j] == r[j]) >= minvotes]
    return RichResult(
        payload={"votes": keep_v, "legislators": keep_l, "matrix": [[M[i][j] for j in keep_v] for i in keep_l]}
    )


def cheatsheet() -> str:
    return (
        "polarization_indices / esteban_ray_index / party_divergence / rollcall_filter -> polarization and roll calls."
    )
