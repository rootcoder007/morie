# morie.fn -- function file (rootcoder007/morie)
"""Space-time interaction (Knox, near-repeat), Mantel tests, geographic profiling, aoristic analysis, KDE, CCF and EOFs."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._sci_core import gammaincc

__all__ = [
    "knox_space_time",
    "near_repeat_table",
    "mantel_matrix_test",
    "geographic_profile",
    "aoristic_weights",
    "kde2d",
    "sample_cross_correlation",
    "eof_analysis",
]


def _perm(n, seed, stream):
    """Fisher-Yates permutation from Philox uniforms."""
    u = [float(v) for v in random_uniform(n, seed=seed, stream=stream)]
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(u[i] * (i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def _pairs_close(P, T, s0, t0, perm=None):
    n = len(P)
    idx = list(range(n)) if perm is None else perm
    x = ns = nt = 0
    for i in range(n):
        for j in range(i + 1, n):
            cs = math.dist(P[i], P[j]) <= s0
            ct = abs(T[idx[i]] - T[idx[j]]) <= t0
            ns += cs
            nt += ct
            x += cs and ct
    return x, ns, nt


def knox_space_time(coords, times, s0: float, t0: float, *, nsim: int = 999, seed: int = 1) -> RichResult:
    r"""Knox (1964) test of space-time interaction.

    ``X`` is the number of event pairs within distance ``s0`` and time
    ``t0``; its expectation under independence is ``N_s N_t / N`` (pairs
    close in space, close in time, all pairs).  The Poisson p-value is
    ``P(Poisson(E) >= X)``; the Monte Carlo p-value permutes the event times
    (Philox streams ``1..nsim``), ``(1 + #{X_sim >= X}) / (1 + nsim)``.

    References
    ----------
    Knox, E. G. (1964). The detection of space-time interactions. *Journal
    of the Royal Statistical Society C*, 13(1), 25-30.

    Examples
    --------
    >>> k = knox_space_time([(0, 0), (0.5, 0), (5, 5), (5.2, 5)], [1, 2, 10, 11], 1.0, 2.0, nsim=0)
    >>> k.observed, round(k.expected, 6)
    (2, 0.666667)
    """
    P = [tuple(float(v) for v in r) for r in np.asarray(coords, dtype=float).tolist()]
    T = [float(v) for v in times]
    n = len(P)
    X, ns, nt = _pairs_close(P, T, s0, t0)
    N = n * (n - 1) / 2
    E = ns * nt / N
    out = {
        "observed": X,
        "expected": E,
        "n_space": ns,
        "n_time": nt,
        "poisson_p": float(1.0 - gammaincc(X, E)) if X > 0 else 1.0,
    }
    if nsim > 0:
        sims = [_pairs_close(P, T, s0, t0, _perm(n, seed, s))[0] for s in range(1, nsim + 1)]
        out["mc_p"] = (1 + sum(1 for v in sims if v >= X)) / (nsim + 1)
    return RichResult(payload=out)


def near_repeat_table(coords, times, space_breaks, time_breaks, *, nsim: int = 99, seed: int = 1) -> RichResult:
    r"""Near-repeat analysis (Townsley, Homel and Chaseling 2003; Ratcliffe's calculator).

    Event pairs are tabulated by spatial band ``(s_{k-1}, s_k]`` (the first
    band includes 0) and temporal band; the expected table is the mean over
    ``nsim`` Philox permutations of the times; ``ratio`` is observed over
    expected and ``p`` the one-sided Monte Carlo p-value of each cell.

    References
    ----------
    Townsley, M., Homel, R. and Chaseling, J. (2003). Infectious burglaries:
    a test of the near repeat hypothesis. *British Journal of Criminology*,
    43(3), 615-633.

    Examples
    --------
    >>> t = near_repeat_table([(0, 0), (0.5, 0), (5, 5), (5.2, 5)], [1, 2, 10, 11], [1, 10], [2, 20], nsim=5)
    >>> t.observed
    [[2, 0], [0, 4]]
    """
    P = [tuple(float(v) for v in r) for r in np.asarray(coords, dtype=float).tolist()]
    T = [float(v) for v in times]
    n = len(P)
    sb, tb = [float(v) for v in space_breaks], [float(v) for v in time_breaks]

    def band(v, br):
        for k, b in enumerate(br):
            if v <= b:
                return k
        return None

    def table(perm):
        M = [[0] * len(tb) for _ in sb]
        for i in range(n):
            for j in range(i + 1, n):
                a = band(math.dist(P[i], P[j]), sb)
                b = band(abs(T[perm[i]] - T[perm[j]]), tb)
                if a is not None and b is not None:
                    M[a][b] += 1
        return M

    obs = table(list(range(n)))
    sims = [table(_perm(n, seed, s)) for s in range(1, nsim + 1)]
    exp = [[ssum(S[a][b] for S in sims) / nsim for b in range(len(tb))] for a in range(len(sb))] if nsim else None
    pv = (
        [
            [(1 + sum(1 for S in sims if S[a][b] >= obs[a][b])) / (nsim + 1) for b in range(len(tb))]
            for a in range(len(sb))
        ]
        if nsim
        else None
    )
    ratio = (
        [
            [obs[a][b] / exp[a][b] if exp and exp[a][b] > 0 else float("nan") for b in range(len(tb))]
            for a in range(len(sb))
        ]
        if nsim
        else None
    )
    return RichResult(payload={"observed": obs, "expected": exp, "ratio": ratio, "p": pv})


def mantel_matrix_test(D1, D2, *, method: str = "pearson", nsim: int = 999, seed: int = 1) -> RichResult:
    r"""Mantel (1967) test, statistic as ``vegan::mantel``: the correlation of the two lower triangles.

    ``method`` is ``pearson`` or ``spearman`` (average ranks); the p-value
    permutes the rows and columns of ``D1`` jointly (Philox streams
    ``1..nsim``), ``(1 + #{r_sim >= r}) / (1 + nsim)``.

    References
    ----------
    Mantel, N. (1967). The detection of disease clustering and a generalized
    regression approach. *Cancer Research*, 27(2), 209-220.

    Examples
    --------
    >>> D = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    >>> round(mantel_matrix_test(D, D, nsim=0).statistic, 12)
    1.0
    """
    A = [[float(v) for v in r] for r in D1]
    B = [[float(v) for v in r] for r in D2]
    n = len(A)

    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    def cor(x, y):
        if method == "spearman":
            x, y = rank(x), rank(y)
        mx, my = ssum(x) / len(x), ssum(y) / len(y)
        return ssum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
            ssum((a - mx) ** 2 for a in x) * ssum((b - my) ** 2 for b in y)
        )

    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    idx = [(i, j) for j in range(n) for i in range(j + 1, n)]
    b = [B[i][j] for i, j in idx]
    r = cor([A[i][j] for i, j in idx], b)
    out = {"statistic": r}
    if nsim > 0:
        sims = []
        for s in range(1, nsim + 1):
            p = _perm(n, seed, s)
            sims.append(cor([A[p[i]][p[j]] for i, j in idx], b))
        out["pvalue"] = (1 + sum(1 for v in sims if v >= r)) / (nsim + 1)
    return RichResult(payload=out)


def geographic_profile(crimes, grid, *, B: float, f: float = 1.2, g: float = 1.2) -> list:
    r"""Rossmo's (1995) criminal geographic targeting score on grid points.

    ``P(x) = sum_n [phi_n / d_n^f + (1 - phi_n) B^{g - f} / (2B - d_n)^g]``
    with Manhattan distances ``d_n`` to the crime sites and ``phi_n = 1``
    outside the buffer (``d_n > B``) and 0 inside; the scores are
    normalised to sum to one over the grid.

    References
    ----------
    Rossmo, D. K. (1995). *Geographic Profiling: Target Patterns of Serial
    Murderers*. PhD thesis, Simon Fraser University.

    Examples
    --------
    >>> s = geographic_profile([(0, 0), (2, 0)], [(1, 0), (5, 5)], B=1.5)
    >>> [round(v, 6) for v in s]
    [0.856744, 0.143256]
    """
    C = [tuple(float(v) for v in r) for r in np.asarray(crimes, dtype=float).tolist()]
    G = [tuple(float(v) for v in r) for r in np.asarray(grid, dtype=float).tolist()]
    raw = []
    for q in G:
        s = 0.0
        for c in C:
            d = abs(q[0] - c[0]) + abs(q[1] - c[1])
            if d > B:
                s += 1.0 / d**f
            else:
                s += B ** (g - f) / (2 * B - d) ** g
        raw.append(s)
    tot = ssum(raw)
    return [v / tot for v in raw]


def aoristic_weights(starts, ends, breaks) -> RichResult:
    r"""Aoristic analysis (Ratcliffe 2002): spread each event's time window over the bins it overlaps.

    Event ``[s, e)`` contributes ``overlap / (e - s)`` to each bin ``[b_k,
    b_{k+1})``; an instantaneous event (``e = s``) counts 1 in the bin
    containing ``s``.  Returns the per-event weights and the bin totals.

    References
    ----------
    Ratcliffe, J. H. (2002). Aoristic signatures and the spatio-temporal
    analysis of high volume crime patterns. *Journal of Quantitative
    Criminology*, 18(1), 23-43.

    Examples
    --------
    >>> aoristic_weights([0.5, 2.0], [2.5, 2.0], [0, 1, 2, 3]).totals
    [0.25, 0.5, 1.25]
    """
    br = [float(v) for v in breaks]
    W = []
    for s, e in zip(starts, ends):
        s, e = float(s), float(e)
        row = [0.0] * (len(br) - 1)
        if e == s:
            for k in range(len(br) - 1):
                if br[k] <= s < br[k + 1]:
                    row[k] = 1.0
        else:
            for k in range(len(br) - 1):
                row[k] = max(0.0, min(e, br[k + 1]) - max(s, br[k])) / (e - s)
        W.append(row)
    return RichResult(
        payload={"weights": W, "totals": [ssum(W[i][k] for i in range(len(W))) for k in range(len(br) - 1)]}
    )


def _bw_nrd(x):
    s = sorted(x)
    n = len(s)

    def q(p):
        h = (n - 1) * p
        lo = int(math.floor(h))
        return s[lo] + (h - lo) * (s[min(lo + 1, n - 1)] - s[lo])

    m = ssum(s) / n
    sd = math.sqrt(ssum((v - m) ** 2 for v in s) / (n - 1))
    return 4 * 1.06 * min(sd, (q(0.75) - q(0.25)) / 1.34) * n ** (-0.2)


def kde2d(x, y, *, h=None, n: int = 25, lims=None) -> RichResult:
    r"""Bivariate Gaussian kernel density on a grid, as ``MASS::kde2d``.

    Bandwidths default to ``bandwidth.nrd`` (``4 * 1.06 min(sd, IQR/1.34)
    n^{-1/5}``) per axis and, as in MASS, are divided by 4 before use as the
    normal kernel standard deviations; the grid spans ``lims`` (default the
    data ranges) with ``n`` points per axis.

    References
    ----------
    Venables, W. N. and Ripley, B. D. (2002). *Modern Applied Statistics
    with S*, 4th edn. Springer, New York.

    Examples
    --------
    >>> k = kde2d([0.0, 1.0, 2.0], [0.0, 1.0, 0.5], h=[1.0, 1.0], n=3)
    >>> round(k.z[1][1], 6)
    0.115199
    """
    xs, ys = [float(v) for v in x], [float(v) for v in y]
    nx = len(xs)
    hh = (
        [_bw_nrd(xs), _bw_nrd(ys)] if h is None else [float(v) for v in (h if isinstance(h, (list, tuple)) else [h, h])]
    )
    hh = [v / 4 for v in hh]
    L = [min(xs), max(xs), min(ys), max(ys)] if lims is None else [float(v) for v in lims]
    gx = [L[0] + (L[1] - L[0]) * k / (n - 1) for k in range(n)]
    gy = [L[2] + (L[3] - L[2]) * k / (n - 1) for k in range(n)]

    def dn(u):
        return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)

    z = [
        [ssum(dn((a - xs[i]) / hh[0]) * dn((b - ys[i]) / hh[1]) for i in range(nx)) / (nx * hh[0] * hh[1]) for b in gy]
        for a in gx
    ]
    return RichResult(payload={"x": gx, "y": gy, "z": z})


def sample_cross_correlation(x, y, lag_max: int | None = None) -> RichResult:
    r"""Sample cross-correlation ``r_xy(k) = c_xy(k) / sqrt(c_xx(0) c_yy(0))``, ``c_xy(k) = (1/n) sum (x_{t+k} - xbar)(y_t - ybar)``, as ``stats::ccf``.

    Lags run from ``-lag_max`` to ``lag_max`` (default ``floor(10 log10
    n)``).

    Examples
    --------
    >>> r = sample_cross_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1)
    >>> [round(v, 6) for v in r.acf]
    [0.25, 1.0, 0.25]
    """
    a, b = [float(v) for v in x], [float(v) for v in y]
    n = len(a)
    K = int(math.floor(10 * math.log10(n))) if lag_max is None else int(lag_max)
    ma, mb = ssum(a) / n, ssum(b) / n
    sa = math.sqrt(ssum((v - ma) ** 2 for v in a) / n)
    sb = math.sqrt(ssum((v - mb) ** 2 for v in b) / n)
    lags = list(range(-K, K + 1))
    r = []
    for k in lags:
        c = ssum((a[t + k] - ma) * (b[t] - mb) for t in range(max(0, -k), min(n, n - k))) / n
        r.append(c / (sa * sb))
    return RichResult(payload={"lag": lags, "acf": r})


def eof_analysis(Z, *, k: int | None = None) -> RichResult:
    r"""Empirical orthogonal functions of a space-time field ``Z`` (times x locations).

    Each location's temporal mean is removed and the anomaly matrix
    decomposed ``U S V'``: EOF ``j`` is ``V[:, j]`` (sign fixed so its
    largest absolute loading is positive), its principal component time
    series ``U[:, j] s_j`` and its explained variance ``s_j^2 / sum s^2`` (as
    ``prcomp`` on the centred field; Lorenz 1956; von Storch and Zwiers
    1999).

    References
    ----------
    von Storch, H. and Zwiers, F. W. (1999). *Statistical Analysis in
    Climate Research*. Cambridge University Press.

    Examples
    --------
    >>> e = eof_analysis([[1, 2], [2, 4], [3, 6]])
    >>> [round(v, 6) for v in e.explained]
    [1.0, 0.0]
    """
    M = [[float(v) for v in r] for r in np.asarray(Z, dtype=float).tolist()]
    nt, ns = len(M), len(M[0])
    mu = [ssum(M[t][s] for t in range(nt)) / nt for s in range(ns)]
    A = [[M[t][s] - mu[s] for s in range(ns)] for t in range(nt)]
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    U, S, Vt = U.tolist(), [float(v) for v in S.tolist()], Vt.tolist()
    kk = len(S) if k is None else min(int(k), len(S))
    for j in range(kk):
        m = max(range(ns), key=lambda s: abs(Vt[j][s]))
        if Vt[j][m] < 0:
            Vt[j] = [-v for v in Vt[j]]
            for t in range(nt):
                U[t][j] = -U[t][j]
    tot = ssum(s * s for s in S)
    return RichResult(
        payload={
            "eofs": [[Vt[j][s] for j in range(kk)] for s in range(ns)],
            "pcs": [[U[t][j] * S[j] for j in range(kk)] for t in range(nt)],
            "explained": [S[j] ** 2 / tot for j in range(kk)],
            "sdev": [s / math.sqrt(nt - 1) for s in S[:kk]],
        }
    )


def cheatsheet() -> str:
    return "knox_space_time / near_repeat_table / mantel_matrix_test / geographic_profile / kde2d / eof_analysis -> space-time tools."

# alias kept from the retired placeholder of the same name
empirical_orthogonal_func = eof_analysis
