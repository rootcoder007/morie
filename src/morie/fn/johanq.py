"""Johansen's heteroscedastic test of a general linear hypothesis about trimmed means.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eqs (10.3)-(10.4).
"""

from ._richresult import RichResult
from ._rrng_core import qchisq
from .trimse import _tmean
from .yuen import _yuen_d

__all__ = ["johanq"]


def _inv(M):
    n = len(M)
    A = [M[i][:] + [float(i == j) for j in range(n)] for i in range(n)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[piv][k]) < 1e-300:
            raise ValueError("C V C' is singular")
        A[k], A[piv] = A[piv], A[k]
        f = A[k][k]
        A[k] = [v / f for v in A[k]]
        for i in range(n):
            if i != k and A[i][k] != 0:
                g = A[i][k]
                A[i] = [a - g * b for a, b in zip(A[i], A[k])]
    return [row[n:] for row in A]


def johanq(groups, C, tr=0.2, alpha=0.05):
    """Test H0: C mu_t = 0 (10.3) with Q = Xbar' C' (C V C')^{-1} C Xbar (10.4).

    V = diag(d_j), Yuen's squared standard errors of the trimmed means. The
    critical value is Johansen's adjusted chi-squared,
    c = chi2_{1-alpha,k} + (chi2/2k) A (1 + 3 chi2/(k + 2)) with k = rank(C)
    rows and A = sum_j R_jj^2/(h_j - 1), R = V C' (C V C')^{-1} C. The p-value
    is the alpha at which c equals Q (solved by bisection).

    Parameters
    ----------
    groups : list of J sequences
    C : k x J nested sequence of contrast coefficients
    tr, alpha : float

    Returns
    -------
    RichResult
        Keys: statistic (Q), crit, reject, p_value, A.

    References
    ----------
    Johansen, S. (1980). Biometrika 67, 85-92.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eqs (10.3)-(10.4).

    Examples
    --------
    >>> r = johanq([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, -1]])
    >>> r["statistic"]
    0.0
    """
    gs = [[float(v) for v in g] for g in groups]
    C = [[float(v) for v in row] for row in C]
    J = len(gs)
    k = len(C)
    if J < 2 or k < 1 or any(len(r) != J for r in C):
        raise ValueError("C must have one column per group")
    xb = [_tmean(g, tr) for g in gs]
    dh = [_yuen_d(g, tr) for g in gs]
    v = [d for d, _ in dh]
    h = [hh for _, hh in dh]
    if min(h) < 2:
        raise ValueError("too few observations left after trimming")
    CVC = [[sum(C[a][j] * v[j] * C[b][j] for j in range(J)) for b in range(k)] for a in range(k)]
    inv = _inv(CVC)
    cx = [sum(C[a][j] * xb[j] for j in range(J)) for a in range(k)]
    Q = sum(cx[a] * inv[a][b] * cx[b] for a in range(k) for b in range(k))
    Rjj = [v[j] * sum(C[a][j] * inv[a][b] * C[b][j] for a in range(k) for b in range(k)) for j in range(J)]
    A = sum(r * r / (hh - 1) for r, hh in zip(Rjj, h))

    def crit(al):
        c = qchisq(1 - al, k)
        return c + c / (2 * k) * A * (1 + 3 * c / (k + 2))

    cr = crit(alpha)
    lo, hi = 1e-12, 1 - 1e-12
    if crit(hi) >= Q:
        p = 1.0
    elif crit(lo) <= Q:
        p = 0.0
    else:
        for _ in range(200):
            mid = (lo + hi) / 2
            if crit(mid) > Q:
                lo = mid
            else:
                hi = mid
        p = (lo + hi) / 2
    return RichResult(
        title="Johansen test for trimmed means",
        summary_lines=[("Q", Q), ("crit", cr), ("p", p)],
        payload={"statistic": Q, "crit": cr, "reject": cr < Q, "p_value": p, "A": A},
    )


def cheatsheet():
    return "johanq: Johansen Q = Xbar'C'(CVC')^-1 C Xbar for trimmed means with adjusted critical value. Wilcox (2017) eq (10.4)."
