# morie.fn -- function file (rootcoder007/morie)
"""Resultant of two polynomials via the Sylvester matrix.

Classical algebra.  Triage confirmed this names no owning source.

Coefficient order: lowest degree first, matching
:mod:`morie.fn.frmlD`.  The determinant is taken with the
Bareiss fraction-free algorithm, which stays exact on integer
coefficients instead of drifting the way an LU factorization would.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["resultant"]


def _trim(c):
    """Drop trailing zero coefficients so the degree is honest."""
    a = [float(v) for v in c]
    while len(a) > 1 and a[-1] == 0.0:
        a.pop()
    return a


def _bareiss(M):
    """Fraction-free Gaussian elimination.  Every intermediate entry
    stays an exact divisor, so an integer matrix yields an integer
    determinant with no rounding."""
    A = [row[:] for row in M]
    n = len(A)
    if n == 0:
        return 1.0
    sign = 1
    prev = 1.0
    for k in range(n - 1):
        if A[k][k] == 0.0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0.0), None)
            if sw is None:
                return 0.0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) / prev
            A[i][k] = 0.0
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def resultant(p, q):
    """Resultant of p and q, the determinant of their Sylvester matrix.

    The resultant vanishes exactly when the two polynomials share a
    root, which is what makes it the standard tool for eliminating a
    variable between two equations without ever finding the roots.

    The Sylvester matrix is (m + n) square for degrees m and n: n rows
    of the coefficients of p shifted one place each time, then m rows
    of the coefficients of q, both written highest degree first.

    Parameters
    ----------
    p, q : coefficient sequences, lowest degree first.

    Returns
    -------
    RichResult with keys estimate (the resultant), resultant,
    sylvester, deg_p, deg_q, share_root, method.
    """
    a = _trim(p)
    b = _trim(q)
    m = len(a) - 1
    n = len(b) - 1
    if m < 1 and n < 1:
        raise ValueError("at least one polynomial must be non-constant")
    ah = a[::-1]
    bh = b[::-1]
    size = m + n
    S = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j, v in enumerate(ah):
            S[i][i + j] = v
    for i in range(m):
        for j, v in enumerate(bh):
            S[n + i][i + j] = v
    res = _bareiss(S)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(res), "resultant": float(res),
        "sylvester": S, "deg_p": m, "deg_q": n,
        "share_root": abs(res) < 1e-12,
        "method": "resultant via the Sylvester matrix (Bareiss)",
    }), "resaln")


def cheatsheet():
    return "resaln: Resultant of two polynomials"


# compact alias per ledger/NAMING.md
resultpoly = resultant
