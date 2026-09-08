# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hermite polynomial basis.

Source consulted: Hermite, C. (1864). Sur un nouveau developpement en serie
des fonctions.  *Comptes Rendus de l'Academie des Sciences* 58, 93-100 and
266-273.  The physicists' polynomials satisfy the Rodrigues formula

    H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}

equivalently the three-term recurrence

    H_0 = 1,  H_1 = 2x,  H_{n+1}(x) = 2 x H_n(x) - 2 n H_{n-1}(x)

which is what is evaluated here.  The probabilists' polynomials
He_n(x) = 2^{-n/2} H_n(x / sqrt 2), orthogonal with respect to the standard
normal density, are available through ``kind="probabilist"``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hermite_basis"]


def hermite_basis(x, K=3, kind="physicist"):
    """Evaluate the Hermite basis up to degree ``K``.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    K : int
        Highest degree; the basis has ``K + 1`` columns.
    kind : {"physicist", "probabilist"}
        Which family to evaluate.

    Returns
    -------
    RichResult
        estimate (mean of the highest-degree column), basis, top, K, n, method.

    References
    ----------
    Hermite (1864), Comptes Rendus Acad. Sci. 58, 93-100, 266-273.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = int(xs.size)
    kk = int(K)
    cols = []
    for i in range(n):
        xi = float(xs[i])
        if kind == "probabilist":
            row = [1.0]
            if kk >= 1:
                row.append(xi)
            for m in range(1, kk):
                row.append(xi * row[m] - float(m) * row[m - 1])
        else:
            row = [1.0]
            if kk >= 1:
                row.append(2.0 * xi)
            for m in range(1, kk):
                row.append(2.0 * xi * row[m] - 2.0 * float(m) * row[m - 1])
        cols.append(row)
    basis = np.asarray(cols, dtype=float)
    top = np.asarray([cols[i][kk] for i in range(n)], dtype=float)
    return RichResult(
        payload={
            "estimate": float(np.mean(top)),
            "basis": basis,
            "top": float(top[n - 1]),
            "K": kk,
            "kind": kind,
            "n": n,
            "method": "Hermite polynomial basis (Hermite 1864)",
        }
    )


# CANONICAL TEST
# >>> # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
# >>> r = hermite_basis([2.0], K=3)
# >>> b = r["basis"]
# >>> assert abs(float(b[0, 0]) - 1.0) < 1e-12
# >>> assert abs(float(b[0, 1]) - 4.0) < 1e-12
# >>> assert abs(float(b[0, 2]) - 14.0) < 1e-12
# >>> assert abs(float(b[0, 3]) - 40.0) < 1e-12


def cheatsheet():
    return "hermitS(x, K, kind): Hermite polynomial basis by recurrence."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
hermitebasis = hermite_basis
