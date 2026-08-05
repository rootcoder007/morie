# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Rank of a matrix and the rank calculation rules.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 2.4.5, printed page 50 (PDF page 84), equation (2.46)::

    rg(A)  = rg(A')
    rg(AB) <= min{rg(A), rg(B)}                                   (2.46)
    rg(A'A) = rg(AA') = rg(A)

with (2.44) ``rg(A_(n x m)) <= min(n, m)``, equality meaning full rank,
and (2.45) a square matrix of full rank is regular, i.e. ``|A| != 0`` and
``A^-1`` exists.

The rank itself is computed by Gaussian elimination with partial
pivoting and a relative tolerance, the same algorithm in both arms so
the integer answers cannot drift.
"""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mrank"]


def _rows(a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    return [[float(a[i][j]) for j in range(len(a[0]))] for i in range(len(a))]


def _rank(rows, tol=None):
    """Rank by Gaussian elimination with partial pivoting."""
    m = [r[:] for r in rows]
    nr = len(m)
    nc = len(m[0]) if nr else 0
    if nr == 0 or nc == 0:
        return 0, 0.0
    big = max(abs(v) for r in m for v in r)
    if tol is None:
        tol = max(nr, nc) * 2.220446049250313e-16 * (big if big > 0.0 else 1.0)
    r = 0
    for c in range(nc):
        if r >= nr:
            break
        piv = max(range(r, nr), key=lambda i: abs(m[i][c]))
        if abs(m[piv][c]) <= tol:
            continue
        m[r], m[piv] = m[piv], m[r]
        pv = m[r][c]
        for i in range(r + 1, nr):
            f = m[i][c] / pv
            if f != 0.0:
                for j in range(c, nc):
                    m[i][j] -= f * m[r][j]
        r += 1
    return r, tol


def _t(rows):
    return [[rows[i][j] for i in range(len(rows))] for j in range(len(rows[0]))]


def _mm(a, b):
    nb = len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(nb)] for i in range(len(a))]


def mrank(A, B=None):
    """Rank of ``A``, with the (2.46) rank rules evaluated.

    Parameters
    ----------
    A : 2-D array-like
        The matrix whose rank is wanted.
    B : 2-D array-like, optional
        A second matrix, conformable as ``A B``.  When given, the product
        rank and the bound ``min{rg(A), rg(B)}`` are reported so the
        inequality in (2.46) can be checked directly.

    Returns
    -------
    RichResult
        Keys: ``rank``, ``nrow``, ``ncol``, ``max_rank``, ``full_rank``,
        ``regular``, ``rank_t``, ``rank_gram``, ``rank_gram_outer``, and
        with ``B`` also ``rank_b``, ``rank_prod``, ``bound``,
        ``bound_holds``.
    """
    a = _rows(A)
    if len(a) == 0 or len(a[0]) == 0:
        raise ValueError("A must have at least one row and one column")
    for r in a:
        for v in r:
            if not (v == v) or v in (float("inf"), float("-inf")):
                raise ValueError("A must be finite")
    nr, nc = len(a), len(a[0])
    for r in a:
        if len(r) != nc:
            raise ValueError("A must be rectangular")
    rk, _tol = _rank(a)
    at = _t(a)
    payload = {
        "rank": rk,
        "nrow": nr,
        "ncol": nc,
        "max_rank": min(nr, nc),
        "full_rank": rk == min(nr, nc),
        "regular": bool(nr == nc and rk == nr),
        "rank_t": _rank(at)[0],
        "rank_gram": _rank(_mm(at, a))[0],
        "rank_gram_outer": _rank(_mm(a, at))[0],
    }
    summary = [
        ("rank", rk),
        ("dim", (nr, nc)),
        ("full rank", payload["full_rank"]),
        ("regular", payload["regular"]),
    ]
    if B is not None:
        b = _rows(B)
        if len(b) != nc:
            raise ValueError("A and B are not conformable for the product A B")
        rb = _rank(b)[0]
        rp = _rank(_mm(a, b))[0]
        bound = min(rk, rb)
        payload.update(
            {"rank_b": rb, "rank_prod": rp, "bound": bound, "bound_holds": rp <= bound}
        )
        summary.append(("rg(AB) <= min{rg A, rg B}", (rp, bound)))
    return RichResult(
        title="Matrix rank and rank rules (Hedderich eq. 2.44-2.46)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet() -> str:
    return "mrank(A, B=None): rank of A plus the (2.46) rank rules -- Hedderich eq. (2.46)."
