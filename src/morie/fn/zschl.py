"""Cholesky (LU) simulation of Gaussian random fields: unconditional, conditional, precision-based and pivoted."""

from __future__ import annotations

import math

from . import _array_core as np
from ._mvcore import cholesky
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_normal
from .krgsys import _dist, _rows, kriging_covariance

__all__ = ["chol_sim", "pivoted_cholesky"]


def pivoted_cholesky(A, *, tol: float = 1e-12, max_rank: int | None = None) -> RichResult:
    r"""Pivoted (incomplete) Cholesky factorisation ``A ~ L L'`` of a positive semi-definite matrix.

    At each step the largest remaining diagonal element of the Schur
    complement is taken as pivot (ties: lowest index); the factorisation
    stops when the trace of the remainder falls to ``tol`` times the trace
    of ``A`` or at ``max_rank`` columns (Harbrecht, Peters and Schneider
    2012).  ``L`` is ``n x r`` in the original row order.

    References
    ----------
    Harbrecht, H., Peters, M. and Schneider, R. (2012). On the low-rank
    approximation by the pivoted Cholesky decomposition. *Applied Numerical
    Mathematics*, 62(4), 428-440.

    Examples
    --------
    >>> r = pivoted_cholesky([[4.0, 2.0], [2.0, 1.0]])
    >>> r.rank, r.pivots, [round(v[0], 6) for v in r.L]
    (1, [0], [2.0, 1.0])
    """
    M = [[float(v) for v in r] for r in np.asarray(A, dtype=float).tolist()]
    n = len(M)
    d = [M[i][i] for i in range(n)]
    tr0 = ssum(d)
    kmax = n if max_rank is None else min(n, int(max_rank))
    cols, piv = [], []
    while len(cols) < kmax and ssum(d[i] for i in range(n) if i not in piv) > tol * tr0:
        p = max((i for i in range(n) if i not in piv), key=lambda i: (d[i], -i))
        if d[p] <= 0:
            break
        s = math.sqrt(d[p])
        c = [0.0] * n
        for i in range(n):
            if i in piv:
                continue
            c[i] = (M[i][p] - ssum(col[i] * col[p] for col in cols)) / s
        c[p] = s
        cols.append(c)
        piv.append(p)
        for i in range(n):
            if i not in piv:
                d[i] -= c[i] * c[i]
    L = [[col[i] for col in cols] for i in range(n)]
    return RichResult(payload={"L": L, "pivots": piv, "rank": len(cols)})


def chol_sim(
    coords,
    model,
    *,
    nsim: int = 1,
    seed: int = 1,
    mean: float = 0.0,
    z=None,
    data_coords=None,
    method: str = "cholesky",
    tol: float = 1e-12,
) -> RichResult:
    r"""Gaussian random field simulation by the Cholesky (LU) method.

    With the covariance matrix ``C`` of the field at ``coords`` (a gstat
    ``vgm``-style model, :func:`morie.fn.krgsys.kriging_covariance`) and
    ``C = L L'``, each realisation is ``mean + L e`` with ``e`` standard
    normal (Davis 1987).  Conditioning on data ``z`` at ``data_coords``
    uses the exact Gaussian conditional distribution: mean ``mean +
    C_{xd} C_{dd}^{-1} (z - mean)`` (the simple kriging predictor) and
    covariance ``C_{xx} - C_{xd} C_{dd}^{-1} C_{dx}`` (the simple kriging
    error covariance), factorised by pivoted Cholesky because it is
    singular at data locations.

    ``method``: ``cholesky`` (``L`` of ``C``); ``precision`` (``Q = C^{-1}
    = R R'`` and ``x = R^{-T} e``, Rue and Held 2005, algorithm 2.4 --
    the same law with different draws); ``pivoted`` (:func:`pivoted_cholesky`
    to relative trace ``tol``, for near-singular models such as the
    Gaussian).  Realisation ``s`` uses Philox stream ``s`` of ``seed``.

    :param coords: (n, d) simulation locations.
    :param model: Covariance model (dict or list of nested dicts).
    :param nsim: Number of realisations.
    :param seed: Philox seed.
    :param mean: Known constant mean.
    :param z: Conditioning data (optional).
    :param data_coords: Locations of ``z``.
    :param method: ``cholesky``, ``precision`` or ``pivoted``.
    :param tol: Relative trace tolerance of the pivoted factorisation.
    :return: :class:`RichResult` with ``simulations`` (nsim x n),
        ``mean`` and ``cov`` (the conditional ones when conditioning).

    References
    ----------
    Davis, M. W. (1987). Production of conditional simulations via the LU
    triangular decomposition of the covariance matrix. *Mathematical
    Geology*, 19(2), 91-98.
    Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields*. Chapman
    and Hall/CRC, Boca Raton.

    Examples
    --------
    >>> r = chol_sim([(0.0, 0.0), (1.0, 0.0)], {"model": "Exp", "psill": 1.0, "range": 1.0}, seed=3)
    >>> [round(v, 6) for v in r.simulations[0]]
    [0.902691, -0.775404]
    """
    P = _rows(coords)
    n = len(P)
    if method not in ("cholesky", "precision", "pivoted"):
        raise ValueError("method must be cholesky, precision or pivoted")
    C = [[kriging_covariance(_dist(P[i], P[j]), model) for j in range(n)] for i in range(n)]
    mu = [float(mean)] * n
    cond = z is not None
    if cond:
        if data_coords is None:
            raise ValueError("data_coords is required with z")
        D = _rows(data_coords)
        zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
        if len(D) != len(zv):
            raise ValueError("data_coords must match z")
        m = len(D)
        Cdd = [[kriging_covariance(_dist(D[i], D[j]), model) for j in range(m)] for i in range(m)]
        Cxd = [[kriging_covariance(_dist(P[i], D[j]), model) for j in range(m)] for i in range(n)]
        Ci = [[float(v) for v in r] for r in inverse(Cdd)]
        W = [[ssum(Cxd[i][k] * Ci[k][j] for k in range(m)) for j in range(m)] for i in range(n)]
        mu = [float(mean) + ssum(W[i][j] * (zv[j] - float(mean)) for j in range(m)) for i in range(n)]
        C = [[C[i][j] - ssum(W[i][k] * Cxd[j][k] for k in range(m)) for j in range(n)] for i in range(n)]
        if method == "cholesky":
            method = "pivoted"
    if method == "cholesky":
        L = cholesky(C)
    elif method == "pivoted":
        L = pivoted_cholesky(C, tol=tol)["L"]
    else:
        R = cholesky([[float(v) for v in r] for r in inverse(C)])
    sims = []
    for s in range(int(nsim)):
        if method == "precision":
            e = [float(v) for v in random_normal(n, seed=seed, stream=s)]
            x = [0.0] * n
            for i in range(n - 1, -1, -1):  # solve R' x = e (R lower, R' upper)
                x[i] = (e[i] - ssum(R[k][i] * x[k] for k in range(i + 1, n))) / R[i][i]
        else:
            r = len(L[0]) if L and L[0] else 0
            e = [float(v) for v in random_normal(r, seed=seed, stream=s)] if r else []
            x = [ssum(L[i][k] * e[k] for k in range(r)) for i in range(n)]
        sims.append([mu[i] + x[i] for i in range(n)])
    return RichResult(payload={"simulations": sims, "mean": mu, "cov": C})


chol = chol_sim


def cheatsheet() -> str:
    return "chol_sim(coords, model) -> Cholesky (LU) Gaussian random field simulation, conditional or not."


# compact alias per ledger/NAMING.md
cholsim = chol_sim
