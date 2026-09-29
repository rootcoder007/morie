# morie.fn -- function file (rootcoder007/morie)
"""Probability kriging (Sullivan 1984): ordinary cokriging of the indicator of a threshold with the
uniform (rank) transform of the data, estimating the local probability of not exceeding it."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve
from ._richresult import RichResult

__all__ = ["probability_kriging"]


def _mat(X):
    return [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]


def _cov(h, model):
    nug, sill, rng = model
    return (nug if h == 0 else 0.0) + sill * math.exp(-h / rng)


def probability_kriging(coords, values, targets, threshold, cov_i, cov_u, cov_iu):
    r"""Probability kriging of ``P(Z(x) <= z_k)`` from the indicator and the uniform transform.

    With ``I_a = 1(z_a <= z_k)`` and ``U_a = (rank(z_a) - 1/2)/n``, the estimate
    ``sum a_a I_a + sum b_a U_a`` minimises the error variance subject to
    ``sum a_a = 1`` and ``sum b_a = 0`` (ordinary cokriging). The direct and
    cross covariances are exponential ``c0 1(h = 0) + c1 exp(-h / a)``, each
    given as ``(c0, c1, a)`` (a linear model of coregionalisation should keep
    the system valid). Returns the estimates (clipped to [0, 1] in
    ``probability``) and the weights.

    References
    ----------
    Sullivan, J. (1984). Conditional recovery estimation through probability
    kriging: theory and practice. In *Geostatistics for Natural Resources
    Characterization*, Reidel, 365-384.

    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
    Oxford University Press, section 7.3.

    Examples
    --------
    >>> r = probability_kriging([[0, 0], [1, 0], [0, 1], [1, 1]], [1.0, 3.0, 2.0, 4.0], [[0.5, 0.5]], 2.5,
    ...                         (0.0, 0.25, 1.0), (0.0, 0.08, 1.0), (0.0, 0.1, 1.0))
    >>> round(r.estimate[0], 12)
    0.5
    """
    P = _mat(coords)
    z = [float(v) for v in np.asarray(values, dtype=float).ravel().tolist()]
    Tg = _mat(targets)
    n = len(z)
    ind = [1.0 if v <= threshold else 0.0 for v in z]
    order = sorted(range(n), key=lambda i: (z[i], i))
    rank = [0.0] * n
    for r, i in enumerate(order):
        rank[i] = (r + 0.5) / n
    D = [[math.dist(P[i], P[j]) for j in range(n)] for i in range(n)]
    m = 2 * n + 2
    K = [[0.0] * m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            K[i][j] = _cov(D[i][j], cov_i)
            K[n + i][n + j] = _cov(D[i][j], cov_u)
            K[i][n + j] = K[n + j][i] = _cov(D[i][j], cov_iu)
        K[i][2 * n] = K[2 * n][i] = 1.0
        K[n + i][2 * n + 1] = K[2 * n + 1][n + i] = 1.0
    est, wts = [], []
    for t in Tg:
        d = [math.dist(t, p) for p in P]
        rhs = [_cov(v, cov_i) if v > 0 else cov_i[0] + cov_i[1] for v in d]
        rhs += [_cov(v, cov_iu) if v > 0 else cov_iu[0] + cov_iu[1] for v in d]
        rhs += [1.0, 0.0]
        w = solve(K, rhs)
        est.append(sum(w[i] * ind[i] for i in range(n)) + sum(w[n + i] * rank[i] for i in range(n)))
        wts.append(w[: 2 * n])
    return RichResult(
        payload={
            "estimate": est,
            "probability": [min(1.0, max(0.0, v)) for v in est],
            "weights": wts,
            "uniform": rank,
            "indicator": ind,
        }
    )


def cheatsheet() -> str:
    return "probability_kriging -> probability kriging."
