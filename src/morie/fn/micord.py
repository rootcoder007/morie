# morie.fn -- function file (rootcoder007/morie)
"""Predictive mean matching imputation."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform


def mi_pmm(y, X, R, K=5, seed=0):
    r"""Predictive mean matching (PMM) imputation of a partially observed outcome.

    ``y`` is regressed on ``[1, X]`` over the observed cases (``R = 1``);
    for each missing case the ``K`` observed cases whose predicted means are
    closest to its own (ties by index) form the donor pool, and one donor,
    drawn uniformly by the Philox uniform of stream 0 of ``seed``, gives its
    OBSERVED value (Little 1988; van Buuren 2018, sec. 3.4, matching type
    0). Imputations are therefore always values that occur in the data.
    Returns the completed ``y``, the donors (0-based) and the regression
    coefficients.

    References
    ----------
    Little, R. J. A. (1988). Missing-data adjustments in large surveys.
    *Journal of Business and Economic Statistics* 6, 287-296.
    van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed.
    CRC Press.

    Examples
    --------
    >>> y = [1.0, 2.1, 0.0, 3.9, 5.2, 0.0]
    >>> X = [[0.0], [1.0], [1.4], [2.0], [3.0], [2.9]]
    >>> mi_pmm(y, X, [1, 1, 0, 1, 1, 0], K=1)["imputed"]
    [1.0, 2.1, 2.1, 3.9, 5.2, 5.2]
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm = [
        [float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in (X.tolist() if hasattr(X, "tolist") else X)
    ]
    rv = [int(v) for v in (R.tolist() if hasattr(R, "tolist") else R)]
    n = len(yv)
    obs = [i for i in range(n) if rv[i] == 1]
    mis = [i for i in range(n) if rv[i] == 0]
    K = int(K)
    if not 1 <= K <= len(obs):
        raise ValueError("K must lie between 1 and the number of observed cases")
    D = [[1.0] + r for r in Xm]
    p = len(D[0])
    G = inverse([[ssum(D[i][a] * D[i][b] for i in obs) for b in range(p)] for a in range(p)])
    beta = [ssum(G[a][b] * ssum(D[i][b] * yv[i] for i in obs) for b in range(p)) for a in range(p)]
    yhat = [ssum(r[a] * beta[a] for a in range(p)) for r in D]
    u = [float(v) for v in random_uniform(max(len(mis), 1), seed=seed, stream=0)]
    out = list(yv)
    donors = []
    for t, i in enumerate(mis):
        pool = sorted(obs, key=lambda j: (abs(yhat[i] - yhat[j]), j))[:K]
        d = pool[min(K - 1, int(math.floor(u[t] * K)))]
        donors.append(d)
        out[i] = yv[d]
    return RichResult(
        payload={
            "imputed": out,
            "donors": donors,
            "missing": mis,
            "coefficients": beta,
            "method": "predictive mean matching (type 0)",
        }
    )


def cheatsheet():
    return "micord: PMM imputation, donor drawn from the K nearest predicted means (Little 1988)"


# compact alias per ledger/NAMING.md
mipmm = mi_pmm
