"""Dimensionality of spatial data: scree, elbow and parallel analysis on the singular values.

Cattell, R. B. (1966). The scree test for the number of factors. Multivariate Behavioral Research
1, 245-276. Horn, J. L. (1965). A rationale and test for the number of factors in factor
analysis. Psychometrika 30, 179-185. Satopaa, V., Albrecht, J., Irwin, D. and Raghavan, B.
(2011). Finding a "kneedle" in a haystack: detecting knee points in system behavior. 31st ICDCS
Workshops, 166-171.
"""

from ._metaheur import Rand
from ._mlfa import eigh_desc
from ._richresult import RichResult

__all__ = ["dimensionality"]


def _eig(X):
    n, m = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(m)]
    C = [[0.0] * m for _ in range(m)]
    for r in X:
        c = [r[j] - mu[j] for j in range(m)]
        for a in range(m):
            for b in range(a, m):
                C[a][b] += c[a] * c[b]
    for a in range(m):
        for b in range(a, m):
            C[a][b] /= n - 1
            C[b][a] = C[a][b]
    return eigh_desc(C)[0]


def dimensionality(X, n_sim=100, seed=0, quantile=0.95):
    r"""Eigenvalues of the column-centred covariance of X (the squared singular values over n - 1),
    their shares (the scree), the elbow (the eigenvalue farthest below the straight line from the
    first to the last, the Kneedle rule), and Horn's parallel analysis: eigenvalues of ``n_sim``
    matrices whose columns are independent Philox-drawn permutations of X's columns, keeping the
    leading eigenvalues that exceed the ``quantile`` of their simulated counterparts. The second
    dimension is tested by the share of simulations whose second eigenvalue reaches the observed.

    Parameters
    ----------
    X : n x m data (e.g. legislators by roll calls coded 1/0)
    n_sim : int
    seed : int
    quantile : float

    Returns
    -------
    RichResult
        Keys: eigenvalues, share, elbow (1-based count before the knee), parallel (retained
        dimensions), threshold (simulated quantiles), p_second (Monte Carlo p-value of the
        second eigenvalue).

    References
    ----------
    Horn, J. L. (1965). Psychometrika 30, 179-185.
    Cattell, R. B. (1966). Multivariate Behavioral Research 1, 245-276.

    Examples
    --------
    >>> r = dimensionality([[1, 2], [2, 4], [3, 6], [4, 8]], n_sim=0)
    >>> [round(v, 12) for v in r["share"]]
    [1.0, 0.0]
    """
    X = [[float(v) for v in r] for r in X]
    n, m = len(X), len(X[0])
    ev = [max(v, 0.0) for v in _eig(X)]
    tot = sum(ev)
    share = [v / tot for v in ev]
    k = len(ev)
    if k >= 3:
        _x0, y0, _x1, y1 = 0.0, ev[0], float(k - 1), ev[-1]
        gap = [(y0 + (y1 - y0) * i / (k - 1)) - ev[i] for i in range(k)]
        elbow = max(range(k), key=lambda i: (gap[i], -i))
    else:
        elbow = 1
    out = {"eigenvalues": ev, "share": share, "elbow": elbow}
    if n_sim:
        rnd = Rand(seed)
        sims = []
        for _ in range(int(n_sim)):
            cols = []
            for j in range(m):
                col = [r[j] for r in X]
                for i in range(n - 1, 0, -1):  # Fisher-Yates with Philox draws
                    s = rnd.idx(i + 1)
                    col[i], col[s] = col[s], col[i]
                cols.append(col)
            sims.append(_eig([[cols[j][i] for j in range(m)] for i in range(n)]))
        thr = []
        for d in range(k):
            vals = sorted(s[d] for s in sims)
            thr.append(vals[min(int(quantile * len(vals)), len(vals) - 1)])
        keep = 0
        while keep < k and ev[keep] > thr[keep]:
            keep += 1
        out.update(
            parallel=keep,
            threshold=thr,
            p_second=(1 + sum(1 for s in sims if s[1] >= ev[1])) / (1 + len(sims)) if k > 1 else None,
        )
    return RichResult(title="Dimensionality", summary_lines=[("elbow", out["elbow"])], payload=out)


def cheatsheet():
    return "svdimn: scree, elbow and parallel analysis for the number of spatial dimensions"
