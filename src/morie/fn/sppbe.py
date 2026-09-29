# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel between estimator."""

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .sppfe import _flat, _rows


def sppbe(y, X, unit_id):
    r"""Between estimator of a panel regression: OLS of the unit means ybar_i on [1, xbar_i].

    The between estimator uses only the cross-sectional variation of the
    unit averages (Baltagi 2021, sec. 2.2; plm::plm(model = "between"));
    constant columns of X are dropped and an intercept is fitted. Returns
    the coefficients (intercept first), their conventional standard errors
    with sigma^2 = RSS / (N - k), the residuals and the unit ids in
    increasing order.

    References
    ----------
    Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*, 6th ed.
    Springer.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> [round(b, 8) for b in sppbe(y, X, uid)["coefficients"]]
    [0.36235142, 1.80175911]
    """
    yv = _flat(y)
    Xm = _rows(X)
    u = list(unit_id.tolist() if hasattr(unit_id, "tolist") else unit_id)
    ids = sorted(set(u))
    keep = [c for c in range(len(Xm[0])) if len({r[c] for r in Xm}) > 1]
    rows, ys = [], []
    for g in ids:
        idx = [k for k in range(len(yv)) if u[k] == g]
        ys.append(ssum(yv[k] for k in idx) / len(idx))
        rows.append([1.0] + [ssum(Xm[k][c] for k in idx) / len(idx) for c in keep])
    n, p = len(ys), len(rows[0])
    G = inverse([[ssum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)])
    beta = [ssum(G[a][b] * ssum(r[b] * v for r, v in zip(rows, ys)) for b in range(p)) for a in range(p)]
    res = [v - ssum(r[a] * beta[a] for a in range(p)) for r, v in zip(rows, ys)]
    s2 = ssum(e * e for e in res) / (n - p)
    return RichResult(
        payload={
            "coefficients": beta,
            "se": [(s2 * G[a][a]) ** 0.5 for a in range(p)],
            "residuals": res,
            "sigma2": s2,
            "units": ids,
        }
    )


sppbe_fn = sppbe


def cheatsheet() -> str:
    return "sppbe(y, X, unit_id) -> between estimator, OLS of unit means (plm model = 'between')."
