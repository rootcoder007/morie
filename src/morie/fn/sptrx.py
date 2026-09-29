# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit random-effects panel."""

from ._qpcore import ssum
from .spdiscrete import spatial_probit_gmm
from .sptfx import _panel


def sptrx(y, X, W, unit_id, time_id=None):
    r"""Correlated random-effects (Mundlak-Chamberlain) spatial probit panel by GMM.

    The unit effect is projected on the unit means of the regressors,
    alpha_i = xbar_i' xi + a_i (Mundlak 1978; Chamberlain 1980), so the
    model becomes a pooled spatial probit in [1, x_it, xbar_i] on the
    stacked panel with block weights I_T kron W, estimated by the
    Pinkse-Slade GMM of :func:`morie.fn.spdiscrete.spatial_probit_gmm`; the
    coefficients are scaled by the composite error standard deviation
    (Wooldridge 2010, sec. 15.8.2). Coefficients: intercept, X columns,
    then the xbar columns.

    References
    ----------
    Mundlak, Y. (1978). On the pooling of time series and cross section
    data. *Econometrica* 46, 69-85.
    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
    Panel Data*, 2nd ed. MIT Press.

    Examples
    --------
    >>> import math
    >>> W = [[0.5 if abs(i - j) in (1, 5) else 0.0 for j in range(6)] for i in range(6)]
    >>> uid = list(range(6)) * 10
    >>> X = [[math.sin(1.7 * k) + 0.3 * math.cos(0.4 * k)] for k in range(60)]
    >>> y = [1 if X[k][0] + 0.8 * math.sin(3.3 * k + 1) + 0.3 * ((k % 6) - 3) / 6 > 0 else 0 for k in range(60)]
    >>> round(sptrx(y, X, W, uid)["rho"], 8)
    0.27037292
    """
    yv, Xm, big, unit, N = _panel(y, X, W, unit_id, time_id)
    k = len(Xm[0])
    means = [
        [ssum(Xm[m][c] for m in range(len(Xm)) if unit[m] == g) / unit.count(g) for c in range(k)] for g in range(N)
    ]
    D = [[1.0] + r + means[unit[m]] for m, r in enumerate(Xm)]
    return spatial_probit_gmm(yv, D, big)


sptrx_fn = sptrx


def cheatsheet() -> str:
    return "sptrx(y, X, W, unit_id) -> Mundlak-Chamberlain CRE spatial probit panel by GMM (I_T kron W)."
