# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel LM test for spatial lag in panel."""

from ._containers import SpatialResult
from ._rrng_core import pchisq
from .sppdiag import _panel_rs


def spplm(y, X, W, time_id, unit_id, effects="pooled", robust=False):
    r"""Panel LM test for a spatial lag, RSlag = (e'(I_T kron W)y / sigma^2)^2 / nJ (or its robust form).

    The Anselin (1988) score test of rho = 0 on the stacked panel with
    block weights I_T kron W (Anselin, Le Gallo and Jayet 2008;
    splm::slmtest(test = "lml")); robust=True gives the
    Anselin-Bera-Florax-Yoon (1996) adjRSlag robust to local spatial error
    (test = "rlml"). See :func:`morie.fn.sppdiag.sppdiag` for
    effects.

    References
    ----------
    Anselin, L., Le Gallo, J. and Jayet, H. (2008). Spatial panel
    econometrics. In L. Matyas and P. Sevestre (eds), *The Econometrics of
    Panel Data*, 3rd ed. Springer, 625-660.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> round(spplm(y, X, W, tid, uid).statistic, 8)
    0.88322221
    """
    c = _panel_rs(y, X, W, time_id, unit_id, effects)
    stat = c["adjRSlag"] if robust else c["RSlag"]
    return SpatialResult(name="spplm", statistic=stat, p_value=float(pchisq(stat, 1, lower_tail=False)))


spplm_fn = spplm


def cheatsheet() -> str:
    return "spplm(y, X, W, time_id, unit_id) -> panel LM test for a spatial lag (splm::slmtest lml)."
