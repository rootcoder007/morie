# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel diagnostics (LM tests)."""

from ._richresult import RichResult
from ._rrng_core import pchisq
from .lmdiag import _rs_core
from .sppanel import _demean
from .sppfe import _stack


def _panel_rs(y, X, W, time_id, unit_id, effects):
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    if effects != "pooled":
        yv = _demean(yv, N, T, effects)
        cols = [_demean([r[c] for r in Xm], N, T, effects) for c in range(len(Xm[0]))]
        Xm = [[c[k] for c in cols] for k in range(N * T)]
    big = [[Wm[i % N][j % N] if i // N == j // N else 0.0 for j in range(N * T)] for i in range(N * T)]
    return _rs_core(yv, Xm, big, intercept=(effects == "pooled"))


def sppdiag(y, X, W, time_id, unit_id, effects="pooled"):
    r"""Lagrange multiplier diagnostics for spatial dependence in a panel regression.

    The Anselin (1988) and Anselin, Bera, Florax and Yoon (1996) Rao score
    tests (RSerr, RSlag, the robust adjRSerr/adjRSlag and
    SARMA, see :func:`morie.fn.lmdiag.lmdiag`) computed on the stacked
    panel with the block weights I_T kron W (Anselin, Le Gallo and Jayet
    2008; Elhorst 2014, sec. 3.3): pooled OLS for effects="pooled", the
    within-transformed data (no intercept) for "individual", "time"
    or "twoways", as splm::slmtest.

    References
    ----------
    Anselin, L., Le Gallo, J. and Jayet, H. (2008). Spatial panel
    econometrics. In L. Matyas and P. Sevestre (eds), *The Econometrics of
    Panel Data*, 3rd ed. Springer, 625-660.
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data
    to Spatial Panels*. Springer.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> round(sppdiag(y, X, W, tid, uid)["RSlag"], 8)
    0.88322221
    """
    c = _panel_rs(y, X, W, time_id, unit_id, effects)
    df = {"RSerr": 1, "RSlag": 1, "adjRSerr": 1, "adjRSlag": 1, "SARMA": 2}
    out = {}
    for key, d in df.items():
        out[key] = c[key]
        out["p_" + key] = float(pchisq(c[key], d, lower_tail=False))
    return RichResult(payload=out)


sppdiag_fn = sppdiag


def cheatsheet() -> str:
    return "sppdiag(y, X, W, time_id, unit_id) -> panel LM tests with I_T kron W (splm::slmtest)."
