# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel SAR (spatial lag) estimator."""

from .sppanel import spatial_panel_ml
from .sppfe import _stack


def sppsar(y, X, W, time_id, unit_id, effects="individual", lee_yu=False):
    r"""Spatial lag (SAR) panel model ``y_t = rho W y_t + X_t beta + effects + e_t`` by concentrated maximum likelihood (Elhorst 2003; ``splm::spml(lag = TRUE)``).

    ``effects`` is ``"individual"``, ``"time"``, ``"twoways"`` or ``"pooled"``
    (demeaning, or an intercept); rows are matched to ``W`` by sorting on
    (period, unit id) and constant columns of ``X`` are dropped. Thin
    front-end to :func:`morie.fn.sppanel.spatial_panel_ml` (``model="lag"``).

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.
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
    >>> round(sppsar(y, X, W, tid, uid)["rho"], 8)
    -0.0808892
    """
    yv, Xm, N, _ = _stack(y, X, time_id, unit_id)
    return spatial_panel_ml(yv, Xm, W, N, model="lag", effects=effects, lee_yu=lee_yu)


sppsar_fn = sppsar


def cheatsheet() -> str:
    return "sppsar(y, X, W, time_id, unit_id) -> spatial panel lag model by ML (Elhorst 2003)."
