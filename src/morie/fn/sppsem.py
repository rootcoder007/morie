# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel SEM estimator."""

from .sppanel import spatial_panel_ml
from .sppfe import _stack


def sppsem(y, X, W, time_id, unit_id, effects="individual", lee_yu=False):
    r"""Spatial error (SEM) panel model ``y_t = X_t beta + effects + u_t``, ``u_t = lambda W u_t + e_t``, by concentrated maximum likelihood with ``beta(lambda)`` by GLS (Elhorst 2003; ``splm::spml(spatial.error = "b")``); ``rho`` holds ``lambda``.

    ``effects`` is ``"individual"``, ``"time"``, ``"twoways"`` or ``"pooled"``
    (demeaning, or an intercept); rows are matched to ``W`` by sorting on
    (period, unit id) and constant columns of ``X`` are dropped. Thin
    front-end to :func:`morie.fn.sppanel.spatial_panel_ml` (``model="error"``).

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
    >>> round(sppsem(y, X, W, tid, uid)["rho"], 8)
    -0.15172903
    """
    yv, Xm, N, _ = _stack(y, X, time_id, unit_id)
    return spatial_panel_ml(yv, Xm, W, N, model="error", effects=effects, lee_yu=lee_yu)


sppsem_fn = sppsem


def cheatsheet() -> str:
    return "sppsem(y, X, W, time_id, unit_id) -> spatial panel error model by ML (Elhorst 2003)."
