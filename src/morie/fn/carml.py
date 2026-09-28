# morie.fn -- function file (rootcoder007/morie)
"""CAR regression by maximum likelihood (spatialreg::spautolm, family = "CAR")."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spdurbin import car_ml


def carml(y, W, X=None):
    """CAR regression by maximum likelihood; ``statistic`` is the estimated ``lambda``.

    Delegates to :func:`morie.fn.spdurbin.car_ml` with the design ``X``
    (intercept only by default) and symmetric weights ``W``; ``p_value`` is
    the likelihood-ratio test of ``lambda = 0``.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = carml([1.0, 2.2, 2.9, 4.1], W, X=[[1, 0.0], [1, 1.0], [1, 2.0], [1, 3.0]])
    >>> r.extra["lr_test"]["df"]
    1
    """
    yv = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    Xm = [[1.0] for _ in yv] if X is None else np.asarray(X, dtype=float).tolist()
    r = car_ml(yv, Xm, np.asarray(W, dtype=float).tolist())
    return SpatialResult(name="carml", statistic=float(r["lambda"]), p_value=r.lr_test["pvalue"], extra=dict(r))


carml_fn = carml


def cheatsheet() -> str:
    return "carml({}) -> CAR regression by maximum likelihood."
