# morie.fn -- function file (rootcoder007/morie)
"""SEM convergence check (lambda feasibility)."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semconv(W, lam):
    r"""SEM convergence check (lambda feasibility).

    The log-likelihood of a spatial lag / error model is defined only where
    ``I - a W`` is non-singular with the sign of ``a = 0``, i.e. inside
    ``(1 / e_min, 1 / e_max)`` with ``e_min``, ``e_max`` the smallest and
    largest real parts of the eigenvalues of ``W`` -- the search interval
    ``spatialreg`` uses with ``method = "eigen"`` (Ord 1975; LeSage and Pace
    2009, sec. 4.1). For a row-standardised ``W`` the upper bound is 1.

    Parameters
    ----------
    W : array-like, shape (n, n)
        Spatial weights.
    lam : float
        Parameter value(s) to check.

    Returns
    -------
    SpatialResult
        ``statistic`` is 1.0 when every parameter lies strictly inside the
        interval, else 0.0; ``extra`` has ``lower``, ``upper``, ``feasible``
        and ``logdet`` (the log-Jacobian, ``None`` when infeasible).

    References
    ----------
    Ord, K. (1975). Estimation methods for models of spatial interaction. *Journal of the American
    Statistical Association*, 70(349), 120-126.

    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> r = semconv([[0, 1, 0], [.5, 0, .5], [0, 1, 0]], 0.5)
    >>> r.statistic, round(r.extra["lower"], 12), round(r.extra["upper"], 12)
    (1.0, -1.0, 1.0)
    """
    lo, hi = sd.bounds(W)
    ok = lo < lam < hi
    return SpatialResult(
        name="semconv",
        statistic=1.0 if ok else 0.0,
        extra={"lower": lo, "upper": hi, "feasible": ok, "logdet": (sd.logdet(W, lam)) if ok else None},
    )


semconv_fn = semconv


def cheatsheet() -> str:
    return "semconv(W, lam) -> inside (1/e_min, 1/e_max)?"
