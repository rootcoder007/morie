# morie.fn -- function file (rootcoder007/morie)
"""SEM score test for spatial error parameter."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semsc(resid, W):
    r"""SEM score test for spatial error parameter.

    The Lagrange multiplier (Rao score) test of ``lambda = 0`` from OLS
    residuals ``e``: ``LM_err = (n e'We / e'e)^2 / T``, ``T = tr(W'W +
    WW)``, against chi-square(1) (Burridge 1980; Anselin 1988, eq. 13.7);
    ``RSerr`` of ``spdep::lm.RStests``. Needs only the residuals, so no
    alternative model is fitted.

    Parameters
    ----------
    resid : array-like, shape (n,)
        OLS residuals.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic``, ``p_value``, ``extra["df"] = 1``.

    References
    ----------
    Burridge, P. (1980). On the Cliff-Ord test for spatial correlation. *Journal of the Royal
    Statistical Society B*, 42(1), 107-108.

    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Rao, C. R. (1948). Large sample tests of statistical hypotheses concerning several parameters.
    *Mathematical Proceedings of the Cambridge Philosophical Society*, 44(1), 50-57.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = semsc([0.3, -0.2, 0.5, -0.4, 0.1, -0.3], W)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (2.64404296875, 0.103938733683)
    """
    stat, p = sd.lm_error(resid, W)
    return SpatialResult(name="semsc", statistic=stat, p_value=p, extra={"df": 1})


semsc_fn = semsc


def cheatsheet() -> str:
    return "semsc(resid, W) -> LM error (RSerr), chi-square(1)"
