# morie.fn -- function file (rootcoder007/morie)
"""LM test for spatial Durbin model."""

from .lmdiag import _result, _rs_core


def lmsdm(y, X, W):
    r"""Joint Rao score test of the spatial Durbin model, rho = 0 and gamma = 0, after OLS.

    The score vector d = [u'Wy, X'W'u] / sigma2 (lag and lagged
    non-constant regressors) is weighted by the inverse of the partitioned
    information block J22 - J21 J11^{-1} J12, giving a chi-square with
    1 + k_x df (Koley and Bera 2024; spreg.lm_spdurbin,
    spdep::SD.RStests(test = "SDM_Joint")). extra holds the robust
    parts adjRSWX = joint - RSlag (k_x df) and adjRSlag = joint -
    RS_WX (1 df).

    References
    ----------
    Koley, M. and Bera, A. K. (2024). To use, or not to use the spatial
    Durbin model? That is the question. *Spatial Economic Analysis* 19,
    30-56.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmsdm([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    3.7535691455
    """
    c = _rs_core(y, X, W)
    if not c["kx"]:
        raise ValueError("X needs at least one non-constant column")
    return _result(
        "lmsdm",
        c["RSjoint_durbin"],
        1 + c["kx"],
        {"adjRSWX": c["adjRSWX"], "adjRSlag": c["adjRSlag_durbin"], "RSWX": c["RSWX"], "RSlag": c["RSlag"]},
    )


lmsdm_fn = lmsdm


def cheatsheet() -> str:
    return "lmsdm(y, X, W) -> joint Rao score test rho = gamma = 0 (Koley-Bera 2024)."
