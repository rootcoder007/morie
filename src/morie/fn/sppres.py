# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel residual Moran test."""

from .miml import miml


def sppres(resid, W, alternative="greater"):
    r"""Moran's I of stacked panel residuals with the block weights ``I_T kron W``.

    ``resid`` holds ``T`` periods of ``N`` residuals stacked by period (units in
    the order of ``W``); the statistic is Moran's I with ``W_NT = I_T kron W``,
    ``I = (NT / S0_NT) e'W_NT e / e'e``, referred to its randomisation
    moments (Cliff and Ord 1981; Elhorst 2014, ch. 3). Thin front-end to
    :func:`morie.fn.miml.miml`.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion.
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data
    to Spatial Panels*. Springer.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> round(sppres([0.5, -0.2, 0.1, 0.3, -0.4, 0.2], W).statistic, 12)
    -0.776595744681
    """
    e = [float(v) for v in (resid.tolist() if hasattr(resid, "tolist") else resid)]
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    N = len(Wm)
    T = len(e) // N
    if len(e) != N * T:
        raise ValueError("len(resid) must be a multiple of the size of W")
    big = [[Wm[i % N][j % N] if i // N == j // N else 0.0 for j in range(N * T)] for i in range(N * T)]
    r = miml(e, big, alternative=alternative)
    r.name = "sppres"
    r.extra["T"] = T
    return r


sppres_fn = sppres


def cheatsheet() -> str:
    return "sppres(resid, W) -> Moran's I of stacked panel residuals with I_T kron W."
