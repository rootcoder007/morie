# morie.fn -- function file (rootcoder007/morie)
"""Spatial GLMM simulation"""

from __future__ import annotations

from ._containers import DescriptiveResult
from .sglmm import spatial_glmm_simulate


def spatial_glmm_sim(X, beta, latent, *, family="poisson", trials=1, size=None, shape=None, sigma=1.0, seed=1):
    r"""Spatial GLMM simulation

    Draws responses of the spatial generalised linear mixed model ``g(E y_i)
    = x_i'beta + S_i`` given a realisation ``S`` of the latent field (Diggle,
    Tawn and Moyeed 1998) -- Gaussian (identity), Poisson (log), binomial
    (logit), negative binomial or gamma (log) -- by inversion of Philox
    uniforms. This is :func:`morie.fn.sglmm.spatial_glmm_simulate`.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    beta : array-like, shape (p,)
        Coefficients.
    latent : array-like, shape (n,)
        Latent field values.
    family : str
        ``gaussian``, ``poisson``, ``binomial``, ``negbin`` or ``gamma``.
    trials, size, shape, sigma : optional
        Family parameters.
    seed : int
        Philox seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean simulated response; ``extra`` has ``y``, ``eta``, ``mu``.

    References
    ----------
    Diggle, P. J., Tawn, J. A. and Moyeed, R. A. (1998). Model-based geostatistics. *Applied Statistics*,
    47(3), 299-350.

    Examples
    --------
    >>> r = spatial_glmm_sim([[1.0], [1.0], [1.0]], [0.5], [0.0, 0.3, -0.2], seed=2)
    >>> r.extra["y"]
    [1.0, 2.0, 4.0]
    """
    r = spatial_glmm_simulate(
        X, beta, latent, family=family, trials=trials, size=size, shape=shape, sigma=sigma, seed=seed
    )
    y = list(r["y"])
    return DescriptiveResult(
        name="zsglm",
        value=sum(y) / len(y),
        extra={"y": y, "eta": list(r["eta"]), "mu": list(r["mu"]), "family": family},
    )


spat = spatial_glmm_sim
spatialglmmsim = spatial_glmm_sim


def cheatsheet() -> str:
    return "spatial_glmm_sim(...) -> Spatial GLMM simulation"
