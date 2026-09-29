# morie.fn -- function file (rootcoder007/morie)
"""Spatial logit ML with GHK simulator."""

from .spdiscrete import spatial_logit_gmm
from .spprmf import _rows


def splgtml(y, X, W, Z=None):
    r"""Spatial autoregressive logit estimated by the Klier-McMillen (2008) linearized GMM.

    The GHK simulator evaluates multivariate normal rectangle probabilities
    and applies to the Gaussian (probit) latent model (see
    :func:`morie.fn.spprml.spprml`); for the logistic SAR model the standard
    estimator is the linearized GMM of Klier and McMillen (2008), to which
    this is a thin front-end (:func:`morie.fn.spdiscrete.spatial_logit_gmm`,
    as McSpatial::splogit). An intercept is prepended when X has no
    constant column; Z are the instruments (default [X, WX]).

    References
    ----------
    Klier, T. and McMillen, D. P. (2008). Clustering of auto supplier plants
    in the United States: generalized method of moments spatial logit for
    large samples. *JBES* 26, 460-471.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> round(splgtml([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], X, W)["rho"], 10)
    -0.5078680767
    """
    Xm = _rows(X)
    if not any(len({r[c] for r in Xm}) == 1 and Xm[0][c] != 0.0 for c in range(len(Xm[0]))):
        Xm = [[1.0] + r for r in Xm]
    return spatial_logit_gmm(y, Xm, W, Z)


splgtml_fn = splgtml


def cheatsheet() -> str:
    return "splgtml(y, X, W) -> SAR logit by Klier-McMillen linearized GMM (McSpatial::splogit)."
