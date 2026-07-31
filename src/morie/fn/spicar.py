"""Intrinsic CAR (ICAR) prior for Bayesian spatial models."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_icar_prior"]


def schabenberger_icar_prior(w, tau2=1.0):
    r"""
    The intrinsic CAR prior: precision structure, not a fit.

    ICAR is the CAR model at :math:`\rho = 1`. Its precision matrix is
    the graph Laplacian

    .. math::

        Q = D - W, \qquad D = \mathrm{diag}(W \mathbf{1})

    which makes the prior IMPROPER: :math:`Q\mathbf{1} = 0`, so it is
    rank deficient by the number of connected components and specifies
    only differences between neighbouring values, not their overall
    level. That is why an ICAR term is used as a prior component -- for
    instance the structured half of a BYM model -- and needs a
    sum-to-zero constraint to be identified.

    This returns the prior structure. To FIT an ICAR to data, see
    :func:`morie.fn.sgicar.intrinsic_car_model`.

    Parameters
    ----------
    w : array-like
        Adjacency weights, shape ``(n, n)``.
    tau2 : float, default 1.0
        Precision scale; ``Q`` is returned scaled by ``1 / tau2``.

    Returns
    -------
    RichResult
        ``Q`` (precision), ``D``, ``rank``, ``n_components`` (the rank
        deficiency), ``is_improper``, ``conditional_variances``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.4.3 "Selected
    Spatial Models".
    """
    W = np.atleast_2d(np.asarray(w, dtype=float))
    if W.shape[0] != W.shape[1]:
        raise ValueError("`w` must be square")
    if tau2 <= 0:
        raise ValueError("`tau2` must be > 0")
    n = W.shape[0]
    d = W.sum(axis=1)
    D = np.diag(d)
    Q = (D - W) / float(tau2)
    rank = int(np.linalg.matrix_rank(Q))
    with np.errstate(divide="ignore"):
        cond_var = np.where(d > 0, float(tau2) / np.where(d > 0, d, 1.0), np.inf)
    return RichResult(
        title="Intrinsic CAR (ICAR) prior",
        summary_lines=[("n", n), ("rank", rank),
                       ("rank deficiency", n - rank)],
        payload={"Q": Q, "D": D, "rank": rank, "n_components": n - rank,
                 "is_improper": rank < n,
                 "conditional_variances": cond_var, "tau2": float(tau2)},
    )


def cheatsheet():
    return "spicar: ICAR prior Q = D - W; improper, Q1 = 0."
