"""Best Linear Unbiased Predictor (BLUP) for spatial prediction."""

import numpy as np

from ._richresult import RichResult
from ._schab_krig import cov_from_model, _dist

__all__ = ["schabenberger_blup"]


def schabenberger_blup(coords, z, target, cov_model=None):
    r"""
    Best linear unbiased predictor -- ordinary kriging.

    Simple kriging assumes the mean is known. When it is unknown but
    constant, unbiasedness requires the weights to sum to one, and the
    constraint is carried by a Lagrange multiplier :math:`m`:

    .. math::

        \begin{bmatrix} \Sigma & 1 \\ 1' & 0 \end{bmatrix}
        \begin{bmatrix} \lambda \\ m \end{bmatrix} =
        \begin{bmatrix} \sigma \\ 1 \end{bmatrix}

    giving :math:`p(Z; s_0) = \lambda' Z(s)` and

    .. math::

        \sigma^2_{ok}(s_0) = \sigma^2 - \lambda'\sigma - m

    The BLUP is never better than simple kriging in mean-squared error --
    it pays for not knowing the mean -- but it does not require one.

    Parameters
    ----------
    coords : array-like
        Observation coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    target : array-like
        Prediction location(s), shape ``(m, d)``.
    cov_model : mapping, optional
        ``{'model', 'nugget', 'sill', 'range'}``.

    Returns
    -------
    RichResult
        ``prediction``, ``variance``, ``weights``, ``lagrange``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Secs. 5.1-5.2, p. 215 ff.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    target = np.atleast_2d(np.asarray(target, dtype=float))
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    n = z.size
    Sigma = cov_from_model(_dist(coords, coords), cov_model)
    sig = cov_from_model(_dist(coords, target), cov_model)
    sigma2 = float(cov_from_model(np.zeros(1), cov_model)[0])

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Sigma
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    b = np.vstack([sig, np.ones((1, target.shape[0]))])
    sol = np.linalg.solve(A, b)
    lam, m = sol[:n, :], sol[n, :]
    pred = lam.T @ z
    var = sigma2 - np.einsum("ij,ij->j", sig, lam) - m
    return RichResult(
        title="BLUP (ordinary kriging)",
        summary_lines=[("n targets", int(pred.size))],
        payload={"prediction": pred, "variance": np.maximum(var, 0.0),
                 "weights": lam, "lagrange": m},
    )


def cheatsheet():
    return "spblup: ordinary kriging BLUP; weights sum to 1 via a Lagrange term."
