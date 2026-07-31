"""Kriging weights from the kriging system solution."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_kriging_weights"]


def schabenberger_kriging_weights(cov_matrix, cov_target, coords=None,
                                  unbiased=False):
    r"""
    Solve the kriging system for the weights.

    Simple kriging solves :math:`\Sigma \lambda = \sigma`. Ordinary
    kriging adds the unbiasedness constraint :math:`\sum_i \lambda_i = 1`
    through a Lagrange multiplier, bordering the system with a row and
    column of ones.

    Parameters
    ----------
    cov_matrix : array-like
        :math:`\Sigma`, the ``(n, n)`` covariance among observations.
    cov_target : array-like
        :math:`\sigma`, covariance between observations and the target,
        shape ``(n,)``.
    coords : array-like, optional
        Unused; accepted for call-site compatibility.
    unbiased : bool, default False
        Impose the sum-to-one constraint (ordinary rather than simple
        kriging).

    Returns
    -------
    RichResult
        ``weights``, ``weight_sum``, and ``lagrange`` when ``unbiased``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.2.
    """
    Sigma = np.atleast_2d(np.asarray(cov_matrix, dtype=float))
    sig = np.asarray(cov_target, dtype=float).ravel()
    n = Sigma.shape[0]
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("`cov_matrix` must be square")
    if sig.size != n:
        raise ValueError("`cov_target` must have one entry per observation")
    if not unbiased:
        lam = np.linalg.solve(Sigma, sig)
        return RichResult(
            title="Simple kriging weights",
            summary_lines=[("n", n), ("sum of weights", float(lam.sum()))],
            payload={"weights": lam, "weight_sum": float(lam.sum()),
                     "lagrange": None, "unbiased": False},
        )
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Sigma
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    b = np.concatenate([sig, [1.0]])
    sol = np.linalg.solve(A, b)
    lam = sol[:n]
    return RichResult(
        title="Ordinary kriging weights",
        summary_lines=[("n", n), ("sum of weights", float(lam.sum()))],
        payload={"weights": lam, "weight_sum": float(lam.sum()),
                 "lagrange": float(sol[n]), "unbiased": True},
    )


def cheatsheet():
    return "spkwt: solve Sigma lambda = sigma; unbiased=TRUE adds sum-to-one."
