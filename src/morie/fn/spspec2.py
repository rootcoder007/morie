# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spectral-decomposition simulation of a Gaussian random field."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_sim import simulate_unconditional, spectral_root

__all__ = ["schabenberger_spectral_sim"]


def schabenberger_spectral_sim(mu, cov_matrix, seed=0, stream=0):
    """Simulate a Gaussian random field via the spectral root, Sec. 7.1.2.

    A second choice of square root. For a real symmetric A there is an
    orthogonal P with A = P Delta P', Delta the eigenvalues; since P'P = I,

        Sigma^(1/2) = P Delta^(1/2) P'

    "has the needed properties to function as the square root matrix to
    generate G(mu, Sigma) deviates by y = mu + Sigma^(1/2) x".

    This root is SYMMETRIC where the Cholesky root is triangular, so from
    the same random stream the two produce different fields -- both with
    covariance Sigma. That is not a discrepancy; it is what choosing a
    different square root means.

    Note on the previous docstring of this module, which gave the formula as
    ``Z = sum_k sqrt(S(omega_k)) [a_k cos(omega_k's) + b_k sin(omega_k's)]``:
    that is the spectral-DENSITY method, which builds a field from harmonics
    of the spectral density. It is a different technique from Sec. 7.1.2,
    which decomposes the covariance MATRIX. The module title and the cited
    section both say the latter, so the latter is what is implemented.

    Parameters
    ----------
    mu : array-like, shape (n,)
        Mean vector.
    cov_matrix : array-like, shape (n, n)
        Covariance matrix; must be positive semi-definite.
    seed, stream : int
        Generator handles.

    Returns
    -------
    RichResult
        Keys: ``field``, ``root``, ``eigenvalues``, ``n``, ``method``.

    References
    ----------
    Schabenberger Ch 7, Sec 7.1.2
    """
    mu = np.asarray(mu, dtype=float).ravel()
    cov = np.atleast_2d(np.asarray(cov_matrix, dtype=float))
    if cov.shape != (mu.size, mu.size):
        raise ValueError("`cov_matrix` must be square and match `mu`")
    field = simulate_unconditional(mu, cov, method="spectral",
                                   seed=seed, stream=stream)
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    return RichResult(
        title="Spectral simulation of a Gaussian random field",
        summary_lines=[("n", mu.size), ("field mean", float(field.mean())),
                       ("smallest eigenvalue", float(vals.min()))],
        payload={"field": field, "root": spectral_root(cov),
                 "eigenvalues": vals, "n": int(mu.size),
                 "method": "spectral decomposition"},
    )


def cheatsheet():
    return "spspec2: spectral simulation of a Gaussian field (Schabenberger Sec 7.1.2)"
