# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cholesky (LU) simulation of a Gaussian random field."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_sim import cholesky_root, simulate_unconditional

__all__ = ["schabenberger_cholesky_sim"]


def schabenberger_cholesky_sim(mu, cov_matrix, seed=0, stream=0):
    """Simulate a Gaussian random field via the Cholesky root, Sec. 7.1.1.

    The chapter's starting point is the reproductive property of the
    Gaussian: if Sigma = Sigma^(1/2) Sigma^(1/2)' and X ~ G(0, I), then
    mu + Sigma^(1/2) X is G(mu, Sigma). Sec. 7.1.1 takes the square root to
    be the Cholesky (LU) root -- "Generate n independent standard Gaussian
    random deviates and store them in vector x ... Return y = mu + U'x as a
    realization from a G(mu, Sigma)".

    The book writes the root as an upper triangular U with Sigma = U'U; the
    lower-triangular L returned here is that U', so L L' = Sigma.

    Draws come from morie's own Philox/AS 241 generator, so the same seed
    gives the same field in the R arm -- not merely the same distribution.

    Parameters
    ----------
    mu : array-like, shape (n,)
        Mean vector.
    cov_matrix : array-like, shape (n, n)
        Covariance matrix; must be positive definite.
    seed, stream : int
        Generator handles. Different streams give independent realizations.

    Returns
    -------
    RichResult
        Keys: ``field``, ``root``, ``n``, ``method``.

    Notes
    -----
    Sec. 7.1.1 flags the cost: "It works well for small to moderate sized
    problems. As n grows large, however, calculating the Cholesky
    decomposition is numerically expensive." It is O(n^3).

    References
    ----------
    Schabenberger Ch 7, Sec 7.1.1
    """
    mu = np.asarray(mu, dtype=float).ravel()
    cov = np.atleast_2d(np.asarray(cov_matrix, dtype=float))
    if cov.shape != (mu.size, mu.size):
        raise ValueError("`cov_matrix` must be square and match `mu`")
    field = simulate_unconditional(mu, cov, method="cholesky",
                                   seed=seed, stream=stream)
    return RichResult(
        title="Cholesky simulation of a Gaussian random field",
        summary_lines=[("n", mu.size), ("field mean", float(field.mean())),
                       ("field sd", float(field.std(ddof=1)))],
        payload={"field": field, "root": cholesky_root(cov), "n": int(mu.size),
                 "method": "Cholesky (LU) decomposition"},
    )


def cheatsheet():
    return "spchol: Cholesky simulation of a Gaussian field (Schabenberger Sec 7.1.1)"
