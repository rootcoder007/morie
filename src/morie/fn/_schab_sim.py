# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simulation of Gaussian random fields.

Schabenberger & Gotway (2005), Ch. 7. The chapter opens from the
reproductive property of the Gaussian: if Sigma = Sigma^(1/2) Sigma^(1/2)'
and X ~ G(0, I), then mu + Sigma^(1/2) X has a G(mu, Sigma) distribution.
Everything in Sec. 7.1 is a choice of square root.

  Sec. 7.1.1  Cholesky (LU): Sigma = U'U with U upper triangular, and
              "Return y = mu + U'x as a realization from a G(mu, Sigma)".
  Sec. 7.1.2  Spectral decomposition: Sigma = P Delta P', so
              Sigma^(1/2) = P Delta^(1/2) P' -- the SYMMETRIC root, not a
              triangular one, which is what makes it a different field from
              the Cholesky construction given the same X.
  Sec. 7.2.2  Conditioning by kriging, eq (7.1):
              Zc(s) = S(s) + c' Sigma^-1 (Z - Sm).

Draws come from morie's own Philox/AS 241 generator, so an R and a Python
run with the same seed produce the same field rather than merely the same
distribution.

References
----------
Schabenberger, O. & Gotway, C. A. (2005) *Statistical Methods for
Spatial Data Analysis*, Texts in Statistical Science, Chapman &
Hall/CRC, Boca Raton, ISBN 1-58488-322-7.
Chapter 7, eq (7.1) (conditional simulation by kriging).

Everything here is internal.
"""

from . import _array_core as np

from ._rng import random_normal

__all__ = []


def cholesky_root(cov, jitter=1e-10):
    """The lower-triangular L with L L' = Sigma, Sec. 7.1.1.

    The book writes the root as an upper triangular U with Sigma = U'U; L
    here is U', so L L' = U'U = Sigma. Same matrix, transposed convention.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("`cov` must be square")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))


def spectral_root(cov, tol=None):
    """The symmetric square root P Delta^(1/2) P', Sec. 7.1.2.

    Negative eigenvalues can only arise from rounding on a matrix that is
    positive semi-definite in exact arithmetic, so they are clipped to zero
    rather than being allowed to produce NaNs through the square root. The
    threshold is the standard numerical rank tolerance, not a tuned value.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("`cov` must be square")
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    if tol is None:
        tol = sym.shape[0] * np.finfo(float).eps * max(abs(vals.max()), 1.0)
    vals = np.where(vals < tol, 0.0, vals)
    return (vecs * np.sqrt(vals)) @ vecs.T


def simulate_unconditional(mean, cov, method="cholesky", seed=0, stream=0):
    """One draw of a Gaussian random field, Sec. 7.1.

    `method` selects the square root: "cholesky" (Sec. 7.1.1) or "spectral"
    (Sec. 7.1.2). Both give a field with covariance `cov`; they do NOT give
    the same field from the same stream, because they are different roots.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    n = mean.size
    if cov.shape != (n, n):
        raise ValueError("`cov` must be square and match `mean`")
    if method == "cholesky":
        root = cholesky_root(cov)
    elif method == "spectral":
        root = spectral_root(cov)
    else:
        raise ValueError("`method` must be 'cholesky' or 'spectral'")
    return mean + root @ random_normal(n, seed=seed, stream=stream)


def simulate_conditional(cov_all, z_obs, n_obs, mean=0.0, method="cholesky",
                         seed=0, stream=0):
    """Conditioning an unconditional simulation by kriging, eq (7.1).

        Zc(s) = S(s) + c' Sigma^-1 (Z - Sm)

    `cov_all` is the covariance over the observed locations followed by the
    targets, `z_obs` the observed values, `n_obs` how many leading rows are
    observed. The text notes the unconditional simulation need not carry the
    same mean -- "Any mean will do, for example E[S(s)] = 0" -- so the
    default is zero and the observed mean rides in through the correction.

    The book states three properties this must have, all asserted in the
    test suites: the realization honors the data exactly at the observed
    locations, it is unconditionally unbiased, and it reproduces the
    covariance of Z.
    """
    cov_all = np.atleast_2d(np.asarray(cov_all, dtype=float))
    z_obs = np.asarray(z_obs, dtype=float).ravel()
    n_obs = int(n_obs)
    n = cov_all.shape[0]
    if cov_all.shape[1] != n:
        raise ValueError("`cov_all` must be square")
    if z_obs.size != n_obs:
        raise ValueError("`z_obs` must have `n_obs` entries")
    if not 0 < n_obs < n:
        raise ValueError("`n_obs` must leave at least one target location")

    mu = np.full(n, float(mean)) if np.isscalar(mean) else np.asarray(mean, float)
    sim = simulate_unconditional(np.zeros(n), cov_all, method=method,
                                 seed=seed, stream=stream)
    sigma_obs = cov_all[:n_obs, :n_obs]
    c = cov_all[:, :n_obs]                       # Cov(all, observed)
    resid = (z_obs - mu[:n_obs]) - sim[:n_obs]
    correction = c @ np.linalg.solve(sigma_obs, resid)
    return mu + sim + correction


def simple_kriging_variance(cov_all, n_obs):
    """sigma^2_sk at every location, for the 2 sigma^2_sk identity.

    Sec. 7.2.2 closes with E[(Zc(s) - Z(s))^2] = 2 sigma^2_sk, which is the
    sharpest available check that a conditional simulation is doing what the
    book says: it must be exactly twice as far from the truth, in mean
    square, as the kriging predictor is.
    """
    cov_all = np.atleast_2d(np.asarray(cov_all, dtype=float))
    n_obs = int(n_obs)
    sigma_obs = cov_all[:n_obs, :n_obs]
    c = cov_all[:, :n_obs]
    quad = np.einsum("ij,ij->i", c, np.linalg.solve(sigma_obs, c.T).T)
    return np.diag(cov_all) - quad
