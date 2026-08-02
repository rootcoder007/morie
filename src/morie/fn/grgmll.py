# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian mixture log-likelihood."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gmm_log_likelihood"]

_METHOD = "Gaussian mixture log-likelihood"


def _log_gauss(X, mu, Sigma):
    """Log density of N(mu, Sigma) at every row of X, via Cholesky."""
    d = X.shape[1]
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError(
            "a covariance matrix is not positive definite, so its Gaussian "
            "density is undefined; add a ridge to the diagonal."
        ) from None
    diff = (X - mu).T                       # (d, m)
    sol = np.linalg.solve(L, diff)          # L is lower-triangular
    maha = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


def geron_gmm_log_likelihood(X, pi, means, covars):
    r"""Log-likelihood of the data under a Gaussian mixture.

    .. math::
        \log L(\theta) = \sum_i \log\Bigl(
        \sum_k \pi_k\, \mathcal N(x_i \mid \mu_k, \Sigma_k)\Bigr)

    The inner sum is evaluated with the log-sum-exp trick -- the largest
    component log-density is factored out before exponentiating.  In
    high dimensions component densities are routinely ``e^-400``, which
    underflows to exactly 0 and turns the log into ``-inf``; the shift
    removes that failure mode entirely.

    Feed the result to :func:`morie.fn.graic.geron_aic_gmm` or
    ``grbic`` to compare mixtures of different ``k``.

    Parameters
    ----------
    X : array-like, shape (m, d)
    pi : array-like, shape (K,)
        Mixing weights, non-negative and summing to 1.
    means : array-like, shape (K, d)
    covars : array-like, shape (K, d, d)
        Positive-definite covariance matrices.

    Returns
    -------
    RichResult
        Payload keys ``log_likelihood``, ``per_sample``,
        ``mean_log_likelihood``, ``component_log_densities``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, GMM section.

    Examples
    --------
    A single standard normal component evaluated at its mean is just
    ``-0.5 log(2 pi)``:

    >>> r = geron_gmm_log_likelihood([[0.0]], [1.0], [[0.0]], [[[1.0]]])
    >>> round(r["log_likelihood"], 10)
    -0.9189385332

    Two identical components with weights 0.5 and 0.5 are the same
    density as one -- the mixture weights sum inside the log:

    >>> r2 = geron_gmm_log_likelihood([[0.0]], [0.5, 0.5], [[0.0], [0.0]],
    ...                               [[[1.0]], [[1.0]]])
    >>> round(r2["log_likelihood"], 10)
    -0.9189385332
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    w = np.asarray(pi, dtype=float).ravel()
    M = np.atleast_2d(np.asarray(means, dtype=float))
    S = np.asarray(covars, dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {A.shape}.")
    m, d = A.shape
    K = w.size
    if M.shape != (K, d):
        raise ValueError(f"means must have shape (K, d) = ({K}, {d}), got {M.shape}.")
    if S.ndim == 2 and K == 1:
        S = S.reshape(1, d, d)
    if S.shape != (K, d, d):
        raise ValueError(f"covars must have shape (K, d, d) = ({K}, {d}, {d}), got {S.shape}.")
    if np.any(w < 0):
        raise ValueError("mixing weights must be non-negative.")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"mixing weights must sum to 1, got {w.sum()}.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(M)) and np.all(np.isfinite(S))):
        raise ValueError("X, means and covars must all be finite.")

    logp = np.empty((m, K))
    for k in range(K):
        if w[k] == 0:
            logp[:, k] = -np.inf
        else:
            logp[:, k] = np.log(w[k]) + _log_gauss(A, M[k], S[k])

    mx = logp.max(axis=1, keepdims=True)
    per = (mx + np.log(np.exp(logp - mx).sum(axis=1, keepdims=True))).ravel()
    if not np.all(np.isfinite(per)):
        raise ValueError(
            "some instance has zero density under every component; the mixture "
            "cannot have generated this data (log-likelihood is -inf)."
        )
    total = float(per.sum())

    return RichResult(
        title="GMM log-likelihood",
        summary_lines=[("log L", total), ("Components", int(K)), ("m", int(m))],
        payload={
            "log_likelihood": total,
            "per_sample": per.tolist(),
            "mean_log_likelihood": float(per.mean()),
            "component_log_densities": logp.tolist(),
            "n_components": int(K),
            "estimate": total,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgmll: log L = sum_i logsumexp_k [log pi_k + log N(x_i | mu_k, Sigma_k)]"
