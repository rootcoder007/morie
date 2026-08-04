# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian mixture model fit via EM."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gaussian_mixture", "gmm_log_pdf"]


def gmm_log_pdf(X, mu, Sigma):
    """Log density of a multivariate normal at every row of ``X``."""
    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("gmm_log_pdf: covariance is not positive definite")
    diff = X - mu
    sol = np.linalg.solve(Sigma, diff.T).T
    quad = np.sum(diff * sol, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def geron_gaussian_mixture(X, n_components=2, seed=0, max_iter=100, tol=1e-6, reg=1e-6):
    """
    Gaussian mixture model fit via EM.

    Formula: p(x) = sum_k pi_k N(x; mu_k, Sigma_k)

    Expectation-maximisation, implemented in full:

    * **E step**: responsibilities ``r_ik = pi_k N(x_i; mu_k, S_k) /
      sum_j (...)``, computed in log space with a max-shift so no
      component underflows to a zero denominator;
    * **M step**: closed-form re-estimates ``pi_k = mean_i r_ik``,
      ``mu_k = sum_i r_ik x_i / sum_i r_ik`` and the responsibility-
      weighted covariance, with ``reg`` on the diagonal so a component
      that collapses onto one point cannot produce a singular matrix.

    The log-likelihood is guaranteed non-decreasing under EM; that is not
    assumed here but checked, and reported as ``monotone``. Means are
    seeded deterministically from an LCG-chosen subset of the data, so a
    run reproduces exactly.

    Parameters
    ----------
    X : array-like, shape (m, d)
    n_components : int, default 2
    seed : int, default 0
    max_iter : int, default 100
    tol : float, default 1e-6
        Convergence threshold on the log-likelihood change.
    reg : float, default 1e-6
        Diagonal covariance regularisation.

    Returns
    -------
    result : RichResult
        Keys: weights, means, covariances, responsibilities, labels,
        log_likelihood, ll_history, n_iter, converged, monotone,
        estimate, n, method.

    Examples
    --------
    Two well-separated clusters are recovered, and the responsibilities
    are essentially hard:

    >>> X = [[0.0], [0.2], [0.1], [10.0], [10.2], [9.9]]
    >>> r = geron_gaussian_mixture(X, n_components=2, seed=1)
    >>> sorted(round(m[0], 1) for m in r["means"])
    [0.1, 10.0]
    >>> [round(w, 6) for w in sorted(r["weights"])]
    [0.5, 0.5]
    >>> len(set(r["labels"]))
    2
    >>> max(r["responsibilities"][0]) > 0.999
    True

    EM never decreases the log-likelihood:

    >>> r["monotone"]
    True
    >>> r["converged"]
    True

    A single component is just a fitted Gaussian, so its mean is the
    sample mean:

    >>> r2 = geron_gaussian_mixture([[1.0], [3.0]], n_components=1)
    >>> round(r2["means"][0][0], 12)
    2.0
    >>> r2["weights"]
    [1.0]

    References
    ----------
    Géron Ch 8
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_gaussian_mixture: X must be a non-empty (m, d) array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_gaussian_mixture: X contains non-finite values")
    m, d = A.shape
    K = int(n_components)
    if K < 1:
        raise ValueError(f"geron_gaussian_mixture: n_components must be >= 1, got {n_components!r}")
    if K > m:
        raise ValueError(f"geron_gaussian_mixture: {K} components cannot be fitted to {m} points")
    T = int(max_iter)
    if T < 1:
        raise ValueError(f"geron_gaussian_mixture: max_iter must be >= 1, got {max_iter!r}")
    rg = float(reg)
    if rg < 0:
        raise ValueError(f"geron_gaussian_mixture: reg must be non-negative, got {reg!r}")

    # Deterministic seeding: pick K distinct points via the LCG.
    s = int(seed) % 2**32
    chosen = []
    while len(chosen) < K:
        s = (1664525 * s + 1013904223) % 2**32
        i = int(((s + 0.5) / 2**32) * m) % m
        if i not in chosen:
            chosen.append(i)
    mu = A[chosen].copy()
    # Hard-assign to the seeds once, so the initial covariances are
    # within-cluster rather than global -- a single global covariance makes
    # every component identical and EM stops at that fixed point.
    d2 = np.stack([np.sum((A - mu[k]) ** 2, axis=1) for k in range(K)], axis=1)
    hard = d2.argmin(axis=1)
    cov0 = np.cov(A.T, bias=True).reshape(d, d) + rg * np.eye(d)
    Sig = np.empty((K, d, d))
    pi = np.empty(K)
    for k in range(K):
        sel = A[hard == k]
        pi[k] = max(sel.shape[0], 1) / m
        if sel.shape[0] > 1:
            mu[k] = sel.mean(axis=0)
            Sig[k] = np.cov(sel.T, bias=True).reshape(d, d) + rg * np.eye(d)
        else:
            Sig[k] = cov0 / max(K, 1)
        if not np.all(np.isfinite(Sig[k])) or np.linalg.det(Sig[k]) <= 0:
            Sig[k] = np.eye(d) * (float(np.trace(cov0)) / d / max(K, 1) + rg)
    pi = pi / pi.sum()

    ll_hist = []
    converged = False
    it = 0
    R = np.full((m, K), 1.0 / K)
    for it in range(1, T + 1):
        logp = np.stack([np.log(pi[k] + 1e-300) + gmm_log_pdf(A, mu[k], Sig[k]) for k in range(K)], axis=1)
        mx = logp.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(axis=1))
        R = np.exp(logp - lse[:, None])
        ll = float(np.sum(lse))
        ll_hist.append(ll)

        Nk = R.sum(axis=0)
        if np.any(Nk <= 0):
            raise ValueError("geron_gaussian_mixture: a component lost all responsibility; try fewer components")
        pi = Nk / m
        mu = (R.T @ A) / Nk[:, None]
        for k in range(K):
            diff = A - mu[k]
            Sig[k] = (R[:, k][:, None] * diff).T @ diff / Nk[k] + rg * np.eye(d)

        if len(ll_hist) > 2 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            converged = True
            break

    mono = all(ll_hist[i + 1] >= ll_hist[i] - 1e-8 for i in range(len(ll_hist) - 1))

    return RichResult(
        title="Gaussian mixture (EM)",
        summary_lines=[("Components", K), ("Log-likelihood", ll_hist[-1]), ("Iterations", it)],
        interpretation="EM never decreases the log-likelihood, but it does converge to a local optimum that depends on the seed.",
        payload={
            "weights": pi.tolist(),
            "means": mu.tolist(),
            "covariances": Sig.tolist(),
            "responsibilities": R.tolist(),
            "labels": R.argmax(axis=1).astype(int).tolist(),
            "log_likelihood": float(ll_hist[-1]),
            "ll_history": ll_hist,
            "n_iter": int(it),
            "converged": bool(converged),
            "monotone": bool(mono),
            "n_components": K,
            "estimate": float(ll_hist[-1]),
            "n": int(m),
            "method": "Gaussian mixture fitted by EM with log-space responsibilities",
        },
    )


def cheatsheet():
    return "hmgmm: Gaussian mixture model fit via EM"


# compact alias per ledger/NAMING.md
gmmlogpdf = gmm_log_pdf
