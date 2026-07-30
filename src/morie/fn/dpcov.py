# morie.fn -- function file (rootcoder007/morie)
"""Differentially private covariance matrix."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, gaussian_sigma
from ._richresult import RichResult

__all__ = ["dp_covariance"]


def dp_covariance(X, C=1.0, epsilon=1.0, delta=1e-5, seed=None, project_psd=True):
    r"""Release a covariance matrix privately by the Gaussian mechanism.

    Rows are clipped to L2 norm ``C``, which bounds one record's influence on
    :math:`X^\top X` by :math:`C^2` and so fixes the sensitivity. Symmetric
    Gaussian noise is added to the upper triangle and mirrored, since adding
    independent noise to both triangles would double the effective budget for
    nothing.

    The noisy matrix is **not** positive semi-definite in general -- noise can
    push small eigenvalues negative, and a covariance with negative
    eigenvalues will break any downstream method that factorises it. Clipping
    the eigenvalues at zero is post-processing, so it costs no privacy, and
    ``project_psd`` does it by default while ``n_negative_eigenvalues`` records
    how much repair was needed. A large count is a signal that the budget is
    too small for the dimension.

    Cost grows as :math:`p^2` entries from a fixed budget, so private
    covariance in high dimension is expensive: doubling ``p`` quadruples the
    number of noisy entries while the budget stays put.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    C : float
        Row-norm clipping bound, positive, chosen independently of the data.
    epsilon, delta : float
        Privacy budget.
    seed : int, optional
        Seed; leave ``None`` for a real release.
    project_psd : bool
        Clip negative eigenvalues to zero.

    Returns
    -------
    RichResult
        ``release``, ``raw``, ``sigma``, ``clipped_fraction``,
        ``n_negative_eigenvalues``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    The release is symmetric and, after projection, positive semi-definite --
    both required of anything calling itself a covariance.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 4)) * 0.3
    >>> r = dp_covariance(X, C=2.0, epsilon=2.0, seed=0)
    >>> bool(np.allclose(r["release"], r["release"].T))
    True
    >>> bool(np.linalg.eigvalsh(r["release"]).min() >= -1e-10)
    True

    Noise decreases as the budget grows.

    >>> lo = dp_covariance(X, 2.0, epsilon=0.1, seed=0)
    >>> hi = dp_covariance(X, 2.0, epsilon=10.0, seed=0)
    >>> bool(hi["sigma"] < lo["sigma"])
    True

    Without projection the raw release can have negative eigenvalues, which is
    what the projection exists to repair.

    >>> raw = dp_covariance(X, 2.0, epsilon=0.05, seed=1, project_psd=False)
    >>> bool(np.linalg.eigvalsh(raw["release"]).min() < 0)
    True

    >>> dp_covariance(X, C=0.0, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: C must be positive
    """
    epsilon, delta = check_budget(epsilon, delta)
    C = float(C)
    if C <= 0:
        raise ValueError("C must be positive")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    norms = np.linalg.norm(X, axis=1)
    scale = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    Xc = X * scale[:, None]
    clipped = float(np.mean(norms > C))

    S = Xc.T @ Xc / n
    sigma = gaussian_sigma(C**2 / n, epsilon, delta)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, (p, p))
    noise = np.triu(noise) + np.triu(noise, 1).T   # symmetric: one draw per pair
    raw = S + noise

    w = np.linalg.eigvalsh(raw)
    n_neg = int(np.sum(w < 0))
    if project_psd and n_neg:
        vals, vecs = np.linalg.eigh(raw)
        out = (vecs * np.clip(vals, 0, None)) @ vecs.T
        out = (out + out.T) / 2
    else:
        out = raw
    return RichResult(
        title="DP covariance",
        summary_lines=[("epsilon", epsilon), ("p", p), ("sigma", sigma),
                       ("negative eigenvalues", n_neg)],
        warnings=([f"{n_neg} of {p} eigenvalues were negative before projection; "
                   "the budget may be too small for this dimension"]
                  if n_neg > p // 2 else []),
        payload={
            "release": out, "raw": raw, "sigma": sigma,
            "clipped_fraction": clipped, "n_negative_eigenvalues": n_neg,
            "epsilon": epsilon, "delta": delta, "C": C, "n": int(n),
            "method": "dp_covariance",
        },
    )


def cheatsheet():
    return "dpcov: noise breaks PSD; projection is free post-processing, and many negatives = budget too small"
