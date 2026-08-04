# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SGD update using single random sample."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_sgd_update"]


def geron_sgd_update(X, y, theta, eta=0.1, seed=0, index=None):
    """
    SGD update using single random sample.

    Formula: theta <- theta - eta_t * 2 x^(i) (x^(i)^T theta - y^(i))

    One stochastic step of least-squares regression. The factor 2 is not
    cosmetic: the gradient of ``(x^T theta - y)^2`` really is
    ``2 x (x^T theta - y)``, and dropping it silently halves the learning
    rate. The sample is chosen by a deterministic LCG (or pinned with
    `index`), and the full-batch gradient is returned alongside so the
    variance of the stochastic estimate is visible.

    Parameters
    ----------
    X : array-like
        Design matrix (n, d).
    y : array-like
        Targets, length n.
    theta : array-like
        Current parameters, length d.
    eta : float, default 0.1
        Learning rate (> 0).
    seed : int, default 0
        LCG seed used to pick the sample.
    index : int, optional
        Pin the sample instead of drawing it.

    Returns
    -------
    result : RichResult
        Keys: theta, gradient, batch_gradient, residual, index,
        estimate, n, method.

    Examples
    --------
    x = (1, 2), y = 1, theta = 0: the residual is -1, so the gradient is
    2*(1, 2)*(-1) = (-2, -4) and theta moves to (0.2, 0.4) with eta = 0.1.

    >>> r = geron_sgd_update([[1.0, 2.0]], [1.0], [0.0, 0.0], eta=0.1)
    >>> [round(float(v), 12) for v in r["gradient"]]
    [-2.0, -4.0]
    >>> [round(float(v), 12) for v in r["theta"]]
    [0.2, 0.4]
    >>> round(float(r["residual"]), 12)
    -1.0

    Pinning the index picks that row exactly:

    >>> r2 = geron_sgd_update([[1.0, 0.0], [0.5, 3.0]], [1.0, 3.0], [0.0, 0.0], eta=0.1, index=1)
    >>> [round(float(v), 12) for v in r2["gradient"]]
    [-3.0, -18.0]

    References
    ----------
    Géron Ch 4
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_sgd_update: X must be a non-empty (n, d) design matrix")
    ya = np.asarray(y, dtype=float).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_sgd_update: X has {Xa.shape[0]} rows but y has {ya.size} targets")
    th = np.asarray(theta, dtype=float).ravel()
    if th.size != Xa.shape[1]:
        raise ValueError(f"geron_sgd_update: theta has {th.size} entries but X has {Xa.shape[1]} features")
    for nm, A in (("X", Xa), ("y", ya), ("theta", th)):
        if not np.all(np.isfinite(A)):
            raise ValueError(f"geron_sgd_update: {nm} contains non-finite values")
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_sgd_update: eta must be positive and finite, got {lr}")

    n = Xa.shape[0]
    if index is None:
        s = (1664525 * (int(seed) % 2**32) + 1013904223) % 2**32
        i = int(((s + 0.5) / 2**32) * n)
        i = min(i, n - 1)
    else:
        i = int(index)
        if not (0 <= i < n):
            raise ValueError(f"geron_sgd_update: index {i} is outside 0..{n - 1}")

    resid = float(Xa[i] @ th - ya[i])
    grad = 2.0 * Xa[i] * resid
    theta_next = th - lr * grad
    full_resid = Xa @ th - ya
    batch_grad = 2.0 / n * (Xa.T @ full_resid)

    return RichResult(
        title="Stochastic gradient descent step",
        summary_lines=[
            ("Sample used", i),
            ("Residual", resid),
            ("Step L2 norm", float(np.linalg.norm(lr * grad))),
        ],
        interpretation=(
            "One sample gives an unbiased but noisy gradient; the noise is what lets SGD escape "
            "shallow local minima, and it is why the learning rate must decay for convergence."
        ),
        payload={
            "theta": theta_next,
            "theta_next": theta_next,
            "gradient": grad,
            "batch_gradient": batch_grad,
            "residual": resid,
            "index": int(i),
            "eta": lr,
            "estimate": float(np.linalg.norm(lr * grad)),
            "n": int(n),
            "method": "Single-sample least-squares SGD update theta <- theta - eta*2x(x^T theta - y)",
        },
    )


def cheatsheet():
    return "hmsgdu: SGD update using single random sample"


# compact alias per ledger/NAMING.md
geronsgdupdate = geron_sgd_update
