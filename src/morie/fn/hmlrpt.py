# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear regression implemented in PyTorch."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_linreg_pytorch"]

_METHOD = "Linear regression by SGD (PyTorch listing, computed in numpy)"


def geron_linreg_pytorch(X, y, epochs=100, lr=0.01, batch_size=None, seed=0):
    """
    Linear regression implemented in PyTorch.

    Formula: y_hat = X @ w + b; MSE loss; SGD

    The book's listing is PyTorch; this package is numpy-only, so the
    *computation* is reproduced here rather than the API.  What PyTorch
    contributes to this particular example is autograd, and for an MSE
    linear model the gradient autograd would produce is known in closed
    form:

    ``dL/dw = (2/b) X_b^T (X_b w + b - y_b)``,  ``dL/db = 2 * mean(r)``

    so the numbers are identical to the listing's, not an approximation
    of them.  Weight and bias are kept separate, exactly as
    ``nn.Linear`` does -- no ones column -- which is why the bias
    gradient appears as its own term.

    The normal-equation solution is computed alongside and the gap is
    reported: SGD on a convex quadratic must approach it, and a gap that
    does not shrink means the learning rate is too large.  The stability
    limit ``2/lambda_max`` of the full-batch Hessian is returned for the
    same reason.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Features (no bias column).
    y : array-like, shape (m,)
        Targets.
    epochs : int
        Passes over the data.
    lr : float
        Learning rate.
    batch_size : int, optional
        Mini-batch size; defaults to the whole dataset (full-batch).
    seed : int
        Seed for the shuffle.

    Returns
    -------
    result : RichResult
        Keys: w, b, loss_history, final_loss, w_closed_form,
        b_closed_form, gap, lr_limit, estimate, n, method.

    Examples
    --------
    ``y = 2x + 1`` recovered by descent:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [1.0, 3.0, 5.0, 7.0]
    >>> r = geron_linreg_pytorch(X, y, epochs=2000, lr=0.05)
    >>> round(float(r["w"][0]), 4), round(float(r["b"]), 4)
    (2.0, 1.0)
    >>> bool(r["final_loss"] < 1e-8)
    True

    It agrees with the closed form, which is the check that the descent
    converged rather than merely stopped:

    >>> bool(r["gap"] < 1e-4)
    True
    >>> round(float(r["w_closed_form"][0]), 9), round(float(r["b_closed_form"]), 9)
    (2.0, 1.0)

    The loss decreases monotonically for full-batch descent below the
    stability limit:

    >>> bool(np.all(np.diff(r["loss_history"]) <= 1e-12))
    True
    >>> bool(r["lr_limit"] > 0.05)
    True

    A learning rate above the limit diverges, and that is raised rather
    than returned as a plausible number:

    >>> geron_linreg_pytorch(X, y, epochs=100, lr=10.0)
    Traceback (most recent call last):
        ...
    ValueError: geron_linreg_pytorch: the loss diverged at lr=10.0; the full-batch stability limit here is 0.23795

    References
    ----------
    Géron Ch 10
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_linreg_pytorch: X must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(f"geron_linreg_pytorch: X has {A.shape[0]} rows but y has {yy.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)):
        raise ValueError("geron_linreg_pytorch: X and y must be finite")
    n_epochs = int(epochs)
    if n_epochs < 1:
        raise ValueError(f"geron_linreg_pytorch: epochs must be at least 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_linreg_pytorch: lr must be positive and finite, got {lr!r}")
    m, n = A.shape
    bs = m if batch_size is None else int(batch_size)
    if not (1 <= bs <= m):
        raise ValueError(f"geron_linreg_pytorch: batch_size must lie in 1..{m}, got {batch_size!r}")

    Ab = np.hstack([np.ones((m, 1)), A])
    theta_cf, *_ = np.linalg.lstsq(Ab, yy, rcond=None)
    b_cf = float(theta_cf[0])
    w_cf = theta_cf[1:]

    H = (2.0 / m) * (Ab.T @ Ab)
    lam = float(np.max(np.linalg.eigvalsh(0.5 * (H + H.T))))
    limit = 2.0 / lam if lam > 0 else float("inf")

    w = np.zeros(n)
    b = 0.0
    rng = np.random.default_rng(int(seed))
    history = []
    for _ in range(n_epochs):
        order = rng.permutation(m) if bs < m else np.arange(m)
        for start in range(0, m, bs):
            idx = order[start : start + bs]
            Xb, yb = A[idx], yy[idx]
            resid = Xb @ w + b - yb
            gw = (2.0 / idx.size) * (Xb.T @ resid)
            gb = 2.0 * float(np.mean(resid))
            w = w - eta * gw
            b = b - eta * gb
        loss = float(np.mean((A @ w + b - yy) ** 2))
        if not np.isfinite(loss) or loss > 1e12 * (1.0 + float(np.mean(yy**2))):
            raise ValueError(
                f"geron_linreg_pytorch: the loss diverged at lr={eta}; the full-batch stability limit "
                f"here is {limit:.6g}"
            )
        history.append(loss)

    gap = float(np.linalg.norm(np.concatenate([[b], w]) - theta_cf))

    return RichResult(
        title="Linear regression by SGD",
        summary_lines=[
            ("Epochs", n_epochs),
            ("Batch size", bs),
            ("Final MSE", history[-1]),
            ("Gap to closed form", gap),
            ("Stability limit for lr", limit),
        ],
        warnings=(
            [f"the descent has not reached the closed-form solution (gap {gap:.4g}); raise epochs or lr."]
            if gap > 1e-3
            else []
        ),
        interpretation=(
            "The gradient here is the one autograd would produce, in closed form; the closed-form "
            "solution is the target the descent must approach."
        ),
        payload={
            "w": w,
            "b": b,
            "loss_history": np.asarray(history),
            "final_loss": history[-1],
            "w_closed_form": w_cf,
            "b_closed_form": b_cf,
            "gap": gap,
            "lr_limit": limit,
            "estimate": history[-1],
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrpt: linear regression by SGD with separate w and b, cross-checked against the normal equations"
