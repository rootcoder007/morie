# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mini-batch gradient descent on subset of size b."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_minibatch_gd"]

_METHOD = "Mini-batch gradient descent step"


def geron_minibatch_gd(X, y, theta, eta, b, seed=0, n_steps=1):
    """
    Mini-batch gradient descent on subset of size b.

    Formula: theta <- theta - eta * (2/b) X_b^T (X_b theta - y_b)

    Batch size ``b`` interpolates between the two extremes: ``b = m`` is
    batch gradient descent (exact gradient, one step per epoch) and
    ``b = 1`` is SGD (noisiest gradient, m steps per epoch).  The
    gradient's standard error scales like ``1/sqrt(b)``, so the
    per-batch gradient and the full-data gradient are both returned --
    their angle is the noise the mini-batch is trading for speed.

    Batches are drawn without replacement from a seeded permutation, so
    each pass covers the data exactly once.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix (include a bias column yourself).
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (n,)
        Current parameters.
    eta : float
        Learning rate (positive).
    b : int
        Batch size, ``1 <= b <= m``.
    seed : int
        Seed for the shuffle.
    n_steps : int
        Number of consecutive mini-batch steps to take.

    Returns
    -------
    result : RichResult
        Keys: theta, gradient, full_gradient, batch_indices, mse,
        estimate, n, method.

    Examples
    --------
    Full batch on ``y = 2x`` starting from theta = 0: the gradient is
    ``(2/2) * X^T(X*0 - y)`` = ``-[1*2 + 2*4] = -10``, so with eta=0.01
    the step is +0.1:

    >>> r = geron_minibatch_gd([[1.0], [2.0]], [2.0, 4.0], [0.0], eta=0.01, b=2)
    >>> float(r["gradient"][0])
    -10.0
    >>> round(float(r["theta"][0]), 12)
    0.1

    At the optimum the gradient vanishes:

    >>> z = geron_minibatch_gd([[1.0], [2.0]], [2.0, 4.0], [2.0], eta=0.01, b=2)
    >>> round(float(z["gradient"][0]), 12)
    0.0

    Repeated steps drive the MSE down:

    >>> many = geron_minibatch_gd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], [0.0],
    ...                           eta=0.02, b=1, seed=1, n_steps=200)
    >>> bool(abs(float(many["theta"][0]) - 2.0) < 0.01)
    True

    References
    ----------
    Géron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_minibatch_gd: X must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(f"geron_minibatch_gd: X has {A.shape[0]} rows but y has {yy.size} entries")
    t = np.atleast_1d(np.asarray(theta, dtype=float)).ravel().copy()
    if t.size != A.shape[1]:
        raise ValueError(f"geron_minibatch_gd: theta has {t.size} entries but X has {A.shape[1]} columns")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)) or not np.all(np.isfinite(t)):
        raise ValueError("geron_minibatch_gd: X, y and theta must be finite")
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_minibatch_gd: eta must be a positive finite learning rate, got {eta!r}")
    m = A.shape[0]
    bs = int(b)
    if not (1 <= bs <= m):
        raise ValueError(f"geron_minibatch_gd: batch size must lie in 1..{m}, got {b!r}")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(f"geron_minibatch_gd: n_steps must be at least 1, got {n_steps!r}")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(m)
    cursor = 0
    grad = None
    idx = None
    for _ in range(steps):
        if cursor + bs > m:
            order = rng.permutation(m)
            cursor = 0
        idx = order[cursor : cursor + bs]
        cursor += bs
        Xb, yb = A[idx], yy[idx]
        grad = (2.0 / bs) * (Xb.T @ (Xb @ t - yb))
        t = t - lr * grad

    full_grad = (2.0 / m) * (A.T @ (A @ t - yy))
    mse = float(np.mean((A @ t - yy) ** 2))

    return RichResult(
        title="Mini-batch gradient descent",
        summary_lines=[
            ("Batch size", bs),
            ("Steps taken", steps),
            ("MSE", mse),
            ("||full gradient||", float(np.linalg.norm(full_grad))),
        ],
        interpretation=(
            "b = m is batch GD, b = 1 is SGD; the mini-batch gradient's noise falls like 1/sqrt(b)."
        ),
        payload={
            "theta": t,
            "gradient": grad,
            "full_gradient": full_grad,
            "batch_indices": np.asarray(idx),
            "mse": mse,
            "batch_size": bs,
            "estimate": mse,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmbgd: mini-batch GD theta <- theta - eta*(2/b) X_b^T(X_b theta - y_b)"
