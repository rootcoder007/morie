# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-based learning: fit parameters theta to minimize a cost function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_model_based"]

_METHOD = "Model-based fit (least squares, closed form and by descent)"


def geron_model_based(X, y, add_bias=True, eta=None, n_iter=1000):
    """
    Model-based learning: fit parameters theta to minimize a cost function.

    Formula: theta_hat = argmin_theta J(theta; X, y)

    The counterpart to :func:`morie.fn.hmins.geron_instance_based`: the
    data are summarised into a handful of parameters and then discarded,
    so prediction costs ``O(n)`` no matter how many rows were used to
    fit.

    The cost here is the MSE and it is minimised twice, deliberately:
    once in closed form via the normal equations (``lstsq``, which is
    stable where an explicit inverse is not) and once by gradient
    descent.  Agreement between the two is the check that the descent
    converged; ``gap`` reports it. A learning rate that diverges shows up
    as a large gap rather than as a plausible-looking wrong answer.

    ``eta`` defaults to ``1/L`` with ``L`` the largest eigenvalue of
    ``(2/m) X^T X`` -- the classical stability limit for gradient
    descent on a quadratic, so the default never diverges.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Features.
    y : array-like, shape (m,)
        Targets.
    add_bias : bool
        Prepend a column of ones.
    eta : float, optional
        Learning rate for the descent check.
    n_iter : int
        Descent iterations.

    Returns
    -------
    result : RichResult
        Keys: theta, theta_gd, mse, r2, gap, eta, estimate, n, method.

    Examples
    --------
    ``y = 3 + 2x`` recovered exactly, and the two routes agree:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [3.0, 5.0, 7.0, 9.0]
    >>> r = geron_model_based(X, y)
    >>> [round(float(v), 9) for v in r["theta"]]
    [3.0, 2.0]
    >>> round(r["mse"], 12), round(r["r2"], 12)
    (0.0, 1.0)
    >>> bool(r["gap"] < 1e-6)
    True

    The fitted parameters are all that is kept -- two numbers stand in
    for the whole dataset:

    >>> r["n_parameters"]
    2

    A constant target is fitted by the intercept alone:

    >>> c = geron_model_based([[1.0], [2.0]], [5.0, 5.0])
    >>> round(float(c["theta"][0]), 9), round(float(c["theta"][1]), 9)
    (5.0, 0.0)

    References
    ----------
    Géron Ch 1
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_model_based: X must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(f"geron_model_based: X has {A.shape[0]} rows but y has {yy.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)):
        raise ValueError("geron_model_based: X and y must be finite")
    if add_bias:
        A = np.hstack([np.ones((A.shape[0], 1)), A])
    m, n = A.shape
    if m < n:
        raise ValueError(
            f"geron_model_based: {m} observations cannot determine {n} parameters; the fit is underdetermined"
        )
    iters = int(n_iter)
    if iters < 1:
        raise ValueError(f"geron_model_based: n_iter must be at least 1, got {n_iter!r}")

    theta, *_ = np.linalg.lstsq(A, yy, rcond=None)

    H = (2.0 / m) * (A.T @ A)
    lam = float(np.max(np.linalg.eigvalsh(0.5 * (H + H.T))))
    if eta is None:
        lr = 1.0 / lam if lam > 0 else 0.1
    else:
        lr = float(eta)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"geron_model_based: eta must be positive and finite, got {eta!r}")

    t = np.zeros(n)
    for _ in range(iters):
        t = t - lr * ((2.0 / m) * (A.T @ (A @ t - yy)))
        if not np.all(np.isfinite(t)):
            raise ValueError(
                f"geron_model_based: gradient descent diverged at eta={lr}; the stability limit here is "
                f"2/{lam:.6g} = {2.0 / lam if lam > 0 else float('inf'):.6g}"
            )

    resid = A @ theta - yy
    mse = float(np.mean(resid**2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    if ss_tot == 0:
        r2 = 1.0 if mse == 0 else float("-inf")
    else:
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot
    gap = float(np.linalg.norm(t - theta))

    return RichResult(
        title="Model-based fit",
        summary_lines=[
            ("Parameters", int(n)),
            ("MSE", mse),
            ("R^2", r2),
            ("Closed-form vs descent gap", gap),
        ],
        warnings=(
            [f"gradient descent has not converged to the closed-form solution (gap {gap:.4g}); "
             f"raise n_iter or check eta."]
            if gap > 1e-4
            else []
        ),
        interpretation=(
            "The data are compressed into the parameters and then no longer needed; prediction is "
            "O(n) regardless of how many rows were fitted."
        ),
        payload={
            "theta": theta,
            "theta_gd": t,
            "mse": mse,
            "r2": r2,
            "gap": gap,
            "eta": lr,
            "residuals": resid,
            "n_parameters": int(n),
            "estimate": mse,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmod: model-based least squares -- closed form and gradient descent, cross-checked"
