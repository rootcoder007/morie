# morie.fn -- function file (rootcoder007/morie)
"""Differentially private logistic regression."""

from __future__ import annotations

from . import _array_core as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_logistic"]


def dp_logistic(X, y, epsilon=1.0, method="objective", lam=0.01, C=1.0,
                n_iter=100, lr=0.1, seed=None):
    r"""Fit logistic regression under differential privacy.

    Two mechanisms, with genuinely different behaviour.

    ``"objective"`` -- objective perturbation. A random linear term
    :math:`b^\top \theta` is added to the regularised objective before
    optimisation, and the *minimiser* of the perturbed objective is released.
    Its accuracy is far better than output perturbation because the noise is
    absorbed by the optimisation rather than added to the answer, but it
    requires the objective to be strongly convex, which is why
    :math:`\lambda > 0` is not optional here.

    ``"output"`` -- fit non-privately, then add noise to the coefficients.
    Simple, and reliably worse.

    Rows are clipped to L2 norm ``C``, which is what bounds the sensitivity of
    the gradient. Regularisation is doing double duty: it makes the problem
    strongly convex *and* it lowers sensitivity, so a larger :math:`\lambda`
    buys accuracy under privacy even where it would cost accuracy without.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Binary 0/1 response.
    epsilon : float
        Privacy budget.
    method : {"objective", "output"}
        Mechanism.
    lam : float
        L2 regularisation, must be positive for ``"objective"``.
    C : float
        Row-norm clipping bound.
    n_iter, lr : int, float
        Optimiser controls.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``beta``, ``prob``, ``accuracy``, ``method``, ``epsilon``,
        ``clipped_fraction``.

    References
    ----------
    Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011). Differentially
        private empirical risk minimization. *JMLR*, 12, 1069-1109.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    At a workable budget the private fit recovers the sign and beats chance.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(2000, 2))
    >>> y = (rng.random(2000) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1])))).astype(float)
    >>> r = dp_logistic(X, y, epsilon=10.0, seed=1)
    >>> bool(r["beta"][0] > 0 and r["beta"][1] < 0 and r["accuracy"] > 0.65)
    True

    Objective perturbation beats output perturbation at the same budget --
    the reason to prefer it.

    >>> obj = dp_logistic(X, y, epsilon=1.0, method="objective", seed=2)["accuracy"]
    >>> out = dp_logistic(X, y, epsilon=1.0, method="output", seed=2)["accuracy"]
    >>> bool(obj >= out)
    True

    Objective perturbation needs strong convexity, so a zero penalty is
    refused rather than silently producing an invalid guarantee.

    >>> dp_logistic(X, y, epsilon=1.0, method="objective", lam=0.0)
    Traceback (most recent call last):
        ...
    ValueError: objective perturbation requires lam > 0 for strong convexity
    """
    epsilon, _ = check_budget(epsilon)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be 0/1")
    if method not in ("objective", "output"):
        raise ValueError('method must be "objective" or "output"')
    if method == "objective" and lam <= 0:
        raise ValueError("objective perturbation requires lam > 0 for strong convexity")

    n = y.size
    norms = np.linalg.norm(X, axis=1)
    Xc = X * np.minimum(1.0, C / np.maximum(norms, 1e-12))[:, None]
    rng = np.random.default_rng(seed)
    p = Xc.shape[1]

    # Work on UNIT-norm rows so Chaudhuri et al.'s constants apply
    # verbatim (the paper assumes ||x|| <= 1); coefficients are mapped
    # back to the clipped-feature scale at the end.
    Xu = Xc / C

    def draw_b(scale):
        # Density (4) of the paper: nu(b) proportional to exp(-||b||/scale)
        # -- a random DIRECTION with a Gamma(p, scale) NORM, not
        # per-coordinate Laplace.
        direction = rng.normal(size=p)
        direction /= max(np.linalg.norm(direction), 1e-12)
        return direction * rng.gamma(p, scale)

    b = np.zeros(p)
    extra_reg = 0.0
    if method == "objective":
        # Algorithm 2 exactly. c bounds |l''| for logistic loss: 1/4.
        # The privacy budget must first pay the slack
        # log(1 + 2c/(n lam) + c^2/(n^2 lam^2)); if nothing is left,
        # the regulariser is raised by Delta and eps' = eps/2.
        # The noise scale is 2/eps' -- NO 1/n: the 1/n lives in the
        # objective term b'theta/n, and putting it in both places
        # (the previous code) under-noised by a factor of n and voided
        # the stated epsilon.
        c_s = 0.25
        slack = np.log(1.0 + 2.0 * c_s / (n * lam)
                       + c_s * c_s / (n * n * lam * lam))
        if epsilon > slack:
            eps_p = epsilon - slack
        else:
            extra_reg = c_s / (n * (np.exp(epsilon / 4.0) - 1.0)) - lam
            extra_reg = max(extra_reg, 0.0)
            eps_p = epsilon / 2.0
        b = draw_b(2.0 / eps_p)

    beta = np.zeros(p)
    for _ in range(int(n_iter)):
        mu = 1.0 / (1.0 + np.exp(-np.clip(Xu @ beta, -500, 500)))
        grad = Xu.T @ (mu - y) / n + (lam + extra_reg) * beta + b / n
        beta = beta - lr * grad

    if method == "output":
        # Algorithm 1: sensitivity 2/(n lam) under unit-norm rows, noise
        # drawn from density (4) with beta = n lam eps / 2.
        beta = beta + draw_b(2.0 / (n * lam * epsilon))

    beta = beta / C  # back to the clipped-feature scale
    prob = 1.0 / (1.0 + np.exp(-np.clip(Xc @ beta, -500, 500)))
    return RichResult(
        title=f"DP logistic regression ({method})",
        summary_lines=[("epsilon", epsilon), ("method", method),
                       ("accuracy", float(np.mean((prob >= 0.5) == y)))],
        payload={
            "beta": beta, "prob": prob,
            "accuracy": float(np.mean((prob >= 0.5) == y)),
            "clipped_fraction": float(np.mean(norms > C)),
            "method_used": method, "epsilon": epsilon, "lam": float(lam),
            "C": float(C), "n": int(n), "method": "dp_logistic",
        },
    )


def cheatsheet():
    return "dplog: objective perturbation beats output perturbation; regularisation buys BOTH convexity and sensitivity"


# compact alias per ledger/NAMING.md
dplogistic = dp_logistic
