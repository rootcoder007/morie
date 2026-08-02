# morie.fn -- function file (rootcoder007/morie)
"""One-step asymptotically efficient estimator for single-index model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_one_step_efficient"]


def horowitz_one_step_efficient(x, y, bandwidth=None, initial_estimator=None,
                                n_steps=1):
    r"""One Newton step from any root-n-consistent start, to
    asymptotic efficiency (Horowitz Sec. 2.6.4), equation (2.52):

    .. math:: \tilde b_n = \tilde b_n^{*} -
              \left[\frac{\partial^2 S_n(\tilde b_n^{*})}
              {\partial\tilde b\,\partial\tilde b'}\right]^{-1}
              \frac{\partial S_n(\tilde b_n^{*})}{\partial\tilde b},

    where :math:`S_n` is the weighted-NLS objective with
    :math:`W = 1/s_n^2(x)` and :math:`\tilde b_n^{*}` is ANY
    :math:`n^{-1/2}`-consistent estimator. The result attains
    :math:`\Omega_{SI}`.

    The word doing the work is **one**. This is not an iterative
    optimiser stopped early: the argument at (2.53)-(2.55) shows the
    single step already removes the leading term,

    .. math:: n^{1/2}(\tilde b_n - \tilde\beta)
              = -C^{-1}n^{1/2}\frac{\partial S_n(\tilde\beta)}
              {\partial\tilde b} + o_p(1),

    so iterating to convergence buys nothing asymptotically. That
    matters practically because the direct estimators of
    Sec. 2.6.1-2.6.3 are fast while minimising :math:`S_n` is slow
    and its objective may be nonconvex or multimodal -- so a cheap
    direct start plus one step beats a full optimisation.

    ``n_steps`` is exposed for experimentation, but the theory needs
    only one; ``theory_requires_steps`` records that.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    bandwidth : float, optional
        Bandwidth for the index regression; Silverman's rule
        otherwise.
    initial_estimator : array-like, optional
        The root-n-consistent start :math:`\tilde b_n^{*}`. Defaults
        to the density-weighted average derivative of
        :mod:`morie.fn.hrzade`, a direct estimator, which is exactly
        the intended use.
    n_steps : int, default 1
        Newton steps to take.

    Returns
    -------
    RichResult
        keys: ``beta``, ``beta_initial``, ``step``, ``omega``,
        ``se``, ``attains_omega_SI`` (True),
        ``theory_requires_steps`` (1), ``n_steps``, ``bandwidth``,
        ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 2.6.4 (one-step asymptotically
    efficient estimators), eqs. (2.52)-(2.55).
    """
    from ._horowitz import nw_regression, silverman_bw

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    if d < 2:
        raise ValueError(f"need at least 2 covariates, got {d}.")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {steps}.")

    if initial_estimator is None:
        from .hrzade import hrz_average_derivative
        delta = hrz_average_derivative(X, yv)["delta"]
        b0 = np.atleast_1d(np.asarray(delta, dtype=float)).ravel()
    else:
        b0 = np.asarray(initial_estimator, dtype=float).ravel()
    if b0.size != d:
        raise ValueError(f"initial_estimator has {b0.size} entries for {d}.")
    if b0[0] == 0:
        raise ValueError("the scale normalisation needs a nonzero first "
                         "coefficient in the initial estimator.")
    b0 = b0 / abs(b0[0])
    b_init = b0.copy()

    hh = float(silverman_bw(X @ b0)) if bandwidth is None else float(bandwidth)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    Xt = X[:, 1:]
    b = b0.copy()
    grad = hess = None
    for _ in range(steps):
        z = X @ b
        Ghat = nw_regression(z, yv, grid=z, h=hh)[1]
        resid = yv - Ghat
        s2 = np.maximum(
            nw_regression(z, resid**2, grid=z, h=hh)[1], 1e-12)
        w = 1.0 / s2
        o = np.argsort(z)
        gp = np.zeros(n)
        gp[o] = np.gradient(Ghat[o], z[o])
        Xbar = np.column_stack([
            nw_regression(z, Xt[:, j], grid=z, h=hh)[1]
            for j in range(d - 1)])
        dG = gp[:, None] * (Xt - Xbar)
        # S_n = (1/n) sum W_i [Y_i - G(X_i'b)]^2
        grad = -2.0 * (dG * (w * resid)[:, None]).sum(axis=0) / n
        hess = 2.0 * (dG * w[:, None]).T @ dG / n
        b = np.r_[1.0, b[1:] - np.linalg.pinv(hess) @ grad]

    omega = np.linalg.pinv(hess)
    return RichResult(payload={
        "beta": b, "beta_initial": b_init,
        "step": b[1:] - b_init[1:],
        "omega": omega, "se": np.sqrt(np.maximum(np.diag(omega), 0.0) / n),
        "attains_omega_SI": True,
        "theory_requires_steps": 1, "n_steps": steps,
        "bandwidth": hh, "n": int(n), "d": int(d),
        "method": "(2.52): ONE Newton step from any root-n start attains Omega_SI; iterating adds nothing"})


def cheatsheet():
    return "hrzasym: one step, not convergence -- a cheap direct start plus one Newton step is efficient"
