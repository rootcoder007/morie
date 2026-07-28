# morie.fn -- function file (rootcoder007/morie)
"""Second cumulative survival function estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_cumulative_survival_2"]


def fauzi_cumulative_survival_2(x, t_grid, h=None, transform="log"):
    r"""Second cumulative survival function estimator
    (Fauzi Eqs. 4.17-4.18):

    .. math:: \tilde{\mathbb S}_{X,2}(t)
              = \frac1n\sum_i
              \mathbb V_{2,h}\big(g^{-1}(t), g^{-1}(X_i)\big),
              \qquad
              \mathbb V_{2,h}(x,y)
              = \int_{-\infty}^{y} g'(z)\,
                V\!\left(\frac{x - z}{h}\right)dz .

    The mirror of :mod:`morie.fn.fzcs1`: integration runs from minus
    infinity up to :math:`y`, and the kernel argument is
    :math:`(x - z)/h`.

    The book is explicit that multiplying :math:`V` by :math:`g'` is
    what makes this an estimator of :math:`\mathbb S_X` at all --
    the change-of-variable property of the cumulative survival
    function is trickier than for the survival function itself, and
    the :math:`g'` factor is what carries it. Its bias coefficient is
    :math:`b_3` (4.21), which differs from the first estimator's
    :math:`b_2` by a sign and a term, and the two nonetheless have
    the SAME covariance with :math:`\tilde S_X` (Remark 4.3).

    Parameters
    ----------
    x : array-like
        Sample.
    t_grid : array-like
        Evaluation points.
    h : float, optional
        Bandwidth.
    transform : {"log", "identity"}
        The bijection.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``S_cumulative``, ``S_survival``,
        ``bandwidth``, ``preserves_derivative_relation`` (False),
        ``bias_coefficient`` ("b_3"), ``n``, ``method``.
    References
    ----------
    Fauzi, R. R. and Maesono, Y. *Statistical Inference Based on
    Kernel Distribution Function Estimators*. Springer, 2023.
    Eqs. (4.17)-(4.18), Theorem 4.2 and Remark 4.3. Transcribed from the PDF: the distilled text file in the
    reference library omits the Jacobian factor and truncates (4.24).
    """
    from ._fauzi import boundary_free_transform, kernel_V

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    tr = boundary_free_transform(transform)
    lo, hi = tr["support"]
    if np.any(xv <= lo) or np.any(xv >= hi):
        raise ValueError("the sample must lie strictly inside the support.")
    tg = np.atleast_1d(np.asarray(t_grid, dtype=float))
    if np.any(tg <= lo) or np.any(tg >= hi):
        raise ValueError("t_grid must lie strictly inside the support.")
    zx = tr["g_inv"](xv)
    zt = tr["g_inv"](tg)
    hh = float(np.std(zx, ddof=1) * n ** -0.2) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    lower = float(np.min(zx) - 6 * hh)
    S_cum = np.empty(tg.size)
    for j, zv in enumerate(zt):
        vals = np.empty(zx.size)
        for i, yy in enumerate(zx):
            zz = np.linspace(lower, yy, 200)
            vals[i] = float(np.trapezoid(
                tr["dg"](zz) * kernel_V((zv - zz) / hh), zz))
        S_cum[j] = float(vals.mean())
    S_surv = kernel_V((zt[:, None] - zx[None, :]) / hh).mean(axis=1)
    return RichResult(payload={
        "t_grid": tg, "S_cumulative": S_cum, "S_survival": S_surv,
        "bandwidth": hh, "preserves_derivative_relation": False,
        "bias_coefficient": "b_3 (4.21)",
        "g_prime_note": "multiplying V by g' is what makes this an estimator "
                        "of the cumulative survival function at all",
        "same_covariance_as_first": True,
        "n": int(n),
        "method": "Second cumulative survival estimator (4.17); mirror of the first, bias b_3"})


def cheatsheet():
    return "fzcs2: mirror limits of fzcs1; the g' factor is what makes it estimate S_cum"
