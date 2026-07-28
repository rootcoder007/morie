# morie.fn -- function file (rootcoder007/morie)
"""First cumulative survival function estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_cumulative_survival_1"]


def fauzi_cumulative_survival_1(x, t_grid, h=None, transform="log"):
    r"""First cumulative survival function estimator
    (Fauzi Eqs. 4.8-4.9):

    .. math:: \tilde{\mathbb S}_{X,1}(t)
              = \frac1n\sum_{i=1}^{n}
              \mathbb V_{1,h}\big(g^{-1}(t), g^{-1}(X_i)\big),
              \qquad
              \mathbb V_{1,h}(x,y)
              = \int_x^{\infty} g'(z)\,
                V\!\left(\frac{z - y}{h}\right)dz .

    Note the direction: the inner integral runs from :math:`x` to
    infinity and the kernel argument is :math:`(z - y)/h`. The
    SECOND estimator (:mod:`morie.fn.fzcs2`) integrates from minus
    infinity to :math:`y` with argument :math:`(x - z)/h` -- the two
    are mirror images, and that asymmetry is the whole difference
    between them.

    This one is the estimator to prefer when the analytic
    relationship matters: Remark 4.2 notes that
    :math:`d\tilde{\mathbb S}_{X,1}/dt = -\tilde S_X(t)`, so the
    first pair preserves exactly the relationship the theoretical
    :math:`\mathbb S_X` and :math:`S_X` have. The second pair does
    not, though it is statistically equivalent.

    Parameters
    ----------
    x : array-like
        Sample.
    t_grid : array-like
        Evaluation points inside the support.
    h : float, optional
        Bandwidth on the transformed scale.
    transform : {"log", "identity"}
        The bijection.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``S_cumulative``, ``S_survival``,
        ``bandwidth``, ``preserves_derivative_relation`` (True),
        ``bias_coefficient`` ("b_2"), ``n``, ``method``.
    References
    ----------
    Fauzi, R. R. and Maesono, Y. *Statistical Inference Based on
    Kernel Distribution Function Estimators*. Springer, 2023.
    Eqs. (4.8)-(4.9), Theorem 4.1 and Remark 4.2. Transcribed from the PDF: the distilled text file in the
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
    # V_{1,h}(x, y) = int_x^inf g'(z) V((z - y)/h) dz, by quadrature
    upper = float(np.max(zx) + 6 * hh)
    S_cum = np.empty(tg.size)
    for j, zv in enumerate(zt):
        zz = np.linspace(zv, upper, 400)
        integ = tr["dg"](zz)[:, None] * kernel_V((zz[:, None] - zx[None, :]) / hh)
        S_cum[j] = float(np.trapezoid(integ, zz, axis=0).mean())
    S_surv = kernel_V((zt[:, None] - zx[None, :]) / hh).mean(axis=1)
    return RichResult(payload={
        "t_grid": tg, "S_cumulative": S_cum, "S_survival": S_surv,
        "bandwidth": hh, "preserves_derivative_relation": True,
        "bias_coefficient": "b_2 (4.15)",
        "mirror_note": "V_1 integrates x to infinity with argument (z - y)/h; "
                       "V_2 integrates minus infinity to y with (x - z)/h",
        "n": int(n),
        "method": "First cumulative survival estimator (4.8); d/dt gives -S_tilde exactly"})


def cheatsheet():
    return "fzcs1: preserves d/dt S_cum = -S exactly -- prefer it when that relation matters"
