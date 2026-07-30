# morie.fn -- function file (rootcoder007/morie)
"""Naive kernel MRL function estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_mrl_naive", "fauzi_naive_mrl"]


def fauzi_mrl_naive(x, t_grid, h=None):
    r"""Naive kernel mean-residual-life estimator (Fauzi Eq. 4.2):

    .. math:: \hat m_X(t)
              = \frac{h\sum_i \mathbb V\!\left(\frac{t-X_i}{h}\right)}
                     {\sum_i V\!\left(\frac{t-X_i}{h}\right)},
              \qquad t \in \Omega .

    The baseline the chapter improves on. It applies a symmetric
    kernel directly on the original scale, so it inherits exactly the
    boundary problem the chapter exists to solve: Remark 4.5 states
    that its bias degrades from :math:`O(h^2)` in the interior to
    :math:`O(h)` or even :math:`O(1)` near the edges, while the
    transformed estimators of :mod:`morie.fn.fzmr2` stay
    :math:`O(h^2)` throughout.

    Parameters
    ----------
    x : array-like
        Non-negative sample.
    t_grid : array-like
        Evaluation points.
    h : float, optional
        Bandwidth.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``mrl``, ``bandwidth``,
        ``interior_bias_order``, ``boundary_bias_order``,
        ``boundary_safe`` (False), ``n``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (4.2) and Remark 4.5. Transcribed
    from the PDF.
    """
    from ._fauzi import kdfe_bandwidth, kernel_V

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    tg = np.atleast_1d(np.asarray(t_grid, dtype=float))
    # a survival / cumulative-survival estimator is
    # distribution-function-type, not density-type: its variance is
    # O(1/n) - O(h/n) (Theorem 4.3), so the optimiser is a cube root
    hh = kdfe_bandwidth(xv) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    upper = float(xv.max() + 8 * hh)
    mrl = np.empty(tg.size)
    for j, t in enumerate(tg):
        den = float(np.sum(kernel_V((t - xv) / hh)))
        if den <= 0:
            mrl[j] = np.nan
            continue
        zz = np.linspace(t, upper, 400)
        num = float(np.trapezoid(
            kernel_V((zz[:, None] - xv[None, :]) / hh).sum(axis=1), zz))
        mrl[j] = num / den
    return RichResult(payload={
        "t_grid": tg, "mrl": mrl, "bandwidth": hh,
        "interior_bias_order": "O(h^2)",
        "boundary_bias_order": "O(h), and can degrade to O(1)",
        "boundary_safe": False,
        "n": int(n),
        "method": "Naive kernel MRL (4.2); the baseline whose boundary failure Ch. 4 fixes"})


def cheatsheet():
    return "fzmrln: interior O(h^2), boundary O(h) or worse -- the baseline, not the recommendation"


#: Catalogue alias for :func:`fauzi_mrl_naive`.
fauzi_naive_mrl = fauzi_mrl_naive
