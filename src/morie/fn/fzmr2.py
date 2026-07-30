# morie.fn -- function file (rootcoder007/morie)
"""Second boundary-free MRL estimator m_tilde_X,2."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_mrl_boundary_free_2", "fauzi_mrl_est2"]


def fauzi_mrl_boundary_free_2(x, t_grid, h=None, transform="log"):
    r"""Second boundary-free mean-residual-life estimator
    (Fauzi Eq. 4.24):

    .. math:: \tilde m_{X,2}(t)
              = \frac{\tilde{\mathbb S}_{X,2}(t)}{\tilde S_X(t)}
              = \frac{\sum_i \int_{-\infty}^{g^{-1}(X_i)}
                       g'(z)V\!\left(\frac{g^{-1}(t)-z}{h}\right)dz}
                     {\sum_i V\!\left(
                       \frac{g^{-1}(t)-g^{-1}(X_i)}{h}\right)} .

    The ratio of the second cumulative survival estimator to the
    survival estimator. Theorem 4.3 gives its bias as
    :math:`\frac{h^2}{2S_X(t)}[b_3(t) + m_X(t)b_1(t)]\int y^2K(y)dy
    + o(h^2)` -- second order, and second order EVERYWHERE, which is
    the entire payoff over :mod:`morie.fn.fzmrln`.

    Remark 4.4 adds a detail worth having: because
    :math:`\tilde S_X(a_1) = 1` and :math:`\tilde S_X(a_2) = 0`, the
    variances vanish AT the boundaries, so the transformed estimators
    are not merely unbiased there but stable.

    The book's own advice, from Remark 4.2, is to prefer
    :math:`\tilde m_{X,1}` when the analytic relationship between the
    survival and cumulative survival functions must be preserved;
    the two are statistically equivalent otherwise.

    Parameters
    ----------
    x : array-like
        Sample strictly inside the support.
    t_grid : array-like
        Evaluation points.
    h : float, optional
        Bandwidth on the transformed scale.
    transform : {"log", "identity"}
        The bijection.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``mrl``, ``bandwidth``,
        ``bias_order`` ("O(h^2) everywhere"),
        ``variance_vanishes_at_boundary`` (True),
        ``prefer_variant_1_when``, ``n``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (4.24), Theorem 4.3 and Remarks
    4.2, 4.4, 4.5. Transcribed from the PDF -- the distilled text
    file truncates (4.24) mid-formula.
    """
    from .fzcs2 import fauzi_cumulative_survival_2

    out = fauzi_cumulative_survival_2(x, t_grid, h=h, transform=transform)
    S = out["S_survival"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mrl = np.where(S > 0, out["S_cumulative"] / np.maximum(S, 1e-300),
                       np.nan)
    return RichResult(payload={
        "t_grid": out["t_grid"], "mrl": mrl, "bandwidth": out["bandwidth"],
        "bias_order": "O(h^2) everywhere, including the boundary region",
        "bias_formula": "h^2/(2 S_X(t)) [b_3(t) + m_X(t) b_1(t)] int y^2 K(y) dy",
        "variance_vanishes_at_boundary": True,
        "prefer_variant_1_when": "the analytic relation between S and S_cum "
                                 "must be preserved (Remark 4.2)",
        "n": out["n"],
        "method": "Boundary-free MRL estimator m_tilde_{X,2} (4.24); Theorem 4.3 bias"})


def cheatsheet():
    return "fzmr2: O(h^2) bias at the BOUNDARY too, and the variance vanishes there"


#: Catalogue alias for :func:`fauzi_mrl_boundary_free_2`.
fauzi_mrl_est2 = fauzi_mrl_boundary_free_2
