# morie.fn -- function file (rootcoder007/morie)
"""Rate of convergence of smoothed maximum-score estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_sms_rate"]


def horowitz_sms_rate(n, smoothness_order=2):
    r"""Rate of convergence of the smoothed maximum-score estimator
    (Horowitz Sec. 4.3.3):

    .. math:: \|\hat\beta - \beta\| = O_p\big(n^{-s/(2s+1)}\big),

    where :math:`s \ge 2` is the order of smoothness imposed on the
    kernel K.

    The exponent is not asserted, it is derived from the theorem's
    own normalisation. Theorem 4.8 states
    :math:`(nh_n)^{1/2}(\tilde b_n - \tilde\beta) \to_D N(\cdot)`
    under :math:`nh_n^{2s+1} \to \lambda`, so
    :math:`h_n \propto n^{-1/(2s+1)}` and

    .. math:: (nh_n)^{1/2} = \big(n \cdot n^{-1/(2s+1)}\big)^{1/2}
              = n^{s/(2s+1)}.

    Two consequences that matter more than the exponent itself:

    * smoothing BUYS a rate. The unsmoothed maximum-score estimator
      converges at :math:`n^{-1/3}` with a non-normal (Chernoff)
      limit; the smoothed one reaches :math:`n^{-s/(2s+1)}`, already
      faster at s = 2 (2/5 against 1/3) and approaching
      :math:`n^{-1/2}` as s grows;
    * it never ATTAINS :math:`n^{-1/2}`. The limit is a supremum over
      smoothness, not a value any finite s achieves, so a smoothed
      maximum-score estimator is never root-n consistent however
      smooth the kernel.

    Parameters
    ----------
    n : int
        Sample size, at least 2.
    smoothness_order : int, default 2
        The order s in the theorem; the book requires s >= 2.

    Returns
    -------
    RichResult
        keys: ``rate``, ``exponent``, ``bandwidth_exponent``,
        ``unsmoothed_rate``, ``unsmoothed_exponent`` (-1/3),
        ``ratio_to_unsmoothed``, ``attains_root_n`` (False),
        ``smoothness_order``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.3.3 (the smoothed maximum-score
    estimator) and Theorem 4.8; Horowitz (1992).
    """
    nn = int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    s = int(smoothness_order)
    if s < 2:
        raise ValueError(
            f"the theorem requires a smoothness order of at least 2, got {s}.")
    expo = -s / (2.0 * s + 1.0)
    return RichResult(payload={
        "rate": float(nn ** expo), "exponent": float(expo),
        "bandwidth_exponent": float(-1.0 / (2.0 * s + 1.0)),
        "unsmoothed_rate": float(nn ** (-1.0 / 3.0)),
        "unsmoothed_exponent": -1.0 / 3.0,
        "ratio_to_unsmoothed": float(nn ** expo / nn ** (-1.0 / 3.0)),
        "attains_root_n": False,
        "smoothness_order": s, "n": nn,
        "method": "n^{-s/(2s+1)} from (n h_n)^{1/2} with n h_n^{2s+1} -> lambda"})


def cheatsheet():
    return "hrzsmsrc: smoothing beats n^{-1/3} but never reaches n^{-1/2} for any finite s"
