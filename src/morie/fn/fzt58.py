# morie.fn -- function file (rootcoder007/morie)
"""Mean-square equivalence of the smoothed and ordinary tests (Theorem 5.8)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["smthconv", "fauzi_thm5_8_smoothed_convergence"]


def smthconv(d, c=1.0, n=None, zstd=None, zsmooth=None):
    r"""Mean-square equivalence of the smoothed and ordinary tests (Theorem 5.8).

    Theorem 5.8: if :math:`f'` exists and is continuous near
    :math:`-\theta`, the bandwidth is :math:`h_n = cn^{-d}` with
    :math:`c>0` and :math:`\tfrac14 < d < \tfrac12`, the kernel is
    symmetric, and two limiting moment conditions hold, then

    .. math::
        \lim_{n\to\infty}E_\theta\Big[
        \frac{S - E_\theta(S)}{\sqrt{V_\theta(S)}}
        - \frac{\tilde S - E_\theta(\tilde S)}
               {\sqrt{V_\theta(\tilde S)}}\Big]^2 &= 0, \\
        \lim_{n\to\infty}E_\theta\Big[
        \frac{W - E_\theta(W)}{\sqrt{V_\theta(W)}}
        - \frac{\tilde W - E_\theta(\tilde W)}
               {\sqrt{V_\theta(\tilde W)}}\Big]^2 &= 0.

    Convergence in MEAN SQUARE of the standardised difference, which is
    stronger than convergence in probability and is what licenses the
    conclusion that Pitman efficiencies coincide. Smoothing costs nothing
    asymptotically.

    The bandwidth window :math:`\tfrac14 < d < \tfrac12` is the operative
    restriction and this routine checks it. The lower end is the same
    :math:`n^{-1/4}` undersmoothing threshold as (3.8) and Theorem 5.7;
    the upper end, :math:`d<\tfrac12`, is new and says the bandwidth may
    not shrink so fast that the smoothing does nothing --
    :math:`nh_n\to\infty` in disguise.

    Given the two standardised statistics it also returns their squared
    difference, the finite-``n`` version of the quantity the theorem sends
    to zero.

    Parameters
    ----------
    d : float
        The bandwidth exponent in ``h_n = c n^{-d}``.
    c : float, default 1.0
        The bandwidth constant; must be positive.
    n : int, optional
        Sample size, used to report ``h``.
    zstd, zsmooth : float, optional
        The two standardised statistics, for the finite-``n`` squared
        difference.

    Returns
    -------
    RichResult
        Keys ``ok``, ``d``, ``h``, ``sqdiff``, ``lower``, ``upper``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.8.
    """
    d = float(d)
    c = float(c)
    if c <= 0:
        raise ValueError(f"the bandwidth constant must be positive, got {c}.")
    lower = bool(d > 0.25)
    upper = bool(d < 0.5)
    if n is None:
        h = np.nan
    else:
        nn = int(n)
        if nn < 1:
            raise ValueError(f"sample size must be at least 1, got {nn}.")
        h = c * float(nn) ** (-d)
    if zstd is None or zsmooth is None:
        sqdiff = np.nan
    else:
        sqdiff = (float(zstd) - float(zsmooth)) ** 2
    return RichResult(
        payload={
            "ok": bool(lower and upper),
            "d": d,
            "h": float(h),
            "sqdiff": float(sqdiff),
            "lower": lower,
            "upper": upper,
            "method": "mean-square equivalence of smoothed and ordinary tests (Theorem 5.8)",
        }
    )


fauzi_thm5_8_smoothed_convergence = smthconv


def cheatsheet():
    return "fzt58: Thm 5.8: mean-square equivalence, valid only for 1/4 < d < 1/2"


# CANONICAL TEST
# >>> r = smthconv(d=1 / 3, n=1000)
# >>> r['ok']
# True
