# morie.fn -- function file (rootcoder007/morie)
"""Detrended fluctuation analysis (Peng et al. 1994); NOT covered by Rangayyan."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_dfa"]


def rangayyan_dfa(x, scales=None, order=1):
    """DFA scaling exponent α (Peng et al. 1994).

    1. Y(k) = Σ_{i=1}^{k} (x_i − mean(x)).
    2. Partition Y into boxes of size ``n``; detrend each with a
       polynomial of order ``order``.
    3. F(n) = sqrt(mean residual variance across boxes).
    4. α = slope of log F(n) vs log n.

    Parameters
    ----------
    x : array-like
    scales : array-like of int, optional
        Box sizes; default geometric 4 .. N/4.
    order : int
        Detrending polynomial order (DFA-1 default).

    Returns
    -------
    RichResult with keys ``alpha``, ``scales``, ``F``, ``log_scales``, ``log_F``.

    References
    ----------
    Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., &
        Goldberger, A. L. (1994). Mosaic organization of DNA nucleotides.
        *Physical Review E*, 49(2), 1685-1689, method on p.1685.
        https://doi.org/10.1103/PhysRevE.49.1685  (PRIMARY -- in the library.)

    Note: this method is NOT in Rangayyan, contrary to the previous
    docstring's "Ch 7". The 2024 edition mentions "detrended fluctuation"
    four times, every one a citation to someone else's application rather
    than a treatment of the method.

    Peng's steps, verbatim in substance (p.1685):
      1. "Divide the entire sequence of length N into N/l nonoverlapping
         boxes, each containing l nucleotides, and define the 'local trend'
         in each box to be the ordinate of a linear least-squares fit."
      2. "Define the 'detrended walk' ... as the difference between the
         original walk y(n) and the local trend. Calculate the variance
         about the detrended walk for each box, and calculate the average of
         these variances over all the boxes of size l, denoted F_d^2(l)."

    So F(l) is the square root of the mean per-box residual variance -- not
    the mean of the per-box standard deviations, which is a different and
    smaller quantity. With no long-range correlation F_d(l) ~ l^(1/2), i.e.
    alpha = 1/2; alpha != 1/2 indicates power-law correlation.

    Peng integrates the raw walk; this implementation subtracts the mean
    before the cumulative sum. The two differ by a linear ramp, which
    order >= 1 detrending removes exactly, so alpha is unchanged for the
    default DFA-1 and above.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 32:
        raise ValueError("DFA needs at least 32 samples.")
    if scales is None:
        log_n = np.linspace(np.log(4), np.log(max(8, N // 4)), 12)
        scales = np.unique(np.round(np.exp(log_n)).astype(int))
        scales = scales[scales >= 4]
    scales = np.asarray(scales, dtype=int)
    order = int(order)
    if order < 0:
        raise ValueError(f"`order` must be >= 0, got {order}.")
    # A box of n points cannot support a polynomial of degree `order` unless
    # n >= order + 2; at n == order + 1 the fit is exact, every residual is
    # zero, and F(n) collapses to 0 -> log F = -inf, silently poisoning the
    # slope. numpy would only emit a RankWarning.
    too_small = scales[scales < order + 2]
    if too_small.size:
        raise ValueError(
            f"box sizes {too_small.tolist()} are too small for order={order}: "
            f"need at least {order + 2} points per box."
        )
    y = np.cumsum(x - x.mean())
    F = np.empty(scales.size, dtype=float)
    for j, n in enumerate(scales):
        nseg = N // n
        if nseg < 1:
            F[j] = np.nan
            continue
        rms = []
        for k in range(nseg):
            seg = y[k * n : (k + 1) * n]
            t = np.arange(n)
            p = np.polyfit(t, seg, order)
            trend = np.polyval(p, t)
            rms.append(np.mean((seg - trend) ** 2))
        F[j] = np.sqrt(np.mean(rms))
    mask = np.isfinite(F) & (F > 0)
    log_n = np.log(scales[mask])
    log_F = np.log(F[mask])
    # alpha is the slope of a log-log fit, so it needs at least two usable
    # scales. With one, polyfit returns an arbitrary line through a single
    # point and only emits a RankWarning -- a meaningless alpha that looks
    # like a number.
    if log_n.size < 2:
        raise ValueError(
            f"need at least 2 usable box sizes to fit a slope, got {log_n.size} "
            f"(from scales={scales.tolist()}); F must be finite and > 0 at each."
        )
    alpha, intercept = np.polyfit(log_n, log_F, 1)
    res = RichResult(
        title="Detrended Fluctuation Analysis",
        summary_lines=[("α", float(alpha)), ("Order", int(order)), ("Scales", len(scales))],
        interpretation=f"α = {alpha:.4g}. 0.5 random, 1 1/f, >1 persistent.",
        payload={"alpha": float(alpha), "scales": scales, "F": F, "log_scales": log_n, "log_F": log_F},
    )
    return with_describe_pointer(res, "rgdfa")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_dfa(rng.standard_normal(500))
# >>> 0.3 < r["alpha"] < 0.7
# True


def cheatsheet():
    return "rgdfa: detrended fluctuation analysis α -- Peng et al. (1994)"
