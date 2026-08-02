# morie.fn -- function file (rootcoder007/morie)
"""Q-Q plot coordinates and linearity summary."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_qq_plot"]


def gibbons_qq_plot(x, F0_inv=None):
    r"""Section 4.8: the quantile-quantile plot pairs

    .. math:: \big(F_0^{-1}((i - 0.5)/n),\; X_{(i)}\big),

    theoretical quantile against observed order statistic, with the
    Hazen plotting position (i - 0.5)/n. Linearity means the family
    fits; the SLOPE and INTERCEPT of the fitted line estimate scale
    and location, so a location-scale family shows up straight even
    when the parameters are wrong -- unlike a P-P plot, which is
    parameter-bound. Q-Q resolves the tails, P-P the centre.

    Parameters
    ----------
    x : array-like
        Sample.
    F0_inv : callable, optional
        Hypothesised quantile function; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``theoretical``, ``observed`` (sorted x),
        ``correlation`` (probability-plot correlation, the Filliben
        statistic), ``slope``, ``intercept``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 4.8.

    Filliben, J. J. (1975). The probability plot correlation
    coefficient test for normality. *Technometrics*, 17(1), 111-117.
    """
    from scipy import stats

    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    pp = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(pp) if F0_inv is None else np.asarray(
        [F0_inv(p) for p in pp], dtype=float
    )
    slope, intercept = np.polyfit(q, x, 1)
    corr = float(np.corrcoef(q, x)[0, 1])
    return RichResult(
        payload={
            "theoretical": q, "observed": x, "correlation": corr,
            "slope": float(slope), "intercept": float(intercept), "n": int(n),
            "method": "Q-Q pairs (F0^{-1}((i-.5)/n), X_(i)); tail-sensitive (Ch. 4.8)",
        }
    )


def cheatsheet():
    return "gb_qq: tail-sensitive; slope/intercept = scale/location estimates"
