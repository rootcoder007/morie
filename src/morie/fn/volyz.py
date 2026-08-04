# morie.fn -- function file (rootcoder007/morie)
"""Yang-Zhang OHLC volatility."""

from . import _array_core as np

from ._richresult import RichResult
from .volrs import _ohlc

__all__ = ["vol_yang_zhang"]


def vol_yang_zhang(o, h, l, c):
    r"""Yang-Zhang combined estimator.

    .. math:: \hat\sigma^2_{YZ} = \hat\sigma^2_{overnight}
              + k\,\hat\sigma^2_{open\text{-}close}
              + (1 - k)\,\hat\sigma^2_{RS},
              \qquad k = \frac{0.34}{1.34 + \tfrac{n+1}{n-1}},

    combining the overnight (close-to-open) variance, the open-to-
    close variance, and the Rogers-Satchell term with the weight k
    that minimises the estimator's variance. Drift-independent AND
    robust to opening jumps, with the smallest variance in its class.

    Parameters
    ----------
    o, h, l, c : array-like, shape (n,), n >= 2
        Daily open, high, low, close.

    Returns
    -------
    RichResult
        keys: ``sigma2`` (per-day YZ variance), ``sigma``,
        ``components`` dict (overnight, open_close, rs), ``k``, ``n``,
        ``method``.

    References
    ----------
    Yang, D. & Zhang, Q. (2000). Drift-independent volatility
    estimation based on high, low, open, and close prices. *The
    Journal of Business*, 73(3), 477-492.
    """
    o, h, l, c = _ohlc(o, h, l, c)
    n = o.size
    if n < 2:
        raise ValueError("need at least 2 days for the overnight component.")

    ov = np.log(o[1:] / c[:-1])  # overnight
    oc = np.log(c / o)  # open-to-close
    s2_ov = float(ov.var(ddof=1)) if ov.size > 1 else 0.0
    s2_oc = float(oc.var(ddof=1))
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    s2_rs = float(rs.mean())

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    s2 = s2_ov + k * s2_oc + (1 - k) * s2_rs

    return RichResult(
        payload={
            "sigma2": float(s2),
            "sigma": float(np.sqrt(max(s2, 0.0))),
            "components": {"overnight": s2_ov, "open_close": s2_oc, "rs": s2_rs},
            "k": float(k),
            "n": int(n),
            "method": "Yang-Zhang OHLC variance (overnight + k*OC + (1-k)*RS)",
        }
    )


def cheatsheet():
    return "volyz: s2_ov + k s2_oc + (1-k) s2_RS, k = 0.34/(1.34 + (n+1)/(n-1)) (YZ 2000)"


# compact alias per ledger/NAMING.md
volyangzhang = vol_yang_zhang
