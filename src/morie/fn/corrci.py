"""Confidence interval for a correlation coefficient via Fisher's z (optionally Hotelling-corrected)."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["correlation_ci"]


def correlation_ci(r, n, conf_level=0.95, method="fisher"):
    r"""Confidence interval for :math:`\rho` from the z-transformation.

    Hedderich, Sachs & Reynarowych (2023, eqs 6.151-6.154):
    :math:`\dot z = \tanh^{-1} r`, :math:`s_{\dot z} = 1/\sqrt{n-3}` and the
    interval :math:`\tanh(\dot z \pm z_{1-\alpha/2} s_{\dot z})` (as
    ``cor.test``). ``method="hotelling"`` uses Hotelling's (1953) small-sample
    correction (eq 6.153) :math:`\dot z_H = \dot z - (3\dot z + r)/(4n)`,
    :math:`s_{\dot z_H} = 1/\sqrt{n-1}`.

    Parameters
    ----------
    r : float
        Sample correlation in (-1, 1).
    n : int
        Number of pairs (> 3).
    conf_level : float
    method : {"fisher", "hotelling"}

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``z``, ``se_z``.

    References
    ----------
    Hotelling, H. (1953). New light on the correlation coefficient and its
    transforms. JRSS B 15, 193-232.
    """
    r = float(r)
    n = int(n)
    if not -1 < r < 1 or n <= 3:
        raise ValueError("need -1 < r < 1 and n > 3")
    z = math.atanh(r)
    if method == "fisher":
        se = 1 / math.sqrt(n - 3)
    elif method == "hotelling":
        z -= (3 * z + r) / (4 * n)
        se = 1 / math.sqrt(n - 1)
    else:
        raise ValueError("`method` must be 'fisher' or 'hotelling'")
    q = float(norm.ppf(1 - (1 - conf_level) / 2))
    lo, hi = math.tanh(z - q * se), math.tanh(z + q * se)
    return RichResult(
        title=f"Correlation CI ({method})",
        summary_lines=[("lower", lo), ("upper", hi)],
        payload={"lower": lo, "upper": hi, "z": z, "se_z": se, "r": r, "n": n, "method": method},
    )


def cheatsheet():
    return "corrci: tanh(atanh r -/+ z_{1-a/2}/sqrt(n-3)); hotelling: z - (3z + r)/4n, 1/sqrt(n-1)"
