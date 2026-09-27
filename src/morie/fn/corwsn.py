"""Sample size for a correlation confidence interval of given width (Bonett & Wright 2000)."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["correlation_width_sample_size"]

_BC = {"pearson": (3, lambda r: 1.0), "spearman": (3, lambda r: 1 + r * r / 2), "kendall": (4, lambda r: 0.437)}


def correlation_width_sample_size(theta, width, method="pearson", conf_level=0.95, approach="exact"):
    r"""Pairs needed for a Fisher-z interval of expected width ``width`` around ``theta``.

    Bonett & Wright (2000): the interval is
    :math:`\tanh(\tanh^{-1}\theta \pm z_{1-\alpha/2}\,c/\sqrt{n-b})` with
    b = 3, c = 1 (Pearson); b = 3, :math:`c^2 = 1 + \theta^2/2` (Spearman);
    b = 4, :math:`c^2 = 0.437` (Kendall's tau-a). ``approach="exact"`` returns
    the smallest n whose width is at most ``width``; ``"two-stage"`` their
    approximation :math:`n_0 = 4c^2(1-\theta^2)^2(z/w)^2 + b`,
    :math:`n = \lceil (n_0 - b)(w_0/w)^2 + b\rceil` with :math:`w_0` the width
    at :math:`n_0`. The exact search reproduces 67 of the 72 entries of
    Hedderich, Sachs & Reynarowych (2023, Table 7.85); it gives 1515, 379 (Spearman,
    0.1), 662, 12 and 478 (Kendall, 0.1/0.1, 0.9/0.2, 0.4/0.1) where the table
    prints 1517, 382, 661, 11 and 448 (the last breaks the table's own
    monotone column).

    Parameters
    ----------
    theta : float
        Planning value of the correlation, in (-1, 1).
    width : float
        Required interval width, in (0, 2).
    method : {"pearson", "spearman", "kendall"}
    conf_level : float
    approach : {"exact", "two-stage"}

    Returns
    -------
    RichResult
        ``n`` and ``width_at_n``.

    References
    ----------
    Bonett, D. G. & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika 65,
    23-28.
    """
    if method not in _BC:
        raise ValueError("`method` must be 'pearson', 'spearman' or 'kendall'")
    theta, width = float(theta), float(width)
    if not -1 < theta < 1 or not 0 < width < 2:
        raise ValueError("need -1 < theta < 1 and 0 < width < 2")
    b, cf = _BC[method]
    c2 = cf(theta)
    z = float(norm.ppf(1 - (1 - conf_level) / 2))
    zt = math.atanh(theta)

    def w_at(n):
        h = z * math.sqrt(c2) / math.sqrt(n - b)
        return math.tanh(zt + h) - math.tanh(zt - h)

    if approach == "exact":
        n0 = 4 * c2 * (1 - theta * theta) ** 2 * (z / width) ** 2 + b
        n = max(b + 1, int(n0 / 2))
        while w_at(n) > width:
            n += 1
        while n > b + 1 and w_at(n - 1) <= width:
            n -= 1
    elif approach == "two-stage":
        n0 = 4 * c2 * (1 - theta * theta) ** 2 * (z / width) ** 2 + b
        n = math.ceil((n0 - b) * (w_at(n0) / width) ** 2 + b)
    else:
        raise ValueError("`approach` must be 'exact' or 'two-stage'")
    return RichResult(
        title=f"Sample size for a {method} correlation CI",
        summary_lines=[("n", n)],
        payload={"n": n, "width_at_n": w_at(n)},
    )


def cheatsheet():
    return "corwsn: smallest n with tanh(atanh t + z c/sqrt(n - b)) - tanh(atanh t - z c/sqrt(n - b)) <= w"
