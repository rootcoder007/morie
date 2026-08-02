# morie.fn -- function file (rootcoder007/morie)
"""Relationship between Kendall's W and the average Spearman rho."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_concordance_rho_link"]


def gibbons_concordance_rho_link(W, k, n=None):
    r"""Convert between Kendall's :math:`W` and the average Spearman rho.

    Gibbons and Chakraborti equation (12.4.6):

    .. math:: r_{av} = \frac{kW - 1}{k - 1},

    equivalently their (12.4.7):

    .. math:: W = r_{av} + \frac{1 - r_{av}}{k}
                = \frac{r_{av}(k-1) + 1}{k}.

    :math:`r_{av}` is the average of the :math:`\binom{k}{2}` Spearman
    coefficients between every pair of the :math:`k` rankings, so the
    two statistics carry exactly the same information and the choice
    between them is presentational.

    The relation explains why :math:`W` has the range it does.
    :math:`W = 1` requires :math:`r_{av} = 1`, i.e. every pair of
    rankings identical. But :math:`r_{av} = -1` is IMPOSSIBLE for
    :math:`k > 2` -- three rankings cannot all disagree perfectly with
    each other -- and (12.4.7) with :math:`W \ge 0` gives the true
    floor

    .. math:: r_{av} \ge -\frac{1}{k - 1}.

    So concordance and discordance are not symmetrical opposites, and
    a 0-to-1 scale is the appropriate one for :math:`k` samples rather
    than the -1-to-1 of a pairwise correlation. ``rho_min`` reports
    that floor.

    Parameters
    ----------
    W : float
        Kendall's coefficient of concordance, in [0, 1].
    k : int
        Number of rankings (observers), at least 2.
    n : int, optional
        Number of objects. When given, the null moments (12.4.8) and
        the chi-square approximation are returned too.

    Returns
    -------
    RichResult
        ``rho_av``, ``W``, ``k``, ``rho_min``, and with ``n``:
        ``expected_W``, ``var_W``, ``chi2``, ``df``.

    References
    ----------
    Gibbons and Chakraborti (2011), *Nonparametric Statistical
    Inference*, 5th ed., section 12.4.1, equations (12.4.6)-(12.4.8),
    pp. 454-456.

    Examples
    --------
    >>> out = gibbons_concordance_rho_link(1.0, 4)
    >>> out["rho_av"]
    1.0
    >>> round(gibbons_concordance_rho_link(0.0, 4)["rho_av"], 6)
    -0.333333
    """
    k = int(k)
    if k < 2:
        raise ValueError("need at least 2 rankings, got %d." % k)
    Wv = float(W)
    if not -1e-12 <= Wv <= 1 + 1e-12:
        raise ValueError("W must lie in [0, 1], got %r." % W)
    rho = (k * Wv - 1.0) / (k - 1.0)
    rho_min = -1.0 / (k - 1.0)
    payload = {
        "estimate": float(rho),
        "rho_av": float(rho),
        "W": Wv,
        "k": k,
        "rho_min": float(rho_min),
        "range_note": (
            "r_av = -1 is impossible for k > 2 -- three rankings cannot all "
            "disagree perfectly -- so the floor is -1/(k-1) and W is "
            "properly a 0-to-1 measure, not a correlation"
        ),
        "inverse_check": float((rho * (k - 1) + 1.0) / k),
        "method": "Kendall W to average Spearman rho, eq (12.4.6)",
    }
    if n is not None:
        nn = int(n)
        if nn < 2:
            raise ValueError("need at least 2 objects, got %d." % nn)
        payload.update({
            "n": nn,
            "expected_W": 1.0 / k,
            "var_W": 2.0 * (k - 1.0) / (k ** 3 * (nn - 1.0)),
            "chi2": float(k * (nn - 1.0) * Wv),
            "df": nn - 1,
            "chi2_note": (
                "k(n-1)W is approximately chi-square on n-1 degrees of "
                "freedom; the beta approximation behind it degrades when "
                "k(n-1) is small"
            ),
        })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "gb1241r: W <-> average Spearman rho by (12.4.6), with the "
        "-1/(k-1) floor that makes W a 0-to-1 measure"
    )
