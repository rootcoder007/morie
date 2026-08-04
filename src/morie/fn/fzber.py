# morie.fn -- function file (rootcoder007/morie)
"""Berry-Esseen bound for the kernel quantile estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["qbebnd", "fauzi_berry_esseen_quantile"]


def qbebnd(x, n, m=4, improved=True):
    r"""Berry-Esseen bound for the kernel quantile estimator.

    Eq. (3.5) and Remark 3.1:

    .. math:: P\big(\sqrt n|\hat Q_{p,h} - Q(p)| \le x\sigma_n\big)
              = 2\Phi(x) - 1 + O(n^{-r}).

    The rate ``r`` is the whole story, and it depends on the ORDER of the
    kernel, not on the sample:

    * :math:`m = 2` gives :math:`r = 1/3`;
    * :math:`m = 3` gives :math:`r = 5/13`;
    * :math:`m = 4` gives :math:`r = 7/17` by the earlier literature,
      improved to :math:`r = 1/2` by Remark 3.1 of this book.

    Those five numbers are quoted verbatim from the text. The book also
    says plainly that :math:`o(n^{-1/2})` is unreachable for ANY kernel
    order without adding the next Edgeworth term, so :math:`r = 1/2` is
    the ceiling of this approach, not a stepping stone.

    Returns the two-sided normal probability and the bound term
    :math:`n^{-r}` separately: the second is an order symbol with no
    constant, and adding it to the first would be arithmetic on something
    that has no numerical value.

    Parameters
    ----------
    x : float or array-like
        Argument, in units of ``sigma_n``.
    n : int
        Sample size.
    m : int, default 4
        Kernel order; 2, 3 or 4.
    improved : bool, default True
        For ``m = 4``, use Remark 3.1's ``r = 1/2`` instead of the
        literature's ``7/17``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``bound``, ``rate``, ``m``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (3.5), Remark 3.1.
    """
    from . import _stats_core as stats

    n = int(n)
    m = int(m)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if m == 2:
        rate = 1.0 / 3.0
    elif m == 3:
        rate = 5.0 / 13.0
    elif m == 4:
        rate = 0.5 if improved else 7.0 / 17.0
    else:
        raise ValueError(f"the book states rates for m = 2, 3, 4 only, got {m}.")
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(xv < 0):
        raise ValueError("(3.5) bounds an absolute deviation; x must be >= 0.")
    est = 2.0 * stats.norm.cdf(xv) - 1.0
    return RichResult(
        payload={
            "estimate": [float(v) for v in est],
            "bound": float(n) ** (-rate),
            "rate": float(rate),
            "m": m,
            "n": n,
            "method": "Berry-Esseen bound for the kernel quantile estimator (3.5)",
        }
    )


fauzi_berry_esseen_quantile = qbebnd


def cheatsheet():
    return "fzber: Berry-Esseen rate set by KERNEL ORDER: 1/3, 5/13, 7/17 -> 1/2 (Remark 3.1)"


# CANONICAL TEST
# >>> r = qbebnd(x=1.96, n=100)
# >>> abs(r['estimate'][0] - 0.95) < 1e-3 and r['rate'] == 0.5
# True
