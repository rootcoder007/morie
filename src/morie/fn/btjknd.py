# morie.fn -- function file (rootcoder007/morie)
"""Delete-d jackknife variance."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["jackd", "bootjackknifed", "boot_jackknife_d"]


def jackd(theta, n, d):
    """Delete-d jackknife variance.

    Delete-d jackknife variance.

    v = (n-d) / (d * C(n,d)) * sum_s (theta_s - theta_bar)^2, the sum
    over the C(n,d) subsets of size n-d.  The delete-1 jackknife is
    inconsistent for non-smooth statistics such as the median; Shao &
    Wu show that deleting d > 1 restores consistency, and the leading
    factor is what keeps the estimator unbiased for the linear case.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Delete-d jackknife variance", payload=_c.jackd(theta=theta, n=n, d=d))


boot_jackknife_d = jackd


def cheatsheet():
    return "btjknd: Delete-d jackknife variance"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
bootjackknifed = jackd
