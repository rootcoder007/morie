"""Sample mean."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def sample_mean(x):
    r"""Sample mean ``xbar = (1/N) sum_n x(n)`` of a signal or data vector (correctly rounded sum, ``math.fsum``).

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> sample_mean([1.0, 2.0, 4.0, 7.0]).value
    3.5
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    mu = math.fsum(v) / len(v)
    return DescriptiveResult(name="sample_mean", value=mu, extra={"mean": mu, "n": len(v)})


smean = sample_mean
# compact alias per ledger/NAMING.md
samplemean = sample_mean


def cheatsheet() -> str:
    return "sample_mean(x) -> (1/N) sum x(n)."
