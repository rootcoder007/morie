"""Central moment."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def central_moment(x, k=2):
    r"""The ``k``-th central moment ``mu_k = (1/N) sum_n (x(n) - xbar)^k`` with the ``1/N`` (population, biased) divisor; ``k = 2`` is the population variance.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> central_moment([1.0, 2.0, 4.0, 7.0], k=2).value
    5.25
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    mu = math.fsum(v) / len(v)
    mk = math.fsum((t - mu) ** k for t in v) / len(v)
    return DescriptiveResult(
        name="central_moment", value=mk, extra={"moment_order": k, "central_moment": mk, "n": len(v)}
    )


smom2 = central_moment
# compact alias per ledger/NAMING.md
centralmoment = central_moment


def cheatsheet() -> str:
    return "central_moment(x, k=2) -> (1/N) sum (x(n) - xbar)^k."
