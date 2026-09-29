"""Raw moment."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def raw_moment(x, k=1):
    r"""The ``k``-th raw (non-central) moment ``m_k = (1/N) sum_n x(n)^k``; ``k = 1`` is the mean, ``k = 2`` the mean power.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> raw_moment([1.0, 2.0, 4.0, 7.0], k=2).value
    17.5
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    mk = math.fsum(t**k for t in v) / len(v)
    return DescriptiveResult(name="raw_moment", value=mk, extra={"moment_order": k, "raw_moment": mk, "n": len(v)})


smom1 = raw_moment
# compact alias per ledger/NAMING.md
rawmoment = raw_moment


def cheatsheet() -> str:
    return "raw_moment(x, k=1) -> (1/N) sum x(n)^k."
