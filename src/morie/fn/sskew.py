"""Skewness coefficient."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def skewness_coeff(x):
    r"""Moment skewness ``g1 = mu_3 / mu_2^(3/2)`` with the ``1/N`` central moments (the population / biased coefficient, ``e1071::skewness(type = 1)``); 0 for a constant signal.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> round(skewness_coeff([1.0, 2.0, 4.0, 7.0]).value, 12)
    0.498783749111
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    n = len(v)
    mu = math.fsum(v) / n
    m2 = math.fsum((t - mu) ** 2 for t in v) / n
    skew = 0.0 if m2 == 0.0 else (math.fsum((t - mu) ** 3 for t in v) / n) / m2**1.5
    return DescriptiveResult(name="skewness_coeff", value=skew, extra={"skewness": skew, "n": n})


sskew = skewness_coeff
# compact alias per ledger/NAMING.md
skewnesscoeff = skewness_coeff


def cheatsheet() -> str:
    return "skewness_coeff(x) -> mu_3 / mu_2^(3/2) (population)."
