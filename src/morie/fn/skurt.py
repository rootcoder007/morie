"""Kurtosis coefficient (excess)."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def kurtosis_coeff(x):
    r"""Excess kurtosis ``g2 = mu_4 / mu_2^2 - 3`` with the ``1/N`` central moments (the population / biased coefficient, ``e1071::kurtosis(type = 1)``); 0 for a constant signal.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> round(kurtosis_coeff([1.0, 2.0, 4.0, 7.0]).value, 12)
    -1.238095238095
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    n = len(v)
    mu = math.fsum(v) / n
    m2 = math.fsum((t - mu) ** 2 for t in v) / n
    kurt = 0.0 if m2 == 0.0 else (math.fsum((t - mu) ** 4 for t in v) / n) / (m2 * m2) - 3.0
    return DescriptiveResult(name="kurtosis_coeff", value=kurt, extra={"excess_kurtosis": kurt, "n": n})


skurt = kurtosis_coeff
# compact alias per ledger/NAMING.md
kurtosiscoeff = kurtosis_coeff


def cheatsheet() -> str:
    return "kurtosis_coeff(x) -> mu_4 / mu_2^2 - 3 (population, excess)."
