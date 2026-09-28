# morie.fn -- function file (rootcoder007/morie)
"""Power of the test of zero correlation (Cohen's approximation as in pwr::pwr.r.test) and Cohen's
benchmarks for the size of a correlation."""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._rrng_core import qt

__all__ = ["pwr_r_test", "cohen_r_magnitude"]


def _pnorm(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _power(n, r, sig_level, alternative):
    if alternative == "two.sided":
        r = abs(r)
        ttt = qt(1 - sig_level / 2, n - 2)
    else:
        if alternative == "less":
            r = -r
        ttt = qt(1 - sig_level, n - 2)
    rc = math.sqrt(ttt * ttt / (ttt * ttt + n - 2))
    zr = math.atanh(r) + r / (2 * (n - 1))
    zrc = math.atanh(rc)
    p = _pnorm((zr - zrc) * math.sqrt(n - 3))
    if alternative == "two.sided":
        p += _pnorm((-zr - zrc) * math.sqrt(n - 3))
    return p


def _bisect(g, lo, hi):
    glo = g(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def pwr_r_test(n=None, r=None, sig_level=0.05, power=None, alternative="two.sided") -> RichResult:
    r"""Power, sample size, correlation or level for the test of ``H0: rho = 0`` (exactly one argument ``None``).

    The critical correlation ``r_c`` solves ``t_{alpha}(n - 2) = r_c sqrt(n - 2) / sqrt(1 - r_c^2)``;
    power is ``Phi((z_r - z_{r_c}) sqrt(n - 3))`` with Fisher's ``z = atanh``
    and the bias-corrected ``z_r = atanh(r) + r / (2(n - 1))`` (Cohen 1988,
    Section 3.4), plus the lower tail for two-sided tests. The missing
    argument is found by bisection (``n`` in ``(4, 1e9)``). Alternatives:
    ``"two.sided"``, ``"greater"``, ``"less"`` (as ``pwr::pwr.r.test``).

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences*, 2nd edn. Lawrence Erlbaum.

    Examples
    --------
    >>> round(pwr_r_test(n=100, r=0.3).power, 10)
    0.8647715463
    >>> round(pwr_r_test(r=0.3, power=0.8).n, 6)
    84.073638
    """
    if [n, r, power, sig_level].count(None) != 1:
        raise ValueError("exactly one of n, r, power and sig_level must be None")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")
    if n is not None and n < 4:
        raise ValueError("number of observations must be at least 4")
    if power is None:
        power = _power(n, r, sig_level, alternative)
    elif n is None:
        n = _bisect(lambda v: _power(v, r, sig_level, alternative) - power, 4 + 1e-10, 1e9)
    elif r is None:
        lo = 1e-10 if alternative == "two.sided" else -1 + 1e-10
        r = _bisect(lambda v: _power(n, v, sig_level, alternative) - power, lo, 1 - 1e-10)
    else:
        sig_level = _bisect(lambda v: _power(n, r, v, alternative) - power, 1e-10, 1 - 1e-10)
    return RichResult(payload={"n": n, "r": r, "sig_level": sig_level, "power": power, "alternative": alternative})


def cohen_r_magnitude(r: float) -> str:
    r"""Cohen's (1988) benchmark label for a correlation: ``|r| = .10`` small, ``.30`` medium, ``.50`` large.

    Returns ``"negligible"`` below .10, then ``"small"``, ``"medium"`` and
    ``"large"``. Cohen warned these are conventions for the behavioural
    sciences, to be replaced by comparisons within one's own field (Lovett
    2021, Practical Psychometrics, ch. 2).

    Examples
    --------
    >>> [cohen_r_magnitude(v) for v in (0.05, -0.2, 0.3, 0.72)]
    ['negligible', 'small', 'medium', 'large']
    """
    a = abs(r)
    if a > 1:
        raise ValueError("a correlation lies in [-1, 1]")
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def cheatsheet() -> str:
    return "pwr_r_test / cohen_r_magnitude -> power of the correlation test and Cohen's benchmarks."
