"""Group delay of a digital filter."""

from __future__ import annotations

import cmath
import math

from ._containers import DescriptiveResult


def group_delay(b, a, worN: int = 512) -> DescriptiveResult:
    r"""Group delay tau_g(w) = -d arg H(e^{jw}) / dw of H(z) = B(z)/A(z), in samples.

    Computed exactly, not by differencing the phase: with c = b * a~
    (convolution with the reversed a) tau_g(w) = Re[sum_k k c_k
    e^{-jwk} / sum_k c_k e^{-jwk}] - (len(a) - 1) on w = pi k / worN,
    k = 0..worN-1 (Oppenheim and Schafer 2010, sec. 5.1.2; the algorithm
    of scipy.signal.group_delay); frequencies where the denominator
    vanishes are set to 0. value is the delay curve, extra holds the
    frequencies.

    References
    ----------
    Oppenheim, A. V. and Schafer, R. W. (2010). *Discrete-Time Signal
    Processing*, 3rd ed. Pearson.

    Examples
    --------
    >>> [round(v, 12) for v in group_delay([1.0, 2.0, 1.0], [1.0], worN=4).value]
    [1.0, 1.0, 1.0, 1.0]
    """
    bb = [float(v) for v in (b.tolist() if hasattr(b, "tolist") else b)]
    aa = [float(v) for v in (a.tolist() if hasattr(a, "tolist") else a)]
    ar = aa[::-1]
    c = [0.0] * (len(bb) + len(ar) - 1)
    for i, u in enumerate(bb):
        for j, v in enumerate(ar):
            c[i + j] += u * v
    ws = [math.pi * k / worN for k in range(int(worN))]
    gd = []
    for w in ws:
        num = sum(k * ck * cmath.exp(-1j * w * k) for k, ck in enumerate(c))
        den = sum(ck * cmath.exp(-1j * w * k) for k, ck in enumerate(c))
        gd.append(0.0 if abs(den) < 10 * 2.220446049250313e-16 else (num / den).real - (len(aa) - 1))
    return DescriptiveResult(name="group_delay", value=gd, extra={"frequencies": ws, "delay": gd})


grpdl = group_delay


def cheatsheet() -> str:
    return "group_delay(b, a, worN=512) -> exact group delay Re[sum k c_k z^-k / sum c_k z^-k] - (len(a)-1)."


# compact alias per ledger/NAMING.md
groupdelay = group_delay
