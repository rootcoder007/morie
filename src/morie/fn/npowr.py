"""Noise power estimation."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def noise_power(x, signal=None) -> DescriptiveResult:
    r"""Noise power of a recording: from a clean reference, or by the difference estimator.

    With a reference signal the noise is x - signal and its power is
    (1/N) sum (x - s)^2. Without one, the noise variance is estimated from
    the first differences, sigma^2 = sum (x_{n+1} - x_n)^2 / (2 (N - 1))
    (von Neumann 1941; Rice 1984): for a slowly varying signal plus white
    noise the differences cancel the signal and have variance 2 sigma^2.

    References
    ----------
    von Neumann, J. (1941). Distribution of the ratio of the mean square
    successive difference to the variance. *Annals of Mathematical
    Statistics* 12, 367-395.
    Rice, J. (1984). Bandwidth choice for nonparametric regression. *Annals
    of Statistics* 12, 1215-1230.

    Examples
    --------
    >>> noise_power([1.0, 2.0, 4.0], signal=[1.0, 1.5, 3.0]).value
    0.4166666666666667
    >>> noise_power([0.0, 1.0, 0.0, 1.0]).value
    0.5
    """
    v = [float(t) for t in (x.tolist() if hasattr(x, "tolist") else x)]
    n = len(v)
    if signal is not None:
        s = [float(t) for t in (signal.tolist() if hasattr(signal, "tolist") else signal)]
        if len(s) != n:
            raise ValueError("x and signal must have equal length")
        pn = math.fsum((a - b) ** 2 for a, b in zip(v, s)) / n
        method = "reference"
    else:
        if n < 2:
            raise ValueError("the difference estimator needs at least two samples")
        pn = math.fsum((v[i + 1] - v[i]) ** 2 for i in range(n - 1)) / (2.0 * (n - 1))
        method = "successive differences"
    return DescriptiveResult(name="noise_power", value=pn, extra={"noise_power": pn, "n": n, "method": method})


npowr = noise_power


def cheatsheet() -> str:
    return "noise_power(x, signal=None) -> mean (x - s)^2, or sum diff(x)^2 / (2(N-1))."


# compact alias per ledger/NAMING.md
noisepower = noise_power
