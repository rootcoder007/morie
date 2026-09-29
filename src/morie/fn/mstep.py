"""Maximum LMS step size."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def max_step_size(x, order: int = 16) -> DescriptiveResult:
    r"""Upper bound of the LMS step size for mean-square stability, mu_max = 2 / (M P_x).

    The LMS filter with M taps converges in the mean square when 0 <
    mu < 2 / tr(R) = 2 / (M P_x), P_x = (1/N) sum x^2 the input power
    (tap-input power, Haykin 2014, sec. 6.4; Widrow and Stearns 1985).

    References
    ----------
    Haykin, S. (2014). *Adaptive Filter Theory*, 5th ed. Pearson.
    Widrow, B. and Stearns, S. D. (1985). *Adaptive Signal Processing*.
    Prentice-Hall.

    Examples
    --------
    >>> max_step_size([1.0, -2.0, 2.0, -1.0], order=4).value
    0.2
    """
    v = [float(t) for t in (x.tolist() if hasattr(x, "tolist") else x)]
    px = math.fsum(t * t for t in v) / len(v)
    if px <= 0:
        raise ValueError("Signal power is zero; step size is undefined.")
    if order <= 0:
        raise ValueError("Filter order must be positive.")
    mu = 2.0 / (order * px)
    return DescriptiveResult(name="max_step_size", value=mu, extra={"mu_max": mu, "order": order, "Px": px})


mstep = max_step_size


def cheatsheet() -> str:
    return "max_step_size(x, order=16) -> LMS stability bound 2 / (M P_x)."


# compact alias per ledger/NAMING.md
maxstepsize = max_step_size
