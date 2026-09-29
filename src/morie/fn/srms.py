"""Root mean square value."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def rms_value(x):
    r"""Root mean square ``sqrt((1/N) sum_n x(n)^2)``, the square root of the mean power.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> rms_value([3.0, -4.0]).value
    3.5355339059327378
    """
    v = _vec(x)
    if not v:
        raise ValueError("x must be non-empty")
    rms = math.sqrt(math.fsum(t * t for t in v) / len(v))
    return DescriptiveResult(name="rms_value", value=rms, extra={"rms": rms, "n": len(v)})


srms = rms_value
# compact alias per ledger/NAMING.md
rmsvalue = rms_value


def cheatsheet() -> str:
    return "rms_value(x) -> sqrt((1/N) sum x(n)^2)."
