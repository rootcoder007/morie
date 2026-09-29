"""Zero-crossing rate of a signal."""

from __future__ import annotations

from ._containers import DescriptiveResult


def _rate(v):
    s = [1 if t >= 0 else -1 for t in v]
    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) / (len(s) - 1)


def zero_crossing_rate(x, frame_length: int | None = None) -> DescriptiveResult:
    r"""Zero-crossing rate: the fraction of successive sample pairs whose signs differ.

    ``ZCR = (1/(N - 1)) sum_{n>=1} 1[sgn x(n) != sgn x(n-1)]`` with zero
    counted as positive (Rabiner and Schafer 1978, sec. 4.4; Rangayyan 2015,
    sec. 5.4). With ``frame_length`` the rate is computed on consecutive
    non-overlapping frames (``extra["per_frame"]``) and ``value`` is their
    mean.

    References
    ----------
    Rabiner, L. R. and Schafer, R. W. (1978). *Digital Processing of Speech
    Signals*. Prentice-Hall.
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed.
    Wiley-IEEE Press.

    Examples
    --------
    >>> zero_crossing_rate([1.0, -1.0, -2.0, 3.0, 0.0]).value
    0.5
    """
    v = [float(t) for t in (x.tolist() if hasattr(x, "tolist") else x)]
    if frame_length is None:
        if len(v) < 2:
            raise ValueError("x needs at least two samples")
        return DescriptiveResult(name="zero_crossing_rate", value=_rate(v), extra={})
    L = int(frame_length)
    if L < 2:
        raise ValueError("frame_length must be at least 2")
    per = [_rate(v[i * L : (i + 1) * L]) for i in range(len(v) // L)]
    return DescriptiveResult(name="zero_crossing_rate", value=sum(per) / len(per), extra={"per_frame": per})


alias = zero_crossing_rate


def cheatsheet() -> str:
    return "zero_crossing_rate(x, frame_length=None) -> share of sign changes between successive samples."
