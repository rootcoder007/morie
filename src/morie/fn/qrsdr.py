"""QRS complex duration measurement."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def qrs_duration(qrs_on, qrs_off, fs: float = 1.0) -> DescriptiveResult:
    r"""QRS durations (offset - onset) / fs of paired beats, their mean and sample SD.

    Onsets and offsets are sample indices of detected QRS boundaries
    (Rangayyan 2015, sec. 4.3; normal adult QRS 60-100 ms); the first
    min(len(on), len(off)) pairs are used. The SD has the n - 1
    divisor.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed.
    Wiley-IEEE Press.

    Examples
    --------
    >>> round(qrs_duration([100, 460, 830], [125, 482, 858], fs=250.0).value, 12)
    0.1
    """
    on = [int(v) for v in qrs_on]
    off = [int(v) for v in qrs_off]
    n = min(len(on), len(off))
    if n == 0:
        return DescriptiveResult(name="qrs_duration", value=0.0, extra={"qrs_durations": [], "n_beats": 0})
    dur = [(off[i] - on[i]) / fs for i in range(n)]
    m = math.fsum(dur) / n
    sd = math.sqrt(math.fsum((d - m) ** 2 for d in dur) / (n - 1)) if n > 1 else 0.0
    return DescriptiveResult(
        name="qrs_duration",
        value=m,
        extra={"qrs_durations": dur, "mean_dur": m, "std_dur": sd, "n_beats": n, "fs": fs},
    )


qrsdr = qrs_duration


def cheatsheet() -> str:
    return "qrs_duration(on, off, fs) -> mean and SD of (off - on)/fs."


# compact alias per ledger/NAMING.md
qrsduration = qrs_duration
