"""Noise power spectral density."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def noise_psd(x, fs=1.0, onesided=False) -> DescriptiveResult:
    r"""Power spectral density level of white noise, sigma^2 / fs (two-sided) or 2 sigma^2 / fs.

    A white process of variance sigma^2 sampled at fs has a flat PSD
    whose integral over (-fs/2, fs/2) is sigma^2: the two-sided level
    is sigma^2 / fs and the one-sided level (over (0, fs/2)) twice that
    (Proakis and Manolakis 2007, sec. 14.1). sigma^2 is the sample
    variance with the 1/N divisor.

    References
    ----------
    Proakis, J. G. and Manolakis, D. G. (2007). *Digital Signal Processing*,
    4th ed. Pearson.

    Examples
    --------
    >>> noise_psd([1.0, -1.0, 1.0, -1.0], fs=2.0).value
    0.5
    """
    v = [float(t) for t in (x.tolist() if hasattr(x, "tolist") else x)]
    n = len(v)
    m = math.fsum(v) / n
    var = math.fsum((t - m) ** 2 for t in v) / n
    psd = (2.0 if onesided else 1.0) * var / fs
    return DescriptiveResult(
        name="noise_psd", value=psd, extra={"psd": psd, "variance": var, "fs": fs, "n": n, "onesided": onesided}
    )


npsd = noise_psd


def cheatsheet() -> str:
    return "noise_psd(x, fs=1) -> white-noise PSD level sigma^2/fs (two-sided) or 2 sigma^2/fs."


# compact alias per ledger/NAMING.md
noisepsd = noise_psd
