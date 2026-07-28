# morie.fn -- function file (rootcoder007/morie)
"""RMS noise level."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_rms_noise"]


def rangayyan_rms_noise(x, noise_segments=None):
    r"""RMS noise level from designated quiet segments (Rangayyan
    Ch. 3):

    .. math:: \sigma_n = \sqrt{\frac1N \sum_n x_{noise}[n]^2}.

    The segments must contain noise ONLY -- any signal leaking in
    inflates sigma and deflates every SNR computed from it. When no
    segments are given the quietest decile of the record is used as a
    fallback, and the result flags that it was estimated rather than
    designated.

    Parameters
    ----------
    x : array-like
        Signal.
    noise_segments : sequence of (start, stop), optional
        Index ranges holding noise only.

    Returns
    -------
    RichResult
        keys: ``rms_noise``, ``n_noise_samples``, ``segments_given``
        (bool), ``snr_db`` (of the whole record against it),
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (noise characterisation).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("x must have at least 2 samples.")
    given = noise_segments is not None
    if given:
        parts = []
        for a, b in noise_segments:
            a, b = int(a), int(b)
            if not 0 <= a < b <= x.size:
                raise ValueError(f"segment ({a}, {b}) is out of range.")
            parts.append(x[a:b])
        noise = np.concatenate(parts)
    else:
        k = max(2, x.size // 10)
        noise = x[np.argsort(np.abs(x))[:k]]
    sigma = float(np.sqrt(np.mean(noise**2)))
    sig_p = float(np.mean(x**2))
    snr = 10.0 * np.log10(sig_p / sigma**2) if sigma > 0 else np.inf
    return RichResult(payload={"rms_noise": sigma, "n_noise_samples": int(noise.size),
                               "segments_given": bool(given), "snr_db": float(snr),
                               "method": "sigma_n from designated noise; leakage inflates it"})


def cheatsheet():
    return "rgrmsnw: signal leaking into the 'noise' segment deflates every SNR"
