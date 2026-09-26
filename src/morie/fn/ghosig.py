# morie.fn -- function file (rootcoder007/morie)
"""Detect hidden (ghost) periodic signals buried in noise via surrogate testing."""

from __future__ import annotations

from . import _array_core as np
from ._containers import TestResult


def ghost_signal(
    data: np.ndarray | list[float],
    *,
    n_surrogates: int = 999,
    alpha: float = 0.05,
    seed: int | None = None,
) -> TestResult:
    """Detect hidden (ghost) periodic signals buried in noise via surrogate testing.

    Computes the power spectrum of the data and compares the peak spectral
    power against a null distribution from randomly permuted surrogates
    (exchangeable-noise null; the Monte Carlo p-value is exact).
    Phase-randomised surrogates are not usable here: they preserve the
    periodogram, and with it the statistic.

    Parameters
    ----------
    data : array
        1-D time series.
    n_surrogates : int
        Number of surrogates for the null distribution.
    alpha : float
        Significance level.
    seed : int or None
        Random seed.

    Returns
    -------
    TestResult
        Statistic = observed peak spectral power.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)
    if n < 8:
        raise ValueError("Need at least 8 data points")
    fft_x = np.fft.rfft(x)
    power = np.abs(fft_x) ** 2
    obs_peak = float(np.max(power[1:]))
    peak_freq_idx = int(np.argmax(power[1:]) + 1)
    peak_freq = float(peak_freq_idx / n)
    # Surrogates are random PERMUTATIONS of the series: under the null of
    # exchangeable noise every ordering is equally likely, so the Monte
    # Carlo p-value is exact.  Phase-randomised surrogates cannot be used
    # with this statistic -- they keep every Fourier amplitude, so each
    # surrogate's periodogram equals the observed one and the "p-value"
    # was decided by rounding noise in the comparison.
    rng = np.random.default_rng(seed)
    null_peaks = np.empty(n_surrogates)
    xs = [float(v) for v in x.tolist()]
    for i in range(n_surrogates):
        perm = [int(k) for k in rng.permutation(n).tolist()]
        s_fft = np.fft.rfft(np.asarray([xs[k] for k in perm]))
        s_power = np.abs(s_fft) ** 2
        null_peaks[i] = np.max(s_power[1:])
    p_value = float((null_peaks >= obs_peak).sum() + 1) / (n_surrogates + 1)
    return TestResult(
        test_name="Ghost signal detection (surrogate)",
        statistic=obs_peak,
        p_value=p_value,
        method="Permutation surrogate",
        n=n,
        extra={
            "peak_frequency": peak_freq,
            "peak_period": 1.0 / peak_freq if peak_freq > 0 else float("inf"),
            "n_surrogates": n_surrogates,
            "null_95th": float(np.percentile(null_peaks, 95)),
            "signal_detected": p_value < alpha,
        },
    )


ghosig = ghost_signal


def cheatsheet() -> str:
    return "ghost_signal({}) -> Ghost signal detection in noise."


# compact alias per ledger/NAMING.md
ghostsignal = ghost_signal
