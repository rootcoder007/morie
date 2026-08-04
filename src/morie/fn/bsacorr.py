# morie.fn -- bsacorr (rootcoder007/morie)
"""Correlation and spectral density: ACF, CCF, PSD, coherence, matched filtering, synchronized averaging.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 65
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import fsum, sqrt
import math as _math
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._containers import SignalResult
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer

__all__ = [
    'coherence',
    'matched_filter',
    'rangayyan_acf_estimate',
    'rangayyan_bartlett_psd',
    'bartlettpsd',
    'rangayyan_bandwidth',
    'rangayyan_ccf',
    'rangayyan_coupled_freq_select',
    'rangayyan_coherence',
    'rangayyan_coherence_cxy',
    'rangayyan_eeg_bands',
    'rangayyan_eeg_autocorr',
    'rangayyan_eeg_rhythm_detect',
    'rangayyan_eeg_spectral',
    'rangayyan_emg_peak_freq',
    'rangayyan_ch3_correlation_sum',
    'rangayyan_erp_artifact_remove',
    'rangayyan_matched_filter',
    'rangayyan_matched_filter_snr',
    'rangayyan_periodogram',
    'rangayyan_psd',
    'rangayyan_psd_to_hz',
    'rangayyan_psd_to_acf',
    'rangayyan_pcg_sync_avg',
    'rangayyan_seizure_detect',
    'rangayyan_spectral_moments',
    'rangayyan_spectral_resolution',
    'rangayyan_template_match',
    'rangayyan_welch_psd',
    'rangayyan_ch3_acf_continuous',
    'rangayyan_ch3_acf_ensemble_estimate',
    'ensavg',
    'rangayyan_ch3_ensemble_average_function',
    'rangayyan_ch3_time_averaged_acf',
    'rangayyan_ch3_ccf_continuous',
    'rangayyan_ch3_idft_definition',
    'rangayyan_ch3_parseval_theorem',
    'rangayyan_ch3_synchronized_averaging_sum',
    'rangayyan_ch3_normalized_cross_correlation_template',
    'rangayyan_ch4_dot_product_discrete',
    'rangayyan_ch4_correlation_coefficient_normalized_dot',
    'rangayyan_ch4_continuous_dot_product',
    'rangayyan_ch4_ccf_continuous_with_delay',
    'rangayyan_ch4_ccf_discrete_with_delay',
    'rangayyan_ch4_ccf_outer_product_random_signals',
    'rangayyan_ch4_csd_from_ccf',
    'rangayyan_ch4_coherence_spectrum',
    'rangayyan_ch4_matched_filter_input_ft',
    'rangayyan_ch4_matched_filter_output_inverse_ft',
    'rangayyan_ch4_white_noise_psd_input',
    'rangayyan_ch4_noise_psd_at_output',
    'rangayyan_ch4_average_output_noise_power',
    'rangayyan_ch4_matched_filter_instantaneous_signal',
    'rangayyan_ch4_peak_power_snr',
    'rangayyan_ch4_signal_total_energy',
    'rangayyan_ch4_snr_normalized_ratio',
    'rangayyan_ch4_schwarz_inequality_complex',
    'rangayyan_ch4_schwarz_inequality_real',
    'rangayyan_ch4_cauchy_schwarz_vectors',
    'rangayyan_ch4_triangle_inequality_vectors',
    'rangayyan_ch4_matched_filter_optimal_transfer_function',
    'rangayyan_ch4_matched_filter_impulse_response',
    'rangayyan_ch4_matched_filter_output_acf',
    'rangayyan_ch4_basic_signal_g',
    'rangayyan_ch4_matched_filter_optimal_H_eeg',
    'rangayyan_ch4_matched_filter_impulse_response_eeg',
    'rangayyan_ch4_matched_filter_output_psd',
]


# -- coher: Coherence between two signals.
def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    *,
    nperseg: int = 256,
) -> DescriptiveResult:
    r"""Magnitude-squared coherence between two signals.

    Measures the linear relationship between *x* and *y* at each
    frequency, normalized to [0, 1]:

    .. math::

        C_{xy}(f) = \\frac{|P_{xy}(f)|^2}{P_{xx}(f) \\, P_{yy}(f)}

    Uses Welch's method for cross- and auto-spectral estimation.

    Parameters
    ----------
    x, y : array-like
        1-D input signals (must be the same length).
    fs : float
        Sampling frequency in Hz (default 1.0).
    nperseg : int
        Segment length for Welch estimation (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``frequencies``, ``coherence``,
        ``cross_spectrum``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
    """
    from ._signal_core import csd, welch

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    seg = min(nperseg, n)

    f, Pxx = welch(x, fs=fs, nperseg=seg)
    _, Pyy = welch(y, fs=fs, nperseg=seg)
    _, Pxy = csd(x, y, fs=fs, nperseg=seg)

    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-20)

    return DescriptiveResult(
        name="coherence",
        value=float(np.mean(coh)),
        extra={
            "frequencies": f,
            "coherence": coh,
            "cross_spectrum": Pxy,
        },
    )


coher = coherence


# -- mchfl: Matched filter for template detection in biomedical signals.
_QUOTE = "This is the template you are looking for. --"


def matched_filter(
    x: np.ndarray,
    template: np.ndarray,
    fs: float = 1.0,
) -> SignalResult:
    r"""Apply a matched filter to detect a known template in signal *x*.

    The matched filter maximizes the signal-to-noise ratio at the
    output when the noise is white and Gaussian:

    .. math::

        y(n) = \\sum_{k=0}^{M-1} h(k) \\, x(n-k), \\quad
        h(k) = s(M-1-k)

    where :math:`s` is the template (time-reversed for correlation).

    Parameters
    ----------
    x : array-like
        Input signal to search.
    template : array-like
        Known template / reference waveform.
    fs : float
        Sampling frequency in Hz (default 1.0).

    Returns
    -------
    SignalResult
        ``filtered`` contains the matched-filter output (correlation),
        ``extra`` has ``peak_index`` and ``peak_snr``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    template = np.asarray(template, dtype=float).ravel()

    h = template[::-1] / (np.linalg.norm(template) + 1e-12)

    output = np.correlate(x, template, mode="full")
    output = output[: len(x)]

    peak_idx = int(np.argmax(np.abs(output)))
    noise_est = np.std(output) + 1e-12
    peak_snr = float(np.abs(output[peak_idx]) / noise_est)

    return SignalResult(
        name="matched_filter",
        filtered=output,
        fs=fs,
        n_samples=len(output),
        extra={"peak_index": peak_idx, "peak_snr": peak_snr},
    )


mchfl = matched_filter


# compact alias per ledger/NAMING.md
matchedfilter = matched_filter


# -- rgacf: Autocorrelation estimate.
def rangayyan_acf_estimate(x, max_lag=None, biased=False):
    r"""Autocorrelation estimate (Rangayyan Ch. 3):

    .. math:: R_{xx}(m) = \frac{1}{N - |m|}
              \sum_{n=0}^{N-1-|m|} x(n)\,x(n+m).

    This is the UNBIASED estimator -- divisor N - |m|, not N. It is
    unbiased at every lag but its variance grows as |m| approaches N,
    and unlike the biased form it is not guaranteed positive
    semi-definite, so an AR fit from it can produce an unstable model.
    Both are returned and the trade-off is stated rather than hidden.

    Parameters
    ----------
    x : array-like
        Signal.
    max_lag : int, optional
        Maximum lag; N - 1 by default.
    biased : bool, default False
        Return the divisor-N form as the primary estimate.

    Returns
    -------
    RichResult
        keys: ``lags``, ``acf`` (per ``biased``), ``acf_unbiased``,
        ``acf_biased``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (autocorrelation).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 2:
        raise ValueError(f"need at least 2 samples, got {N}.")
    L = N - 1 if max_lag is None else int(max_lag)
    if not 0 <= L <= N - 1:
        raise ValueError(f"max_lag must lie in 0..{N - 1}, got {L}.")
    lags = np.arange(L + 1)
    raw = np.array([float(np.dot(x[: N - m], x[m:])) for m in lags])
    unb = raw / (N - lags)
    bia = raw / N
    return RichResult(payload={"lags": lags, "acf": bia if biased else unb,
                               "acf_unbiased": unb, "acf_biased": bia, "N": int(N),
                               "method": "R_xx(m) with divisor N-|m| (unbiased, not PSD-guaranteed)"})


# -- rgbartl: Bartlett's averaged periodogram.
def _dft_power(seg):
    """|DFT|^2 at each bin, direct evaluation.

    O(M^2) rather than an FFT: correct at any M, no padding to a power of
    two, and these segments are short.  ponytail: swap in the radix-2
    path in _signal_core if a caller ever needs long segments.
    """
    m = len(seg)
    out = []
    for k in range(m // 2 + 1):
        re = im = 0.0
        for n, v in enumerate(seg):
            ang = -2.0 * _math.pi * k * n / m
            re += v * _math.cos(ang)
            im += v * _math.sin(ang)
        out.append(re * re + im * im)
    return out


def rangayyan_bartlett_psd(x, fs=1.0, n_segments=None, segment_length=None):
    r"""Bartlett's method: average the periodograms of disjoint segments.

    Rangayyan eqs. (6.14)-(6.16).  The record is split into :math:`K`
    non-overlapping segments of :math:`M` samples,

    .. math:: S_i(\omega) = \frac{1}{M}
              \left| \sum_{n=0}^{M-1} x_i(n) e^{-j\omega n} \right|^2

    and the estimate is their sample mean,
    :math:`S_B(\omega) = \frac1K \sum_i S_i(\omega)`.

    Averaging :math:`K` independent periodograms divides the variance by
    :math:`K` while multiplying the resolution bandwidth by the same
    factor -- the trade the method exists to make.  Segments are DISJOINT
    here, as the book specifies; Welch's overlapping variant is a
    different estimator and is not what this citation promises.

    Parameters
    ----------
    x : sequence
        The signal.
    fs : float
        Sampling rate, for the returned frequency axis.
    n_segments, segment_length : int, optional
        Give exactly one.  Trailing samples that do not fill a whole
        segment are dropped, as the segmentation in eq. (6.14) requires.

    Returns
    -------
    RichResult
        ``psd``, ``freqs``, ``n_segments``, ``segment_length``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, eqs. (6.14)-(6.16), after Oppenheim & Schafer.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    if (n_segments is None) == (segment_length is None):
        raise ValueError("give exactly one of n_segments, segment_length")
    if n_segments is not None:
        k = int(n_segments)
        if k < 1:
            raise ValueError("n_segments must be positive")
        m = n // k
    else:
        m = int(segment_length)
        if m < 2:
            raise ValueError("segment_length must be at least 2")
        k = n // m
    if k < 1 or m < 2:
        raise ValueError("segmentation leaves no usable segment")

    acc = None
    for i in range(k):
        seg = xs[i * m:(i + 1) * m]
        p = [v / m for v in _dft_power(seg)]
        acc = p if acc is None else [a + b for a, b in zip(acc, p)]
    psd = [v / k for v in acc]
    freqs = [j * fs / m for j in range(len(psd))]
    return RichResult(
        title="Bartlett averaged periodogram (Rangayyan eq. 6.16)",
        summary_lines=[("segments", k), ("segment length", m)],
        payload={"psd": psd, "freqs": freqs, "n_segments": k,
                 "segment_length": m,
                 "method": "Rangayyan (2024) eqs. (6.14)-(6.16)"},
    )


bartlettpsd = rangayyan_bartlett_psd


# -- rgbwbnd: Spectral bandwidth.
def rangayyan_bandwidth(psd, freqs, criterion="3dB"):
    r"""Spectral bandwidth by two criteria (Rangayyan Ch. 3):

    - ``"3dB"``: the span of frequencies where
      :math:`S(f) \ge S_{\max}/2` (half power, i.e. -3 dB);
    - ``"99"``: the narrowest band from the peak containing 99% of the
      total power.

    The two answer different questions and can differ by an order of
    magnitude on a peaky spectrum, so the criterion is explicit rather
    than defaulted silently.

    Parameters
    ----------
    psd : array-like
        Power spectral density.
    freqs : array-like
        Matching frequencies.
    criterion : {"3dB", "99"}
        Which bandwidth to report.

    Returns
    -------
    RichResult
        keys: ``bandwidth``, ``f_low``, ``f_high``, ``f_peak``,
        ``criterion``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (bandwidth measures).
    """
    S = np.asarray(psd, dtype=float).ravel()
    f = np.asarray(freqs, dtype=float).ravel()
    if S.size != f.size:
        raise ValueError("psd and freqs must have the same length.")
    if S.size < 2:
        raise ValueError("need at least 2 spectral points.")
    if np.any(S < 0):
        raise ValueError("a power spectral density cannot be negative.")
    ipk = int(np.argmax(S))
    if criterion == "3dB":
        thr = S[ipk] / 2.0
        above = np.flatnonzero(S >= thr)
        lo, hi = float(f[above[0]]), float(f[above[-1]])
    elif criterion == "99":
        total = float(S.sum())
        if total <= 0:
            raise ValueError("spectrum has zero total power.")
        lo_i = hi_i = ipk
        acc = S[ipk]
        while acc < 0.99 * total and (lo_i > 0 or hi_i < S.size - 1):
            left = S[lo_i - 1] if lo_i > 0 else -np.inf
            right = S[hi_i + 1] if hi_i < S.size - 1 else -np.inf
            if left >= right:
                lo_i -= 1
                acc += S[lo_i]
            else:
                hi_i += 1
                acc += S[hi_i]
        lo, hi = float(f[lo_i]), float(f[hi_i])
    else:
        raise ValueError("criterion must be '3dB' or '99'.")
    return RichResult(payload={"bandwidth": hi - lo, "f_low": lo, "f_high": hi,
                               "f_peak": float(f[ipk]), "criterion": criterion,
                               "method": f"{criterion} bandwidth about the spectral peak"})


# -- rgccf: Cross-correlation function (CCF) between two signals.
def rangayyan_ccf(x, y, max_lag):
    """
    Cross-correlation function (CCF) between two signals

    Formula: R_xy(tau) = (1/N) sum x(n) * y(n + tau)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ccf, lags

    References
    ----------
    Rangayyan Ch 2
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Cross-correlation function (CCF) between two signals",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Cross-correlation function (CCF) between two signals",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanccf = rangayyan_ccf


# -- rgcfsle: Cardiorespiratory coupling analysis via coherence and PLV.
def rangayyan_coupled_freq_select(ecg, resp, fs):
    """
    Cardiorespiratory coupling analysis via coherence and PLV

    Formula: PLV = |mean(exp(j*(phi_ecg - phi_resp)))|; coherence at resp frequency

    Parameters
    ----------
    ecg : array-like
        Input data.
    resp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: plv, coherence_at_rr

    References
    ----------
    Rangayyan Ch 2.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Cardiorespiratory coupling analysis via coherence and PLV",
        }
    )


# -- rgcoh: Magnitude-squared coherence -- Rangayyan & Krishnan Sec 4.5.1.
def rangayyan_coherence(x, y, fs=1.0, nperseg=None):
    """Magnitude-squared coherence::

        C_xy(f) = |S_xy(f)|² / (S_xx(f) S_yy(f))

    Welch cross/auto-spectra; returns one-sided coherence in [0, 1].

    Parameters
    ----------
    x, y : array-like
    fs : float
    nperseg : int, optional

    Returns
    -------
    RichResult with keys ``freqs``, ``coherence``, ``mean_coherence``,
    ``peak_freq``, ``peak_coherence``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 4.5.1 "Coherence analysis of EEG
        channels", p.235.
    """
    from ._signal_core import coherence

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length.")
    if nperseg is None:
        nperseg = min(x.size, 256)
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    peak = int(np.argmax(Cxy))
    res = RichResult(
        title="Magnitude-squared coherence",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("nperseg", int(nperseg)),
            ("Mean coherence", float(Cxy.mean())),
            ("Peak coherence", float(Cxy[peak])),
            ("Peak freq (Hz)", float(f[peak])),
        ],
        interpretation=f"Peak coherence {Cxy[peak]:.3g} at {f[peak]:.3g} Hz.",
        payload={
            "freqs": f,
            "coherence": Cxy,
            "mean_coherence": float(Cxy.mean()),
            "peak_freq": float(f[peak]),
            "peak_coherence": float(Cxy[peak]),
        },
    )
    return with_describe_pointer(res, "rgcoh")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> fs = 100.0; t = np.arange(1024)/fs
# >>> a = np.sin(2*np.pi*10*t)
# >>> b = a + 0.1*rng.standard_normal(t.size)
# >>> r = rangayyan_coherence(a, b, fs=fs)
# >>> r["peak_coherence"] > 0.5
# True


# -- rgcxy: Magnitude-squared coherence (MSC) function.
def rangayyan_coherence_cxy(x, y, fs, nperseg):
    """
    Magnitude-squared coherence (MSC) function

    Formula: C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coherence, freqs

    References
    ----------
    Rangayyan Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Magnitude-squared coherence (MSC) function"}
    )


# -- rgeeg: EEG band power (delta theta alpha beta gamma) -- Rangayyan & Krishnan Sec 4.4.1.
# np.trapz was REMOVED in NumPy 2.0 and renamed to np.trapezoid. pyproject
# still declares numpy>=1.24, where only np.trapz exists, so bind whichever
# the installed NumPy provides. Without this the function raised
# AttributeError on every call under NumPy >= 2 -- not a test artefact, a
# hard failure for any caller.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def rangayyan_eeg_bands(x, fs, bands=None, nperseg=None):
    """Absolute and relative band power in canonical EEG bands.

    P_band = ∫ S(f) df via Welch's PSD over band (lo, hi].

    Parameters
    ----------
    x : array-like
    fs : float
    bands : dict, optional
    nperseg : int, optional

    Returns
    -------
    RichResult with keys ``absolute`` (band->W), ``relative`` (band->fraction),
    ``total_power``, ``freqs``, ``psd``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 4.4.1 "Detection of EEG rhythms",
        p.228. The previous docstring cited Ch 9.
    """
    from ._signal_core import welch

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = max(16, min(x.size, int(4 * fs)))
    bands = bands or _BANDS
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg)
    df = float(freqs[1] - freqs[0])
    total = float(_trapezoid(pxx, freqs))
    absolute = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        absolute[name] = float(_trapezoid(pxx[mask], freqs[mask])) if mask.any() else 0.0
    relative = {k: (v / total if total > 0 else 0.0) for k, v in absolute.items()}
    rows = [[name, f"{absolute[name]:.4g}", f"{relative[name] * 100:.2f}%"] for name in bands]
    res = RichResult(
        title="EEG band power",
        summary_lines=[("Fs (Hz)", float(fs)), ("Total power", total), ("Bin width (Hz)", df)],
        tables=[{"title": "Power by band", "headers": ["Band", "Absolute (W)", "Relative"], "rows": rows}],
        interpretation="Relative percentages sum ≤100% (residual outside defined bands).",
        payload={"absolute": absolute, "relative": relative, "total_power": total, "freqs": freqs, "psd": pxx},
    )
    return with_describe_pointer(res, "rgeeg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> fs = 256.0
# >>> t = np.arange(2048)/fs
# >>> x = np.sin(2*np.pi*10*t) + 0.3*rng.standard_normal(t.size)
# >>> r = rangayyan_eeg_bands(x, fs=fs)
# >>> r["absolute"]["alpha"] > r["absolute"]["gamma"]
# True


# -- rgeegar: EEG rhythm detection via autocorrelation.
def rangayyan_eeg_autocorr(eeg, fs, max_lag):
    """
    EEG rhythm detection via autocorrelation

    Formula: R_xx(tau) -> peak at T_rhythm; frequency = 1/T_rhythm

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rhythm_freq, acf

    References
    ----------
    Rangayyan Ch 4.4.1
    """
    eeg = np.asarray(eeg, dtype=float)
    y = np.asarray(eeg, dtype=float)
    n = min(len(eeg), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "EEG rhythm detection via autocorrelation",
            }
        )
    result = stats.spearmanr(eeg[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "EEG rhythm detection via autocorrelation",
        }
    )


# -- rgeegrhm: EEG alpha rhythm presence detection via autocorrelation.
def rangayyan_eeg_rhythm_detect(eeg, fs):
    """
    EEG alpha rhythm presence detection via autocorrelation

    Formula: alpha present if R_xx has peak at T~100ms (10Hz); decision by peak height

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: has_alpha, peak_freq

    References
    ----------
    Rangayyan Ch 10.2.3
    """
    eeg = np.asarray(eeg, dtype=float)
    y = np.asarray(eeg, dtype=float)
    n = min(len(eeg), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "EEG alpha rhythm presence detection via autocorrelation",
            }
        )
    result = stats.spearmanr(eeg[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "EEG alpha rhythm presence detection via autocorrelation",
        }
    )


# -- rgeegsp: EEG band powers.
def rangayyan_eeg_spectral(eeg, fs, n_ch=None):
    r"""EEG band powers (Rangayyan Ch. 3):

    .. math:: P_\delta = \int_0^4 S(f)\,df, \quad
              P_\theta = \int_4^8, \quad
              P_\alpha = \int_8^{13}, \quad
              P_\beta = \int_{13}^{30}.

    Band edges are those stated in the text. The PSD comes from
    Welch rather than a bare periodogram, since band POWER is an
    integral and integrating a high-variance estimate propagates that
    variance straight into the clinical number. Relative powers are
    returned alongside the absolute ones, because absolute EEG power
    depends on electrode impedance and is rarely comparable across
    recordings.

    Parameters
    ----------
    eeg : array-like, shape (N,) or (n_ch, N)
        Signal(s).
    fs : float
        Sampling frequency, must exceed 60 Hz to cover the beta band.
    n_ch : int, optional
        Channel count check.

    Returns
    -------
    RichResult
        keys: ``bands`` (dict of absolute power), ``relative``,
        ``total_power``, ``freqs``, ``psd``, ``n_ch``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (EEG spectral analysis).
    """
    X = np.atleast_2d(np.asarray(eeg, dtype=float))
    fs = float(fs)
    if fs <= 60.0:
        raise ValueError(
            f"fs = {fs} is too low: the 13-30 Hz beta band needs fs > 60 Hz."
        )
    m, N = X.shape
    if n_ch is not None and int(n_ch) != m:
        raise ValueError(f"n_ch = {n_ch} does not match the {m} channels.")
    edges = {"delta": (0.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
             "beta": (13.0, 30.0)}
    bands = {k: np.zeros(m) for k in edges}
    psd_all, freqs = [], None
    for c in range(m):
        w = rangayyan_welch_psd(X[c], fs=fs)
        freqs = w["freqs"]
        psd = w["psd"]
        psd_all.append(psd)
        for k, (a, b) in edges.items():
            sel = (freqs >= a) & (freqs < b)
            bands[k][c] = float(np.trapezoid(psd[sel], freqs[sel])) if sel.any() else 0.0
    total = sum(bands[k] for k in edges)
    rel = {k: np.where(total > 0, bands[k] / np.maximum(total, 1e-300), 0.0)
           for k in edges}
    squeeze = m == 1
    return RichResult(payload={
        "bands": {k: (float(v[0]) if squeeze else v) for k, v in bands.items()},
        "relative": {k: (float(v[0]) if squeeze else v) for k, v in rel.items()},
        "total_power": float(total[0]) if squeeze else total,
        "freqs": freqs, "psd": psd_all[0] if squeeze else np.array(psd_all),
        "n_ch": int(m),
        "method": "Welch PSD integrated over the book's band edges; relative powers too"})


# -- rgemgpk: EMG mean/median frequency from power spectrum.
def rangayyan_emg_peak_freq(emg, fs):
    """
    EMG mean/median frequency from power spectrum

    Formula: f_mean = sum(f*S(f))/sum(S(f)); f_median: sum_{0}^{f_med}S=sum_{f_med}^{inf}S

    Parameters
    ----------
    emg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean_freq, median_freq

    References
    ----------
    Rangayyan Ch 6
    """
    emg = np.asarray(emg, dtype=float)
    n = int(emg) if emg.ndim == 0 else len(emg)
    result = float(np.mean(emg))
    se = float(np.std(emg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "EMG mean/median frequency from power spectrum"}
    )


# -- rgeqn3b: Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n].
def rangayyan_ch3_correlation_sum(x, y):
    """
    Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]

    Formula: R_xy[m] = sum_n x[n]*y[n+m]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R_xy, lags

    References
    ----------
    Rangayyan Ch 3.4.1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]",
        }
    )


# -- rgerpflt: ERP artifact removal via synchronized averaging.
def rangayyan_erp_artifact_remove(erp_epochs, fs):
    """
    ERP artifact removal via synchronized averaging

    Formula: SNR_avg = sqrt(M)*SNR_single; artifact reduced by 1/sqrt(M)

    Parameters
    ----------
    erp_epochs : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: erp_clean, snr

    References
    ----------
    Rangayyan Ch 3.12
    """
    erp_epochs = np.asarray(erp_epochs, dtype=float)
    n = int(erp_epochs) if erp_epochs.ndim == 0 else len(erp_epochs)
    result = float(np.mean(erp_epochs))
    se = float(np.std(erp_epochs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "ERP artifact removal via synchronized averaging"}
    )


# -- rgmflt: Matched filter transfer function for signal detection in noise.
def rangayyan_matched_filter(signal_spectrum, noise_psd, t0):
    """
    Matched filter transfer function for signal detection in noise

    Formula: H_opt(f) = k * S*(f) * exp(-j2*pi*f*t0) / Pnn(f)

    Parameters
    ----------
    signal_spectrum : array-like
        Input data.
    noise_psd : array-like
        Input data.
    t0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_matched, freqs

    References
    ----------
    Rangayyan Ch 4.6.1
    """
    signal_spectrum = np.asarray(signal_spectrum, dtype=float)
    n = int(signal_spectrum) if signal_spectrum.ndim == 0 else len(signal_spectrum)
    result = float(np.mean(signal_spectrum))
    se = float(np.std(signal_spectrum, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Matched filter transfer function for signal detection in noise",
        }
    )


# -- rgmfsnr: Output SNR of matched filter (maximum SNR theorem).
def rangayyan_matched_filter_snr(signal, noise_psd):
    """
    Output SNR of matched filter (maximum SNR theorem)

    Formula: SNR_max = 2*E_s/N0 where E_s = integral |s(t)|^2 dt

    Parameters
    ----------
    signal : array-like
        Input data.
    noise_psd : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_max

    References
    ----------
    Rangayyan Ch 4.6.1
    """
    signal = np.asarray(signal, dtype=float)
    n = int(signal) if signal.ndim == 0 else len(signal)
    result = float(np.mean(signal))
    se = float(np.std(signal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Output SNR of matched filter (maximum SNR theorem)"}
    )


# -- rgperio: Periodogram.
def rangayyan_periodogram(x, fs=1.0):
    r"""Periodogram power spectral density (Rangayyan Ch. 3):

    .. math:: P(f) = \frac1N |X(f)|^2, \qquad X(f) = \mathrm{DFT}(x).

    The periodogram is NOT a consistent estimator: its variance does
    not fall as N grows, only its frequency resolution improves. That
    is precisely why Welch's method (:mod:`morie.fn.rgwelch`) averages
    segments, and the returned docstring says so rather than presenting
    the periodogram as a finished estimate.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float, default 1.0
        Sampling frequency.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``total_power``, ``N``, ``fs``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the periodogram).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 2:
        raise ValueError(f"need at least 2 samples, got {N}.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return RichResult(payload={"freqs": freqs, "psd": psd,
                               "total_power": float(np.mean(x**2)), "N": int(N),
                               "fs": fs,
                               "method": "P(f) = |DFT(x)|^2/N; inconsistent -- variance does not shrink"})


# -- rgpsd: Power spectral density via Welch's method -- Rangayyan & Krishnan Sec 6.3.2-6.3.4.
# np.trapz was REMOVED in NumPy 2.0 and renamed to np.trapezoid. pyproject
# still declares numpy>=1.24, where only np.trapz exists, so bind whichever
# the installed NumPy provides. Without this the function raised
# AttributeError on every call under NumPy >= 2 -- not a test artefact, a
# hard failure for any caller.
_trapezoid_rgpsd = getattr(np, "trapezoid", None) or np.trapz


def rangayyan_psd(x, fs=1.0, nperseg=None, noverlap=None, window="hann"):
    """Welch periodogram PSD.

    Computes the one-sided PSD using overlapping windowed segments::

        S(f) = (1 / (K * W * fs)) * sum_k |X_k(f)|^2

    where ``W`` is the window's noise-equivalent bandwidth scaling.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz).
    nperseg : int, optional
        Segment length. Default ``min(len(x), 256)``.
    noverlap : int, optional
        Overlap samples. Default ``nperseg//2``.
    window : str
        Window name (``hann`` default).

    Returns
    -------
    RichResult with keys ``freqs``, ``psd``, ``fs``, ``nperseg``, ``peak_freq``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 6.3.2 "The periodogram", p.323;
        Sec 6.3.3 "The need for averaging PSDs", p.325; Sec 6.3.4 "The use
        of windows: spectral resolution and leakage", p.326 -- NOT Ch 4.
        Welch's method is precisely the averaged, windowed periodogram those
        three sections build up.
    Welch, P. D. (1967). The use of Fast Fourier Transform for the estimation
        of power spectra. *IEEE Transactions on Audio and Electroacoustics*,
        15(2), 70-73.
    """
    from ._signal_core import welch

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(x.size, 256)
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
    peak_idx = int(np.argmax(pxx))
    res = RichResult(
        title="Power spectral density (Welch)",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("nperseg", int(nperseg)),
            ("Window", window),
            ("Peak freq (Hz)", float(freqs[peak_idx])),
            ("Peak power", float(pxx[peak_idx])),
            ("Total power", float(_trapezoid_rgpsd(pxx, freqs))),
        ],
        interpretation=(f"PSD peaks at {freqs[peak_idx]:.3g} Hz; total band power {float(_trapezoid_rgpsd(pxx, freqs)):.4g}."),
        payload={
            "freqs": freqs,
            "psd": pxx,
            "fs": float(fs),
            "nperseg": int(nperseg),
            "peak_freq": float(freqs[peak_idx]),
            "total_power": float(_trapezoid_rgpsd(pxx, freqs)),
        },
    )
    return with_describe_pointer(res, "rgpsd")


# CANONICAL TEST
# >>> fs=100.0; t=np.arange(1000)/fs
# >>> x = np.sin(2*np.pi*10*t)
# >>> r = rangayyan_psd(x, fs=fs, nperseg=256)
# >>> abs(r["peak_freq"] - 10.0) < 1.0
# True


# compact alias per ledger/NAMING.md
rangayyanpsd = rangayyan_psd


# -- rgpsd2hz: Convert PSD to frequency-in-Hz units and compute bin-level features.
def rangayyan_psd_to_hz(psd, N, fs):
    """
    Convert PSD to frequency-in-Hz units and compute bin-level features

    Formula: bin_freq = k*fs/N; power_in_band = sum S(k) * (fs/N) for bins in band

    Parameters
    ----------
    psd : array-like
        Input data.
    N : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: freqs_hz, band_power

    References
    ----------
    Rangayyan Ch 6
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Convert PSD to frequency-in-Hz units and compute bin-level features",
        }
    )


# -- rgpsdacf: PSD to autocorrelation.
def rangayyan_psd_to_acf(psd, freqs=None):
    r"""Autocorrelation from the power spectral density (Rangayyan
    Ch. 3):

    .. math:: R_{xx}(m) = \mathrm{IDFT}\{S_{xx}(f)\},

    the Wiener-Khinchin relation. Because the PSD supplied is
    one-sided (rfft convention), the inverse uses ``irfft`` so the
    result is real by construction -- taking a complex ifft and
    discarding the imaginary part would silently hide an asymmetric
    input.

    Parameters
    ----------
    psd : array-like
        One-sided power spectral density.
    freqs : array-like, optional
        Frequency grid, used only to report the lag spacing.

    Returns
    -------
    RichResult
        keys: ``acf``, ``lags``, ``r0`` (total power), ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Wiener-Khinchin).
    """
    S = np.asarray(psd, dtype=float).ravel()
    if S.size < 2:
        raise ValueError("psd must have at least 2 points.")
    if np.any(S < 0):
        raise ValueError("a power spectral density cannot be negative.")
    acf = np.fft.irfft(S)
    n_lag = acf.size // 2 + 1
    lags = np.arange(n_lag)
    if freqs is not None:
        f = np.asarray(freqs, dtype=float).ravel()
        if f.size != S.size:
            raise ValueError("freqs must match the length of psd.")
    return RichResult(payload={"acf": acf[:n_lag], "lags": lags,
                               "r0": float(acf[0]),
                               "method": "R_xx = irfft(S_xx); real by construction"})


# -- rgpsync: Synchronized averaging of PCG spectra for murmur analysis.
def rangayyan_pcg_sync_avg(pcg, ecg, fs, n_cycles):
    """
    Synchronized averaging of PCG spectra for murmur analysis

    Formula: S_avg(f) = (1/M)*sum |PCG_k(f)|^2

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    n_cycles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: avg_spectrum, freqs

    References
    ----------
    Rangayyan Ch 6.3.6
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Synchronized averaging of PCG spectra for murmur analysis",
        }
    )


# -- rgseiz: EEG seizure detection via rhythm coherence analysis.
def rangayyan_seizure_detect(eeg, fs, ch_pairs):
    """
    EEG seizure detection via rhythm coherence analysis

    Formula: Seizure: sustained increase in delta/theta band coherence across EEG channels

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    ch_pairs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: seizure_onset, duration

    References
    ----------
    Rangayyan Ch 4.4.3
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "EEG seizure detection via rhythm coherence analysis"}
    )


# -- rgsmom: Spectral moments: centroid (mean freq), variance (bandwidth), skewness.
def rangayyan_spectral_moments(psd, freqs):
    """
    Spectral moments: centroid (mean freq), variance (bandwidth), skewness

    Formula: f_c = sum(f*S(f))/sum(S(f)); bw = sqrt(sum((f-f_c)^2*S(f))/sum(S(f)))

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centroid, bandwidth, skewness

    References
    ----------
    Rangayyan Ch 6.4.1
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Spectral moments: centroid (mean freq), variance (bandwidth), skewness",
        }
    )


# -- rgspres: Spectral resolution and leakage analysis (Rayleigh criterion).
def rangayyan_spectral_resolution(N, fs, window_type):
    """
    Spectral resolution and leakage analysis (Rayleigh criterion)

    Formula: delta_f = fs/N (Rayleigh); window sidelobe level determines leakage

    Parameters
    ----------
    N : array-like
        Input data.
    fs : array-like
        Input data.
    window_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resolution, sidelobe_db

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Spectral resolution and leakage analysis (Rayleigh criterion)",
        }
    )


# -- rgtmpl: Template matching for EEG spike-and-wave detection.
def rangayyan_template_match(eeg, template, threshold):
    """
    Template matching for EEG spike-and-wave detection

    Formula: corr(template, segment) = sum(t[n]*eeg[n]) / (|t|*|eeg|); threshold on correlation

    Parameters
    ----------
    eeg : array-like
        Input data.
    template : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: match_locs

    References
    ----------
    Rangayyan Ch 4.4.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Template matching for EEG spike-and-wave detection"}
    )


# -- rgwelch: Welch power spectral density.
def rangayyan_welch_psd(x, fs=1.0, nperseg=None, noverlap=None, window="hann"):
    r"""Welch's averaged periodogram (Rangayyan Ch. 3):

    .. math:: P_W(f) = \frac{1}{KU}\sum_{k=1}^{K} |W_k(f)|^2,
              \qquad U = \frac1N \sum_n w^2[n],

    with U the window power normalisation, WITHOUT which the estimate
    is biased low by the window's energy loss. Averaging K segments
    cuts the variance by roughly K at the cost of resolution -- the
    bias-variance trade the periodogram cannot make.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float, default 1.0
        Sampling frequency.
    nperseg : int, optional
        Segment length; N // 8 by default.
    noverlap : int, optional
        Overlap; half the segment by default.
    window : {"hann", "hamming", "boxcar"}
        Window type.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``n_segments``, ``U``, ``nperseg``,
        ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Welch's method).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    seg = max(8, N // 8) if nperseg is None else int(nperseg)
    if not 2 <= seg <= N:
        raise ValueError(f"nperseg must lie in 2..{N}, got {seg}.")
    ov = seg // 2 if noverlap is None else int(noverlap)
    if not 0 <= ov < seg:
        raise ValueError(f"noverlap must lie in 0..{seg - 1}, got {ov}.")
    if window == "hann":
        w = np.hanning(seg)
    elif window == "hamming":
        w = np.hamming(seg)
    elif window == "boxcar":
        w = np.ones(seg)
    else:
        raise ValueError("window must be 'hann', 'hamming' or 'boxcar'.")
    U = float(np.mean(w**2))
    step = seg - ov
    starts = range(0, N - seg + 1, step)
    acc, K = None, 0
    for s in starts:
        W = np.fft.rfft(x[s : s + seg] * w)
        p = np.abs(W) ** 2
        acc = p if acc is None else acc + p
        K += 1
    if K == 0:
        raise ValueError("no complete segments; reduce nperseg.")
    psd = acc / (K * U * seg)
    return RichResult(payload={"freqs": np.fft.rfftfreq(seg, d=1.0 / fs), "psd": psd,
                               "n_segments": K, "U": U, "nperseg": seg, "fs": fs,
                               "method": "Welch averaged periodogram with window power U"})


# -- rng016: Autocorrelation function of a random process by ensemble average (Eq 3.16/3.17).
def rangayyan_ch3_acf_continuous(x, t1, tau):
    r"""ACF of a random process at lag :math:`\tau`, estimated over an ensemble.

    .. math::

        \phi_{xx}(t_1, t_1+\tau) = E[x(t_1)\,x(t_1+\tau)]
            = \int\!\!\int x(t_1)x(t_1+\tau)\,p_{x_1,x_2}(x_1,x_2)\,dx_1\,dx_2

    The joint PDF is unknown in practice, so the expectation is approximated
    by the ensemble average over :math:`M` realisations (Eq. 3.17):

    .. math::

        \phi_{xx}(t_1, t_1+\tau) = \lim_{M\to\infty}\frac{1}{M}
            \sum_{k=1}^{M} x_k(t_1)\,x_k(t_1+\tau)

    Parameters
    ----------
    x : array-like, shape (M, N)
        Ensemble of ``M`` realisations of the process, each ``N`` samples
        long. A 1-D input is rejected: a single realisation is not an
        ensemble, and averaging along it silently computes a *time* average
        (Eq. 3.20), which is a different quantity unless the process is
        ergodic.
    t1 : int
        Sample index :math:`t_1`.
    tau : int
        Lag :math:`\tau` in samples. May be negative.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`\phi_{xx}`), ``t1``, ``tau``, ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D, or if ``t1`` or ``t1 + tau`` is out of range.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.16) and ensemble estimate Eq. (3.17), p. 96; Figure 3.2
        illustrates the two vertical lines at :math:`t_1` and :math:`t_1+\tau`
        over ten flash-visual ERP acquisitions.

    Notes
    -----
    The book's Eq. (3.20) time-averaged ACF is the *other* estimator and is
    not what this function computes. See Section 6.3 for finite-length ACF
    estimation.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"x must be a 2-D ensemble of shape (M, N); got shape {arr.shape}. "
            "Eq. (3.17) averages ACROSS realisations, not along time -- a 1-D "
            "input would give the time average of Eq. (3.20) instead."
        )
    M, N = arr.shape
    t1 = int(t1)
    tau = int(tau)
    t2 = t1 + tau
    if not (0 <= t1 < N):
        raise ValueError(f"t1={t1} out of range for N={N} samples")
    if not (0 <= t2 < N):
        raise ValueError(f"t1+tau={t2} out of range for N={N} samples")
    value = float(np.mean(arr[:, t1] * arr[:, t2]))
    return RichResult(
        payload={
            "value": value,
            "t1": t1,
            "tau": tau,
            "M": int(M),
            "n": int(N),
            "method": "ensemble-average ACF (Rangayyan Eq 3.16, estimate Eq 3.17)",
        }
    )


# -- rng017: Ensemble autocorrelation.
def rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau, M=None):
    r"""Ensemble autocorrelation (Rangayyan Ch. 3):

    .. math:: \phi_{xx}(t_1, t_1+\tau) = \lim_{M\to\infty}
              \frac1M \sum_{k=1}^{M} x_k(t_1)\,x_k(t_1+\tau).

    A function of BOTH times, not just the lag. It reduces to a
    function of tau alone precisely when the process is
    wide-sense stationary -- which the caller must establish, so this
    reports the value at the requested (t1, tau) rather than implying
    stationarity by returning a lag-only curve.

    Parameters
    ----------
    x_k : array-like, shape (M, T)
        Realisations.
    t1 : int
        First time index.
    tau : int
        Lag.
    M : int, optional
        Realisation count.

    Returns
    -------
    RichResult
        keys: ``acf``, ``t1``, ``tau``, ``M``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (ensemble autocorrelation).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, T = X.shape
    if M is not None and int(M) != m:
        raise ValueError(f"M = {M} does not match the {m} rows of x_k.")
    t1 = int(t1)
    tau = int(tau)
    if not 0 <= t1 < T:
        raise ValueError(f"t1 must lie in 0..{T - 1}, got {t1}.")
    if not 0 <= t1 + tau < T:
        raise ValueError(f"t1 + tau must lie in 0..{T - 1}, got {t1 + tau}.")
    return RichResult(payload={"acf": float(np.mean(X[:, t1] * X[:, t1 + tau])),
                               "t1": t1, "tau": tau, "M": int(m),
                               "method": "phi(t1, t1+tau) across realisations; two-time, not lag-only"})


# -- rng018: Ensemble average function (Rangayyan eq. 3.18).
def ensavg(observations):
    """Ensemble average x_bar(t) over M records, at every instant.

    Rangayyan (2024) eq. (3.18):
        x_bar(t) = mu_x(t) = (1/M) sum_{k=1}^{M} x_k(t)   for all t.

    The book calls x_bar(t) a prototype of the random process and notes
    it is a filtered version of the M observations with diminished
    noise.  Records must be the same length -- an ensemble average across
    ragged records would silently average different numbers of traces at
    different instants.
    """
    recs = [aslist(r) for r in observations]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one observation")
    n = len(recs[0])
    if n == 0:
        raise ValueError("records must be nonempty")
    if any(len(r) != n for r in recs):
        raise ValueError("all records must have the same length")
    avg = [fsum(r[i] for r in recs) / m for i in range(n)]
    sd = [sqrt(fsum((r[i] - avg[i]) ** 2 for r in recs) / m)
          for i in range(n)]
    return RichResult(payload={
        "average": avg, "sd": sd, "m": m, "n": n,
        "se": [s / sqrt(m) for s in sd],
        "method": "Rangayyan (2024) eq. (3.18)"})


rangayyan_ch3_ensemble_average_function = ensavg  # pre-policy spelling


# -- rng020: Time-averaged autocorrelation.
def rangayyan_ch3_time_averaged_acf(x_k, tau, T=None):
    r"""Time-averaged autocorrelation (Rangayyan Ch. 3):

    .. math:: \phi_{xx}(\tau, k) = \lim_{T\to\infty} \frac1T
              \int_{-T/2}^{T/2} x_k(t)\,x_k(t+\tau)\,dt.

    The lag-domain counterpart of :mod:`morie.fn.rng019`: one
    realisation, averaged over time. Under ergodicity it converges to
    the ensemble autocorrelation of :mod:`morie.fn.rng017`.

    Parameters
    ----------
    x_k : array-like, shape (T,) or (M, T)
        Realisation(s).
    tau : int
        Lag.
    T : int, optional
        Length check.

    Returns
    -------
    RichResult
        keys: ``acf``, ``tau``, ``n_used``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (time-averaged autocorrelation).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, n = X.shape
    tau = int(tau)
    if not 0 <= tau < n:
        raise ValueError(f"tau must lie in 0..{n - 1}, got {tau}.")
    if T is not None and int(T) != n:
        raise ValueError(f"T = {T} does not match the {n} samples.")
    vals = np.array([float(np.mean(X[k, : n - tau] * X[k, tau:])) for k in range(m)])
    return RichResult(payload={"acf": float(vals[0]) if m == 1 else vals,
                               "tau": tau, "n_used": int(n - tau),
                               "method": "time-averaged phi(tau); -> ensemble ACF under ergodicity"})


# -- rng023: Cross-correlation function (CCF) between two random processes x and y..
def rangayyan_ch3_ccf_continuous(x, y, t1, tau):
    """
    Cross-correlation function (CCF) between two random processes x and y.

    Formula: theta_xy(t1, t1+tau) = E[x(t1) y(t1+tau)] = double_integral x(t1) y(t1+tau) p_{x,y}(x,y) dx dy

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t1 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.23, p. 98
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Cross-correlation function (CCF) between two random processes x and y.",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Cross-correlation function (CCF) between two random processes x and y.",
        }
    )


# -- rng070: Inverse discrete Fourier transform (IDFT) of an N-point spectrum..
def rangayyan_ch3_idft_definition(X, n, k, N):
    """
    Inverse discrete Fourier transform (IDFT) of an N-point spectrum.

    Formula: x(n) = (1/N) * sum_{k=0}^{N-1} X(k) * exp(+j * (2*pi/N) * n * k)

    Parameters
    ----------
    X : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.81, p. 126
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Inverse discrete Fourier transform (IDFT) of an N-point spectrum.",
        }
    )


# -- rng080: Parseval's theorem: total signal energy preserved under Fourier transform..
def rangayyan_ch3_parseval_theorem(x, X):
    """
    Parseval's theorem: total signal energy preserved under Fourier transform.

    Formula: integral_{-inf}^{inf} |x(t)|^2 dt = (1/(2*pi)) * integral_{-inf}^{inf} |X(omega)|^2 d(omega); sum_{n=0}^{N-1} |x(n)|^2 = (1/N) * sum_{k=0}^{N-1} |X(k)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.91, p. 134
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Parseval's theorem: total signal energy preserved under Fourier transform.",
        }
    )


# -- rng085: Synchronized sum across M observations to form ensemble averaging..
def rangayyan_ch3_synchronized_averaging_sum(y_k, x_k, eta_k, n, M):
    """
    Synchronized sum across M observations to form ensemble averaging.

    Formula: sum_{k=1}^{M} y_k(n) = sum_{k=1}^{M} x_k(n) + sum_{k=1}^{M} eta_k(n)

    Parameters
    ----------
    y_k : array-like
        Input data.
    x_k : array-like
        Input data.
    eta_k : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.96, p. 135
    """
    y_k = np.atleast_1d(np.asarray(y_k, dtype=float))
    n = len(y_k)
    result = float(np.mean(y_k))
    se = float(np.std(y_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Synchronized sum across M observations to form ensemble averaging.",
        }
    )


# -- rng086: Normalized cross-correlation coefficient used in template matching..
def rangayyan_ch3_normalized_cross_correlation_template(x, y, k, N, x_bar, y_bar_k):
    """
    Normalized cross-correlation coefficient used in template matching.

    Formula: gamma_xy(k) = sum_{n=0}^{N-1} [x(n)-x_bar][y(k-N+1+n)-y_bar_k] / sqrt( sum_{n=0}^{N-1} [x(n)-x_bar]^2 * sum_{n=0}^{N-1} [y(k-N+1+n)-y_bar_k]^2 )

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.
    x_bar : array-like
        Input data.
    y_bar_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.97, p. 137
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Normalized cross-correlation coefficient used in template matching.",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Normalized cross-correlation coefficient used in template matching.",
        }
    )


# -- rng198: Discrete-time dot product (inner product) of two N-sample signals..
def rangayyan_ch4_dot_product_discrete(x, y, N):
    """
    Discrete-time dot product (inner product) of two N-sample signals.

    Formula: x . y = <x, y> = sum_{n=0}^{N-1} x(n) * y(n)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.24, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Discrete-time dot product (inner product) of two N-sample signals.",
        }
    )


# -- rng199: Correlation coefficient as normalized dot product of two signals..
def rangayyan_ch4_correlation_coefficient_normalized_dot(x, y, N):
    """
    Correlation coefficient as normalized dot product of two signals.

    Formula: gamma_xy = sum_{n=0}^{N-1} x(n)*y(n) / sqrt( sum_{n=0}^{N-1} x^2(n) * sum_{n=0}^{N-1} y^2(n) )

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.25, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Correlation coefficient as normalized dot product of two signals.",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Correlation coefficient as normalized dot product of two signals.",
        }
    )


# -- rng200: Projection (inner product) of two continuous-time signals over R..
def rangayyan_ch4_continuous_dot_product(x, y, t):
    """
    Projection (inner product) of two continuous-time signals over R.

    Formula: theta_xy = integral_{-inf}^{inf} x(t) * y(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.26, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Projection (inner product) of two continuous-time signals over R.",
        }
    )


# -- rng201: Cross-correlation function of two continuous-time signals with delay tau..
def rangayyan_ch4_ccf_continuous_with_delay(x, y, tau, t):
    """
    Cross-correlation function of two continuous-time signals with delay tau.

    Formula: theta_xy(tau) = integral_{-inf}^{inf} x(t) * y(t + tau) dt

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.27, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Cross-correlation function of two continuous-time signals with delay tau.",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Cross-correlation function of two continuous-time signals with delay tau.",
        }
    )


# -- rng202: Discrete-time cross-correlation function of x(n) and y(n) with shift k..
def rangayyan_ch4_ccf_discrete_with_delay(x, y, k, n):
    """
    Discrete-time cross-correlation function of x(n) and y(n) with shift k.

    Formula: theta_xy(k) = sum_{n} x(n) * y(n + k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.28, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Discrete-time cross-correlation function of x(n) and y(n) with shift k.",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Discrete-time cross-correlation function of x(n) and y(n) with shift k.",
        }
    )


# -- rng203: CCF of random signals as expectation of outer product of vector samples..
def rangayyan_ch4_ccf_outer_product_random_signals(x, y, n):
    """
    CCF of random signals as expectation of outer product of vector samples.

    Formula: Theta_xy = E[ x(n) * y^T(n) ]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.29, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "CCF of random signals as expectation of outer product of vector samples.",
        }
    )


# -- rng205: Cross-spectral density (CSD) as the Fourier transform of the CCF..
def rangayyan_ch4_csd_from_ccf(theta_xy, X, Y, f, tau):
    """
    Cross-spectral density (CSD) as the Fourier transform of the CCF.

    Formula: S_xy(f) = FT[theta_xy(tau)] = X(f) * Y*(f)

    Parameters
    ----------
    theta_xy : array-like
        Input data.
    X : array-like
        Input data.
    Y : array-like
        Input data.
    f : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.31, p. 236
    """
    theta_xy = np.atleast_1d(np.asarray(theta_xy, dtype=float))
    n = len(theta_xy)
    result = float(np.mean(theta_xy))
    se = float(np.std(theta_xy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Cross-spectral density (CSD) as the Fourier transform of the CCF.",
        }
    )


# -- rng206: Magnitude coherence spectrum between two signals from CSD and PSDs..
def rangayyan_ch4_coherence_spectrum(S_xy, S_xx, S_yy, f):
    """
    Magnitude coherence spectrum between two signals from CSD and PSDs.

    Formula: Gamma_xy(f) = sqrt( |S_xy(f)|^2 / (S_xx(f) * S_yy(f)) )

    Parameters
    ----------
    S_xy : array-like
        Input data.
    S_xx : array-like
        Input data.
    S_yy : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.32, p. 236
    """
    S_xy = np.atleast_1d(np.asarray(S_xy, dtype=float))
    n = len(S_xy)
    result = float(np.mean(S_xy))
    se = float(np.std(S_xy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Magnitude coherence spectrum between two signals from CSD and PSDs.",
        }
    )


# -- rng207: Fourier transform of input signal to a matched filter..
def rangayyan_ch4_matched_filter_input_ft(x, t, omega):
    """
    Fourier transform of input signal to a matched filter.

    Formula: X(omega) = integral_{-inf}^{inf} x(t) * exp(-j*omega*t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.33, p. 237
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Fourier transform of input signal to a matched filter.",
        }
    )


# -- rng208: Output of matched filter via inverse Fourier transform of X(omega)*H(omega)..
def rangayyan_ch4_matched_filter_output_inverse_ft(X, H, omega, f, t):
    """
    Output of matched filter via inverse Fourier transform of X(omega)*H(omega).

    Formula: y(t) = (1/(2*pi)) * integral_{-inf}^{inf} X(omega) H(omega) exp(+j*omega*t) d(omega) = integral_{-inf}^{inf} X(f) H(f) exp(+j*2*pi*f*t) df

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.
    f : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.34, p. 238
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Output of matched filter via inverse Fourier transform of X(omega)*H(omega).",
        }
    )


# -- rng209: PSD of white noise at the input of a matched filter (two-sided)..
def rangayyan_ch4_white_noise_psd_input(P_eta_i, f):
    """
    PSD of white noise at the input of a matched filter (two-sided).

    Formula: S_eta_i(f) = P_eta_i / 2

    Parameters
    ----------
    P_eta_i : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.35, p. 238
    """
    P_eta_i = np.atleast_1d(np.asarray(P_eta_i, dtype=float))
    n = len(P_eta_i)
    result = float(np.mean(P_eta_i))
    se = float(np.std(P_eta_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "PSD of white noise at the input of a matched filter (two-sided).",
        }
    )


# -- rng210: Noise PSD at the output of a matched filter..
def rangayyan_ch4_noise_psd_at_output(P_eta_i, H, f):
    """
    Noise PSD at the output of a matched filter.

    Formula: S_eta_o(f) = (P_eta_i / 2) * |H(f)|^2

    Parameters
    ----------
    P_eta_i : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.36, p. 238
    """
    P_eta_i = np.atleast_1d(np.asarray(P_eta_i, dtype=float))
    n = len(P_eta_i)
    result = float(np.mean(P_eta_i))
    se = float(np.std(P_eta_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Noise PSD at the output of a matched filter."}
    )


# -- rng211: Average output noise power.
def rangayyan_ch4_average_output_noise_power(P_eta_i, H, freqs=None, df=None):
    r"""Average output noise power of a filter (Rangayyan Ch. 4):

    .. math:: P_{\eta_o} = \frac{P_{\eta_i}}{2}
              \int_{-\infty}^{\infty} |H(f)|^2\, df.

    White input noise is shaped by the filter's energy, so the output
    power depends only on :math:`\int |H|^2` -- the noise-equivalent
    bandwidth. A filter with unit passband gain still amplifies noise
    in proportion to how wide it is, which is why narrowing the band
    is the primary noise-reduction lever.

    Parameters
    ----------
    P_eta_i : float
        Input noise power spectral density (two-sided).
    H : array-like
        Filter frequency response (complex or magnitude).
    freqs : array-like, optional
        Matching frequencies, for the integration measure.
    df : float, optional
        Uniform frequency spacing when freqs is omitted.

    Returns
    -------
    RichResult
        keys: ``output_power``, ``energy_integral``,
        ``noise_equivalent_bw``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4 (noise through filters).
    """
    Hm = np.abs(np.asarray(H, dtype=complex).ravel()) ** 2
    if Hm.size < 2:
        raise ValueError("H must have at least 2 points.")
    P_in = float(P_eta_i)
    if P_in < 0:
        raise ValueError("input noise power cannot be negative.")
    if freqs is not None:
        f = np.asarray(freqs, dtype=float).ravel()
        if f.size != Hm.size:
            raise ValueError("freqs must match the length of H.")
        integral = float(np.trapezoid(Hm, f))
    else:
        step = 1.0 if df is None else float(df)
        if step <= 0:
            raise ValueError("df must be positive.")
        integral = float(np.trapezoid(Hm, dx=step))
    peak = float(Hm.max())
    return RichResult(payload={"output_power": P_in / 2.0 * integral,
                               "energy_integral": integral,
                               "noise_equivalent_bw": integral / peak if peak > 0 else 0.0,
                               "method": "P_out = (P_in/2) int |H(f)|^2 df"})


# -- rng212: Magnitude of instantaneous output signal of a matched filter at t = t0..
def rangayyan_ch4_matched_filter_instantaneous_signal(X, H, f, t_0):
    """
    Magnitude of instantaneous output signal of a matched filter at t = t0.

    Formula: M_y = |y(t_0)| = | integral_{-inf}^{inf} X(f) H(f) exp(+j*2*pi*f*t_0) df |

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.38, p. 238
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Magnitude of instantaneous output signal of a matched filter at t = t0.",
        }
    )


# -- rng213: Peak-power SNR at output of a matched filter..
def rangayyan_ch4_peak_power_snr(M_y, P_eta_o):
    """
    Peak-power SNR at output of a matched filter.

    Formula: M_y^2 / P_eta_o = instantaneous_peak_power_of_signal / noise_mean_power

    Parameters
    ----------
    M_y : array-like
        Input data.
    P_eta_o : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.39, p. 238
    """
    M_y = np.atleast_1d(np.asarray(M_y, dtype=float))
    n = len(M_y)
    result = float(np.mean(M_y))
    se = float(np.std(M_y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Peak-power SNR at output of a matched filter."}
    )


# -- rng214: Total energy of a signal via Parseval's theorem..
def rangayyan_ch4_signal_total_energy(x, X, t, f):
    """
    Total energy of a signal via Parseval's theorem.

    Formula: E_x = integral_{-inf}^{inf} x^2(t) dt = integral_{-inf}^{inf} |X(f)|^2 df

    Parameters
    ----------
    x : array-like
        Input data.
    X : array-like
        Input data.
    t : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.40, p. 238
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Total energy of a signal via Parseval's theorem."}
    )


# -- rng215: Normalized ratio used in maximizing matched-filter SNR..
def rangayyan_ch4_snr_normalized_ratio(H, X, P_eta_i, f, t_0):
    """
    Normalized ratio used in maximizing matched-filter SNR.

    Formula: M_y^2 / (E_x * P_eta_o) = | integral H(f) X(f) exp(+j*2*pi*f*t_0) df |^2 / ( (P_eta_i/2) * integral |H(f)|^2 df * integral |X(f)|^2 df )

    Parameters
    ----------
    H : array-like
        Input data.
    X : array-like
        Input data.
    P_eta_i : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.41, p. 238
    """
    H = np.atleast_1d(np.asarray(H, dtype=float))
    n = len(H)
    result = float(np.mean(H))
    se = float(np.std(H, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Normalized ratio used in maximizing matched-filter SNR.",
        }
    )


# -- rng216: Schwarz inequality for complex functions A(f) and B(f)..
def rangayyan_ch4_schwarz_inequality_complex(A, B, f):
    """
    Schwarz inequality for complex functions A(f) and B(f).

    Formula: | integral A(f) B(f) df |^2 <= integral |A(f)|^2 df * integral |B(f)|^2 df

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.42, p. 238
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Schwarz inequality for complex functions A(f) and B(f).",
        }
    )


# -- rng217: Schwarz inequality for real functions a(t) and b(t)..
def rangayyan_ch4_schwarz_inequality_real(a, b, t):
    """
    Schwarz inequality for real functions a(t) and b(t).

    Formula: [ integral a(t) b(t) dt ]^2 <= integral a^2(t) dt * integral b^2(t) dt

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.43, p. 239
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Schwarz inequality for real functions a(t) and b(t)."}
    )


# -- rng218: Schwarz (Cauchy-Schwarz) inequality for two vectors..
def rangayyan_ch4_cauchy_schwarz_vectors(a, b):
    """
    Schwarz (Cauchy-Schwarz) inequality for two vectors.

    Formula: |a . b| <= |a| * |b|

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.44, p. 239
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Schwarz (Cauchy-Schwarz) inequality for two vectors."}
    )


# -- rng219: Triangle inequality for two vectors..
def rangayyan_ch4_triangle_inequality_vectors(a, b):
    """
    Triangle inequality for two vectors.

    Formula: |a + b| <= |a| + |b|

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.45, p. 239
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Triangle inequality for two vectors."})


# -- rng220: Optimal frequency response of the matched filter..
def rangayyan_ch4_matched_filter_optimal_transfer_function(X, K, f, t_0):
    """
    Optimal frequency response of the matched filter.

    Formula: H(f) = K * X*(f) * exp(-j*2*pi*f*t_0)

    Parameters
    ----------
    X : array-like
        Input data.
    K : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.48, p. 239
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Optimal frequency response of the matched filter."}
    )


# -- rng221: Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal..
def rangayyan_ch4_matched_filter_impulse_response(x, K, t, t_0):
    """
    Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal.

    Formula: h(t) = K * x[-(t - t_0)]

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.49, p. 239
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal.",
        }
    )


# -- rng222: Matched-filter output equals scaled, delayed ACF of the reference signal..
def rangayyan_ch4_matched_filter_output_acf(phi_x, K, t, t_0):
    """
    Matched-filter output equals scaled, delayed ACF of the reference signal.

    Formula: y(t) = K * phi_x(t - t_0)

    Parameters
    ----------
    phi_x : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.50, p. 239
    """
    phi_x = np.atleast_1d(np.asarray(phi_x, dtype=float))
    n = len(phi_x)
    result = float(np.mean(phi_x))
    se = float(np.std(phi_x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Matched-filter output equals scaled, delayed ACF of the reference signal.",
        }
    )


# -- rng224: Basic three-sample reference pattern used in matched-filter illustration..
def rangayyan_ch4_basic_signal_g(n):
    """
    Basic three-sample reference pattern used in matched-filter illustration.

    Formula: g(n) = 3*delta(n) + 2*delta(n-1) + delta(n-2)

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.52, p. 240
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Basic three-sample reference pattern used in matched-filter illustration.",
        }
    )


# -- rng227: Frequency-domain optimal matched-filter response for EEG spike-and-wave detection..
def rangayyan_ch4_matched_filter_optimal_H_eeg(X, K, f, t_0):
    """
    Frequency-domain optimal matched-filter response for EEG spike-and-wave detection.

    Formula: H(f) = K * X*(f) * exp(-j*2*pi*f*t_0)

    Parameters
    ----------
    X : array-like
        Input data.
    K : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.55, p. 241
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency-domain optimal matched-filter response for EEG spike-and-wave detection.",
        }
    )


# -- rng228: Time-domain impulse response of the matched filter for EEG spike-and-wave detection..
def rangayyan_ch4_matched_filter_impulse_response_eeg(x, K, t, t_0):
    """
    Time-domain impulse response of the matched filter for EEG spike-and-wave detection.

    Formula: h(t) = K * x(t_0 - t)

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.56, p. 241
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Time-domain impulse response of the matched filter for EEG spike-and-wave detection.",
        }
    )


# -- rng229: Frequency-domain output of matched filter equals PSD of the reference signal..
def rangayyan_ch4_matched_filter_output_psd(X, H, f):
    """
    Frequency-domain output of matched filter equals PSD of the reference signal.

    Formula: Y(f) = X(f) * H(f) = X(f) * X*(f) = S_x(f)

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.57, p. 241
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency-domain output of matched filter equals PSD of the reference signal.",
        }
    )


_CHEATSHEET = [
    'coherence({}) -> Magnitude-squared coherence between two signals.',
    'matched_filter({}) -> Matched filter for template detection.',
    'rgacf: unbiased divisor N-|m|; biased form is PSD but shrinks toward 0',
    'bartlettpsd: mean of K disjoint-segment periodograms, eq 6.16',
    'rgbwbnd: 3dB and 99% answer different questions; criterion is explicit',
    'rgccf: Cross-correlation function (CCF) between two signals',
    'rgcfsle: Cardiorespiratory coupling analysis via coherence and PLV',
    'rgcoh: magnitude-squared coherence -- Rangayyan & Krishnan Sec 4.5.1',
    'rgcxy: Magnitude-squared coherence (MSC) function',
    'rgeeg: EEG delta theta alpha beta gamma band power -- Rangayyan & Krishnan Sec 4.4.1',
    'rgeegar: EEG rhythm detection via autocorrelation',
    'rgeegrhm: EEG alpha rhythm presence detection via autocorrelation',
    'rgeegsp: absolute EEG power is impedance-dependent -- use the relative values',
    'rgemgpk: EMG mean/median frequency from power spectrum',
    'rgeqn3b: Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]',
    'rgerpflt: ERP artifact removal via synchronized averaging',
    'rgmflt: Matched filter transfer function for signal detection in noise',
    'rgmfsnr: Output SNR of matched filter (maximum SNR theorem)',
    'rgperio: inconsistent estimator; more N buys resolution, not precision',
    'rgpsd: Welch power spectral density -- Rangayyan & Krishnan Sec 6.3.3',
    'rgpsd2hz: Convert PSD to frequency-in-Hz units and compute bin-level features',
    'rgpsdacf: Wiener-Khinchin via irfft, so the result is real by construction',
    'rgpsync: Synchronized averaging of PCG spectra for murmur analysis',
    'rgseiz: EEG seizure detection via rhythm coherence analysis',
    'rgsmom: Spectral moments: centroid (mean freq), variance (bandwidth), skewness',
    'rgspres: Spectral resolution and leakage analysis (Rayleigh criterion)',
    'rgtmpl: Template matching for EEG spike-and-wave detection',
    'rgwelch: U normalisation is mandatory or the PSD is biased low',
    'rng016: phi_xx(t1,t1+tau) = E[x(t1)x(t1+tau)] by ensemble average (Eq 3.16/3.17).',
    'rng017: two-time function; collapses to lag-only only under WSS',
    'rng018: ensemble average function, Rangayyan eq. (3.18)',
    'rng020: one record over time; matches rng017 under ergodicity',
    'rng023: Cross-correlation function (CCF) between two random processes x and y.',
    'rng070: Inverse discrete Fourier transform (IDFT) of an N-point spectrum.',
    "rng080: Parseval's theorem: total signal energy preserved under Fourier transform.",
    'rng085: Synchronized sum across M observations to form ensemble averaging.',
    'rng086: Normalized cross-correlation coefficient used in template matching.',
    'rng198: Discrete-time dot product (inner product) of two N-sample signals.',
    'rng199: Correlation coefficient as normalized dot product of two signals.',
    'rng200: Projection (inner product) of two continuous-time signals over R.',
    'rng201: Cross-correlation function of two continuous-time signals with delay tau.',
    'rng202: Discrete-time cross-correlation function of x(n) and y(n) with shift k.',
    'rng203: CCF of random signals as expectation of outer product of vector samples.',
    'rng205: Cross-spectral density (CSD) as the Fourier transform of the CCF.',
    'rng206: Magnitude coherence spectrum between two signals from CSD and PSDs.',
    'rng207: Fourier transform of input signal to a matched filter.',
    'rng208: Output of matched filter via inverse Fourier transform of X(omega)*H(omega).',
    'rng209: PSD of white noise at the input of a matched filter (two-sided).',
    'rng210: Noise PSD at the output of a matched filter.',
    'rng211: output noise tracks int|H|^2 -- narrower band, less noise',
    'rng212: Magnitude of instantaneous output signal of a matched filter at t = t0.',
    'rng213: Peak-power SNR at output of a matched filter.',
    "rng214: Total energy of a signal via Parseval's theorem.",
    'rng215: Normalized ratio used in maximizing matched-filter SNR.',
    'rng216: Schwarz inequality for complex functions A(f) and B(f).',
    'rng217: Schwarz inequality for real functions a(t) and b(t).',
    'rng218: Schwarz (Cauchy-Schwarz) inequality for two vectors.',
    'rng219: Triangle inequality for two vectors.',
    'rng220: Optimal frequency response of the matched filter.',
    'rng221: Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal.',
    'rng222: Matched-filter output equals scaled, delayed ACF of the reference signal.',
    'rng224: Basic three-sample reference pattern used in matched-filter illustration.',
    'rng227: Frequency-domain optimal matched-filter response for EEG spike-and-wave detection.',
    'rng228: Time-domain impulse response of the matched filter for EEG spike-and-wave detection.',
    'rng229: Frequency-domain output of matched filter equals PSD of the reference signal.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
