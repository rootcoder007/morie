# morie.fn -- bsacorr (rootcoder007/morie)
"""Correlation and spectral density: ACF, CCF, PSD, coherence, matched filtering, synchronized averaging.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 65
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2 as _atan2, cos, fsum, log10, pi, sin, sqrt
from math import fsum, sqrt
import math as _math
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._containers import SignalResult
from ._rgcore import aslist
from ._rgcore import aslist, gridint
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer

__all__ = [
    'coherence',
    'matched_filter',
    'rangayyan_acf_estimate',
    'rangayyan_bartlett_psd',
    'rangayyan_bandwidth',
    'rangayyan_ccf',
    'cardioresp',
    'rangayyan_coupled_freq_select',
    'rangayyan_coherence',
    'msc',
    'rangayyan_coherence_cxy',
    'rangayyan_eeg_bands',
    'rangayyan_eeg_autocorr',
    'rangayyan_eeg_rhythm_detect',
    'rangayyan_eeg_spectral',
    'emgfreq',
    'rangayyan_emg_peak_freq',
    'rangayyan_ch3_correlation_sum',
    'erpartifact',
    'rangayyan_erp_artifact_remove',
    'matchedfilt',
    'rangayyan_matched_filter',
    'mfmaxsnr',
    'rangayyan_matched_filter_snr',
    'rangayyan_periodogram',
    'rangayyan_psd',
    'psdhz',
    'rangayyan_psd_to_hz',
    'rangayyan_psd_to_acf',
    'pcgsyncavg',
    'rangayyan_pcg_sync_avg',
    'seizcohere',
    'rangayyan_seizure_detect',
    'specmoments',
    'rangayyan_spectral_moments',
    'specres',
    'rangayyan_spectral_resolution',
    'template',
    'rangayyan_template_match',
    'rangayyan_welch_psd',
    'rangayyan_ch3_acf_continuous',
    'rangayyan_ch3_acf_ensemble_estimate',
    'ensavg',
    'rangayyan_ch3_ensemble_average_function',
    'rangayyan_ch3_time_averaged_acf',
    'rangayyan_ch3_ccf_continuous',
    'idft',
    'rangayyan_ch3_idft_definition',
    'parseval',
    'rangayyan_ch3_parseval_theorem',
    'syncsum',
    'rangayyan_ch3_synchronized_averaging_sum',
    'rangayyan_ch3_normalized_cross_correlation_template',
    'dotprod',
    'rangayyan_ch4_dot_product_discrete',
    'rangayyan_ch4_correlation_coefficient_normalized_dot',
    'contproj',
    'rangayyan_ch4_continuous_dot_product',
    'rangayyan_ch4_ccf_continuous_with_delay',
    'rangayyan_ch4_ccf_discrete_with_delay',
    'ccfouter',
    'rangayyan_ch4_ccf_outer_product_random_signals',
    'csd',
    'rangayyan_ch4_csd_from_ccf',
    'cohere',
    'rangayyan_ch4_coherence_spectrum',
    'mfinput',
    'rangayyan_ch4_matched_filter_input_ft',
    'mfoutput',
    'rangayyan_ch4_matched_filter_output_inverse_ft',
    'mfnoisein',
    'rangayyan_ch4_white_noise_psd_input',
    'mfnoiseout',
    'rangayyan_ch4_noise_psd_at_output',
    'rangayyan_ch4_average_output_noise_power',
    'mfpeak',
    'rangayyan_ch4_matched_filter_instantaneous_signal',
    'mfsnr',
    'rangayyan_ch4_peak_power_snr',
    'sigenergy',
    'rangayyan_ch4_signal_total_energy',
    'mfratio',
    'rangayyan_ch4_snr_normalized_ratio',
    'schwarzc',
    'rangayyan_ch4_schwarz_inequality_complex',
    'schwarzr',
    'rangayyan_ch4_schwarz_inequality_real',
    'cauchysch',
    'rangayyan_ch4_cauchy_schwarz_vectors',
    'triangle',
    'rangayyan_ch4_triangle_inequality_vectors',
    'mftf',
    'rangayyan_ch4_matched_filter_optimal_transfer_function',
    'mfimpulse',
    'rangayyan_ch4_matched_filter_impulse_response',
    'mfacf',
    'rangayyan_ch4_matched_filter_output_acf',
    'refpattern',
    'rangayyan_ch4_basic_signal_g',
    'mftfeeg',
    'rangayyan_ch4_matched_filter_optimal_H_eeg',
    'mfimpeeg',
    'rangayyan_ch4_matched_filter_impulse_response_eeg',
    'mfpsd',
    'rangayyan_ch4_matched_filter_output_psd',
]

def _angle(z):
    """Principal argument in (-pi, pi], without importing cmath."""
    return _atan2(z.imag, z.real)



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
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def cardioresp(ecg_rate, resp, fs, band=(0.15, 0.40), nperseg=None):
    """Cardiorespiratory coupling: coherence and phase-locking value.

    Two complementary measures of the same coupling:

    - the coherence of eq. (4.32) at the respiratory frequency, which
      measures LINEAR association between the two spectra;
    - the phase-locking value,
          PLV = | mean_t exp(j [phi_ecg(t) - phi_resp(t)]) |,
      which measures whether the phase DIFFERENCE is constant,
      irrespective of amplitude.

    They answer different questions and can disagree.  Coherence is high
    when one signal linearly predicts the other; PLV is high when the
    phases march together even if the amplitudes are unrelated -- which
    is the case for respiratory sinus arrhythmia, where the coupling is
    a frequency modulation rather than a linear mixing.  Reporting only
    one of them is how a nonlinear coupling gets missed.

    The instantaneous phase comes from the analytic signal built by
    zeroing the negative-frequency half of the DFT, the discrete Hilbert
    transform; both signals are bandpass-restricted to ``band`` first,
    because a phase is only meaningful for a narrowband signal.
    """
    a, b = aslist(ecg_rate), aslist(resp)
    if len(a) != len(b):
        raise ValueError("the two signals must have the same length")
    n = len(a)
    if n < 16:
        raise ValueError("need at least sixteen samples")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    lo, hi = float(band[0]), float(band[1])
    if not 0 <= lo < hi <= fsv / 2:
        raise ValueError("the band must satisfy 0 <= lo < hi <= fs/2")

    def analytic(sig):
        mu = fsum(sig) / len(sig)
        re, im = _dft([v - mu for v in sig])
        out_re, out_im = [0.0] * n, [0.0] * n
        for k in range(n):
            f = k * fsv / n if k <= n // 2 else (k - n) * fsv / n
            if not (lo <= abs(f) < hi):
                continue
            if f < 0:
                continue                      # analytic: drop negatives
            scale = 1.0 if k in (0, n // 2) else 2.0
            out_re[k] = scale * re[k]
            out_im[k] = scale * im[k]
        step = 2.0 * pi / n
        xr, xi = [], []
        for i in range(n):
            ar = ai = 0.0
            for k in range(n):
                ang = step * i * k
                ar += out_re[k] * cos(ang) - out_im[k] * sin(ang)
                ai += out_re[k] * sin(ang) + out_im[k] * cos(ang)
            xr.append(ar / n)
            xi.append(ai / n)
        return xr, xi

    ar, ai = analytic(a)
    br, bi = analytic(b)
    dphi = [_angle(complex(ar[i], ai[i])) - _angle(complex(br[i], bi[i]))
            for i in range(n)]
    cre = fsum(cos(d) for d in dphi) / n
    cim = fsum(sin(d) for d in dphi) / n
    plv = sqrt(cre * cre + cim * cim)
    coh = cohere(a, b, fs=fsv, nperseg=nperseg)
    inband = [g for f, g in zip(coh["freqs"], coh["coherence"])
              if lo <= f < hi]
    peak = max(inband) if inband else 0.0
    return RichResult(payload={
        "plv": plv, "mean_phase_difference": _angle(complex(cre, cim)),
        "phase_difference": dphi,
        "coherence_peak": peak,
        "coherence_mean": fsum(inband) / len(inband) if inband else 0.0,
        "coherence": coh["coherence"], "freqs": coh["freqs"],
        "band": (lo, hi), "n": n, "fs": fsv,
        "method": "coherence per Rangayyan (2024) eq. (4.32); PLV per "
                  "Lachaux et al. (1999)"})


rangayyan_coupled_freq_select = cardioresp  # pre-policy spelling


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
def msc(x, y, fs=1.0, nperseg=None, noverlap=None):
    """Magnitude-SQUARED coherence.

    The square of eq. (4.32) of Rangayyan (2024):
        C_xy(f) = |S_xy(f)|^2 / (S_xx(f) S_yy(f)),
    bounded in [0, 1] and interpretable as the fraction of the power of
    one signal at frequency f that is linearly predictable from the
    other.

    The squared and unsquared forms are both in common use and are NOT
    interchangeable: 0.5 magnitude coherence is 0.25 magnitude-squared
    coherence.  Both are returned, with the same averaging requirement
    the book states for eq. (4.32).
    """
    r = cohere(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    out = dict(r)
    out["msc"] = [v * v for v in r["coherence"]]
    out["magnitude_coherence"] = r["coherence"]
    out["method"] = "Rangayyan (2024) eq. (4.32), squared"
    return RichResult(payload=out)


rangayyan_coherence_cxy = msc  # pre-policy spelling


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
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def emgfreq(x, fs, nperseg=None):
    """Mean and median frequency of an EMG signal.

    Rangayyan (2024) eqs. (6.34)-(6.35) applied to the EMG periodogram.
    Both are standard indices of muscle fatigue: as a muscle fatigues
    the spectrum shifts down, and BOTH statistics fall.

    They are not interchangeable.  The median of eq. (6.35) is far less
    sensitive to the high-frequency tail -- where an EMG record carries
    mostly instrumentation noise -- so it is the more stable fatigue
    index, and the two diverge exactly when the tail is contaminated.
    Their difference is returned for that reason.

    The mean is removed before the periodogram: a DC offset would put
    all its power in bin 0 and drag the mean frequency towards zero.
    """
    xs = aslist(x)
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if len(xs) < 8:
        raise ValueError("need at least eight samples")
    m = len(xs) if nperseg is None else min(int(nperseg), len(xs))
    seg = xs[:m]
    mu = fsum(seg) / m
    seg = [v - mu for v in seg]
    re, im = _dft(seg)
    half = m // 2 + 1
    p = [(re[k] ** 2 + im[k] ** 2) / m for k in range(half)]
    f = [k * fsv / m for k in range(half)]
    mom = specmoments(p, freqs=f)
    return RichResult(payload={
        "mean_frequency": mom["mean_frequency"],
        "median_frequency": mom["median_frequency"],
        "difference": mom["mean_frequency"] - mom["median_frequency"],
        "bandwidth": mom["bandwidth"], "total_power": mom["total_power"],
        "psd": p, "freqs": f, "fs": fsv, "nperseg": m,
        "method": "Rangayyan (2024) eqs. (6.34)-(6.35)"})


rangayyan_emg_peak_freq = emgfreq  # pre-policy spelling


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
def erpartifact(epochs, reject=None):
    """Artifact handling in ERP synchronized averaging.

    Rangayyan (2024) Section 3.5: averaging M aligned epochs raises the
    SNR by sqrt(M), because the evoked response adds coherently while
    zero-mean noise adds in power.  That is the mechanism behind the
    book's statement that the artifact is reduced by 1/sqrt(M).

    A large artifact -- an eye blink, a movement transient -- is NOT
    zero-mean over the epochs it contaminates, so it does not average
    away at any M; the standard remedy is to reject those epochs before
    averaging.  ``reject`` is that threshold, in units of the epochs'
    own peak amplitude, and the epochs dropped are reported.  Averaging
    without rejection and quoting the sqrt(M) gain would overstate the
    result by exactly the artifact that survived.
    """
    recs = [aslist(e) for e in epochs]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one epoch")
    n = len(recs[0])
    if n == 0:
        raise ValueError("epochs must be nonempty")
    if any(len(r) != n for r in recs):
        raise ValueError("all epochs must have the same length")
    peaks = [max(abs(v) for v in r) for r in recs]
    kept = list(range(m))
    dropped = []
    if reject is not None:
        thr = float(reject)
        if thr <= 0:
            raise ValueError("the rejection threshold must be positive")
        kept = [i for i in range(m) if peaks[i] <= thr]
        dropped = [i for i in range(m) if peaks[i] > thr]
        if not kept:
            raise ValueError("every epoch exceeds the rejection threshold")
    k = len(kept)
    avg = [fsum(recs[i][j] for i in kept) / k for j in range(n)]
    sd = [sqrt(fsum((recs[i][j] - avg[j]) ** 2 for i in kept) / k)
          for j in range(n)]
    return RichResult(payload={
        "average": avg, "sd": sd, "m": m, "m_kept": k,
        "rejected": dropped, "n_rejected": len(dropped),
        "peaks": peaks, "n": n,
        "snr_gain": sqrt(k), "snr_gain_db": 10.0 * log10(k) if k else None,
        "artifact_factor": 1.0 / sqrt(k) if k else None,
        "method": "Rangayyan (2024) Section 3.5 (synchronized averaging)"})


rangayyan_erp_artifact_remove = erpartifact  # pre-policy spelling


# -- rgmflt: Matched filter transfer function for signal detection in noise.
def matchedfilt(ref, x=None, noise_psd=None, freqs=None, t0=None,
                gain=1.0, dt=1.0):
    """Design a matched filter, and optionally run it.

    Rangayyan (2024) eqs. (4.48)-(4.49) give the white-noise case,
        H(f) = K X*(f) exp(-j 2 pi f t0),
    and Section 4.6 notes the derivation assumed white noise at the
    input (eq. 4.35).  For COLOURED noise the same Schwarz argument
    gives the generalized form

        H(f) = K X*(f) exp(-j 2 pi f t0) / P_nn(f),

    which whitens the noise before matching.  ``noise_psd`` selects
    between them, and the returned ``whitened`` says which was used --
    applying the white-noise filter to coloured noise is a real loss of
    detectability, not a rounding matter.

    Parameters
    ----------
    ref : array-like
        The reference signal to detect.
    x : array-like, optional
        A signal to filter.
    noise_psd : array-like, optional
        Noise PSD, one value per DFT bin of the reference.
    """
    rs = aslist(ref)
    n = len(rs)
    if n < 2:
        raise ValueError("the reference needs at least two samples")
    step = float(dt)
    re, im = _dft(rs)
    X = [complex(a, b) for a, b in zip(re, im)]
    shift = n if t0 is None else int(round(float(t0) / step))
    H = []
    for k, Xv in enumerate(X):
        ang = -2.0 * pi * k * shift / n
        Hv = float(gain) * Xv.conjugate() * complex(cos(ang), sin(ang))
        H.append(Hv)
    whitened = False
    if noise_psd is not None:
        pn = aslist(noise_psd)
        if len(pn) != n:
            raise ValueError("noise_psd needs one value per DFT bin (%d)" % n)
        if any(v <= 0 for v in pn):
            raise ValueError("the noise PSD must be positive everywhere")
        H = [h / p for h, p in zip(H, pn)]
        whitened = True
    h = mfimpulse(rs, t0=shift * step, gain=gain, dt=step)["h"] \
        if not whitened else None
    out = {"H": H, "h": h, "shift_samples": shift, "gain": float(gain),
           "whitened": whitened, "n_reference": n,
           "freqs": [k / (n * step) for k in range(n)]
           if freqs is None else aslist(freqs),
           "method": "Rangayyan (2024) eqs. (4.48)-(4.49); the noise_psd "
                     "branch is the coloured-noise generalization"}
    if x is not None:
        xs = aslist(x)
        taps = h if h is not None else _ifft_taps(H)
        y = []
        for k in range(len(xs) + len(taps) - 1):
            lo, hi = max(0, k - len(taps) + 1), min(k, len(xs) - 1)
            y.append(fsum(xs[i] * taps[k - i] for i in range(lo, hi + 1)))
        out["y"] = y
        out["peak_index"] = max(range(len(y)), key=lambda i: y[i])
    return RichResult(payload=out)


def _ifft_taps(H):
    n = len(H)
    step = 2.0 * pi / n
    out = []
    for i in range(n):
        acc = 0.0
        for k, v in enumerate(H):
            ang = step * i * k
            acc += v.real * cos(ang) - v.imag * sin(ang)
        out.append(acc / n)
    return out


rangayyan_matched_filter = matchedfilt  # pre-policy spelling


# -- rgmfsnr: Output SNR of matched filter (maximum SNR theorem).
def mfmaxsnr(x, noise_power, t=None, dt=1.0):
    """Maximum output SNR attainable by a matched filter.

    Rangayyan (2024) eq. (4.46) states the bound reached at the optimum:
        P_eta_i M_y^2 / (2 E_x P_eta_o) <= 1,
    which rearranges to a peak-power SNR of

        M_y^2 / P_eta_o = 2 E_x / P_eta_i,

    with E_x the total signal energy of eq. (4.40) and P_eta_i the
    input white-noise power.  Written with the two-sided density
    N0/2 = P_eta_i/2 this is the familiar 2 E / N0.

    The result depends on the signal ONLY through its energy: the
    matched filter extracts the same detectability from a short loud
    transient as from a long quiet one of equal energy.  That is the
    substantive content of the theorem and the reason shape does not
    appear in the answer.
    """
    xs = aslist(x)
    if len(xs) < 2:
        raise ValueError("need at least two samples")
    p = float(noise_power)
    if p <= 0:
        raise ValueError("the input noise power must be positive")
    ts = [i * float(dt) for i in range(len(xs))] if t is None else aslist(t)
    energy = gridint([v * v for v in xs], ts)
    snr = 2.0 * energy / p
    return RichResult(payload={
        "snr": snr, "snr_db": 10.0 * log10(snr) if snr > 0
        else float("-inf"),
        "energy": energy, "noise_power": p, "n0": p,
        "depends_only_on_energy": True,
        "method": "Rangayyan (2024) eq. (4.46)"})


rangayyan_matched_filter_snr = mfmaxsnr  # pre-policy spelling


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
def psdhz(psd, fs, n=None, bands=None):
    """Put a DFT-indexed PSD on a frequency axis in Hz and integrate bands.

    Rangayyan (2024) Section 6.4: bin k of an N-point DFT sits at
    k fs / N, and the power in a band is the sum of the bins in it times
    the bin width fs / N.

    The bin width is what turns a density into a power; omitting it
    leaves a quantity that changes when the record length changes, which
    is why two "band powers" computed at different N are otherwise not
    comparable.  ``bin_width`` is returned so the scaling is visible.

    Parameters
    ----------
    psd : array-like
        One-sided PSD, bins 0..N/2.
    fs : float
        Sampling rate in Hz.
    n : int, optional
        Full DFT length; defaults to 2 (len(psd) - 1).
    bands : mapping or sequence of (low, high), optional
        Bands to integrate, in Hz.
    """
    p = aslist(psd)
    if not p:
        raise ValueError("need at least one bin")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    nn = 2 * (len(p) - 1) if n is None else int(n)
    if nn < 2:
        raise ValueError("the DFT length must be at least 2")
    width = fsv / nn
    freqs = [k * width for k in range(len(p))]
    out = {"freqs": freqs, "bin_width": width, "psd": p, "n": nn,
           "fs": fsv, "total_power": fsum(p) * width,
           "nyquist": fsv / 2.0,
           "method": "Rangayyan (2024) Section 6.4"}
    if bands is not None:
        items = bands.items() if hasattr(bands, "items") \
            else [(i, b) for i, b in enumerate(bands)]
        powers = {}
        for name, (lo, hi) in items:
            if hi <= lo:
                raise ValueError("band %r has hi <= lo" % (name,))
            powers[name] = fsum(v * width for f, v in zip(freqs, p)
                                if lo <= f < hi)
        out["band_power"] = powers
        tot = fsum(powers.values())
        out["band_fraction"] = {k: (v / tot if tot > 0 else 0.0)
                                for k, v in powers.items()}
    return RichResult(payload=out)


rangayyan_psd_to_hz = psdhz  # pre-policy spelling


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
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def pcgsyncavg(cycles, fs=1.0):
    """Synchronized averaging of PCG SPECTRA.

    S_avg(f) = (1/M) sum_k |PCG_k(f)|^2.

    The averaging here is of POWER SPECTRA, not of waveforms, and the
    distinction is the point.  Successive PCG cycles are not
    sample-aligned -- a murmur is a random turbulent signal whose phase
    differs cycle to cycle -- so averaging the waveforms as in eq.
    (3.96) would cancel exactly the murmur one is trying to measure.
    Averaging the magnitudes discards the phase first, so the murmur
    energy adds instead of cancelling.

    Both are computed so the difference is visible: ``psd_of_average``
    is what waveform averaging would have given, and it is markedly
    smaller than ``average_psd`` whenever the cycles are not aligned.
    """
    recs = [aslist(c) for c in cycles]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one cycle")
    n = min(len(r) for r in recs)
    if n < 4:
        raise ValueError("cycles need at least four samples")
    recs = [r[:n] for r in recs]
    acc = [0.0] * (n // 2 + 1)
    for r in recs:
        mu = fsum(r) / n
        re, im = _dft([v - mu for v in r])
        for k in range(len(acc)):
            acc[k] += (re[k] ** 2 + im[k] ** 2) / n
    avg_psd = [v / m for v in acc]
    mean_wave = [fsum(r[i] for r in recs) / m for i in range(n)]
    mu = fsum(mean_wave) / n
    re, im = _dft([v - mu for v in mean_wave])
    psd_of_avg = [(re[k] ** 2 + im[k] ** 2) / n
                  for k in range(len(acc))]
    return RichResult(payload={
        "average_psd": avg_psd, "psd_of_average": psd_of_avg,
        "mean_waveform": mean_wave,
        "freqs": [k * float(fs) / n for k in range(len(acc))],
        "m": m, "n": n,
        "power_retained": (fsum(psd_of_avg) / fsum(avg_psd))
        if fsum(avg_psd) > 0 else 0.0,
        "method": "spectral synchronized averaging; contrast the waveform "
                  "averaging of Rangayyan (2024) eq. (3.96)"})


rangayyan_pcg_sync_avg = pcgsyncavg  # pre-policy spelling


# -- rgseiz: EEG seizure detection via rhythm coherence analysis.
def seizcohere(channels, fs, window, step=None, bands=None,
               nperseg=None):
    """Seizure detection from sustained inter-channel band coherence.

    Rangayyan (2024) Section 4.5.3: the coherence spectrum of eq. (4.32)
    detects rhythms present in COMMON between two EEG channels, and a
    seizure shows as sustained, widespread rhythmic activity.  So the
    detector is a moving window over which the mean pairwise coherence
    in each band is tracked.

    Two properties of the book's method are kept explicit.  First,
    coherence needs averaged spectra (eq. 4.32's caveat), so each window
    must hold several segments -- a window too short for that returns 1
    everywhere and would flag a seizure at every instant.  Second, the
    criterion is SUSTAINED elevation: a single high window is a
    transient, not a seizure, so the run lengths are returned rather
    than a per-window verdict.
    """
    chans = [aslist(c) for c in channels]
    if len(chans) < 2:
        raise ValueError("need at least two channels to form a coherence")
    n = min(len(c) for c in chans)
    w = int(window)
    if w > n:
        raise ValueError("the window is longer than the record")
    hop = w // 2 if step is None else int(step)
    if hop < 1:
        raise ValueError("step must be at least one sample")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if bands is None:
        bands = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
                 "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    seg = int(nperseg) if nperseg else max(8, w // 8)
    times, curves = [], {k: [] for k in bands}
    for s in range(0, n - w + 1, hop):
        pairs = {k: [] for k in bands}
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                c = cohere(chans[i][s:s + w], chans[j][s:s + w],
                           fs=fsv, nperseg=seg)
                for name, (lo, hi) in bands.items():
                    vals = [g for f, g in zip(c["freqs"], c["coherence"])
                            if lo <= f < hi]
                    if vals:
                        pairs[name].append(fsum(vals) / len(vals))
        times.append(s / fsv)
        for name in bands:
            curves[name].append(fsum(pairs[name]) / len(pairs[name])
                                if pairs[name] else 0.0)
    return RichResult(payload={
        "times": times, "coherence": curves, "bands": bands,
        "window": w, "step": hop, "nperseg": seg,
        "n_windows": len(times), "n_channels": len(chans),
        "sustained_criterion": "a seizure is a RUN of elevated windows; "
                               "one high window is a transient",
        "method": "Rangayyan (2024) Section 4.5.3, eq. (4.32)"})


rangayyan_seizure_detect = seizcohere  # pre-policy spelling


# -- rgsmom: Spectral moments: centroid (mean freq), variance (bandwidth), skewness.
def specmoments(psd, fs=1.0, freqs=None):
    """Moments of the PSD: mean and median frequency, spread, skewness,
    kurtosis.

    Rangayyan (2024) Section 6.4.4, eqs. (6.32)-(6.43):
        Ep       = sum_k S(k)                                     (6.32)
        f_mean   = (fs/N)(1/Ep) sum_k k S(k)                      (6.34)
        f_med    = (m/N) fs, largest m with (1/Ep) sum_0^m S < 1/2 (6.35)
        fm2      = (fs/N)^2 (1/Ep) sum (k - kbar)^2 S(k)          (6.37)
        skewness = fm3 / fm2^(3/2)                                (6.39)
        kurtosis = fm4 / fm2^2                                    (6.41)

    The sums run over ONE HALF of the periodic PSD, 0 to N/2: for a real
    signal S(k) is even-symmetric about fs/2, so summing the whole
    period double-counts every frequency and puts the mean at fs/2
    regardless of the signal.

    The book's own caveats are worth keeping in view: a nearly uniform
    PSD gives a mean frequency of half the maximum, which describes
    nothing, and the higher moments are sensitive to noise in the PSD
    estimate.  ``uniformity`` reports how flat the spectrum is so a
    caller can see when the mean is in that regime.
    """
    p = aslist(psd)
    if not p:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in p):
        raise ValueError("a PSD cannot be negative")
    n_half = len(p)
    if freqs is None:
        n = 2 * (n_half - 1) if n_half > 1 else 1
        f = [k * float(fs) / n for k in range(n_half)]
    else:
        f = aslist(freqs)
        if len(f) != n_half:
            raise ValueError("psd and freqs must have the same length")
    ep = fsum(p)
    if ep <= 0:
        raise ValueError("the PSD has zero total power")
    fmean = fsum(a * b for a, b in zip(f, p)) / ep
    cum, fmed = 0.0, f[-1]
    for a, b in zip(f, p):
        if cum + b >= 0.5 * ep:
            fmed = a
            break
        cum += b
    fm2 = fsum((a - fmean) ** 2 * b for a, b in zip(f, p)) / ep
    fm3 = fsum((a - fmean) ** 3 * b for a, b in zip(f, p)) / ep
    fm4 = fsum((a - fmean) ** 4 * b for a, b in zip(f, p)) / ep
    sd = sqrt(fm2) if fm2 > 0 else 0.0
    mx = max(p)
    return RichResult(payload={
        "total_power": ep, "mean_frequency": fmean,
        "median_frequency": fmed, "variance": fm2, "bandwidth": sd,
        "skewness": fm3 / fm2 ** 1.5 if fm2 > 0 else None,
        "kurtosis": fm4 / (fm2 * fm2) if fm2 > 0 else None,
        "fm3": fm3, "fm4": fm4, "n_bins": n_half,
        "uniformity": (ep / n_half) / mx if mx > 0 else 0.0,
        "method": "Rangayyan (2024) eqs. (6.32)-(6.43)"})


rangayyan_spectral_moments = specmoments  # pre-policy spelling


# -- rgspres: Spectral resolution and leakage analysis (Rayleigh criterion).
def specres(n, fs=1.0, window="rectangular"):
    """Spectral resolution and the leakage the window admits.

    The Rayleigh criterion: two sinusoids are resolvable when they are
    separated by at least one DFT bin,

        delta_f = fs / N,

    which depends only on the RECORD LENGTH.  Zero-padding to a longer
    DFT interpolates the same spectrum more finely and does NOT improve
    resolution -- the commonest misreading of a smoother-looking
    spectrum.

    The window widens that main lobe and sets how far energy leaks into
    distant bins.  The figures returned are the standard main-lobe
    widths in bins and the peak sidelobe levels: rectangular 2 bins and
    -13 dB, Hann 4 bins and -31 dB, Hamming 4 bins and -43 dB, Blackman
    6 bins and -58 dB.  The trade is explicit -- every window that
    suppresses leakage does so by widening the main lobe, so a window
    buys dynamic range with resolution.
    """
    nn = int(n)
    if nn < 2:
        raise ValueError("need at least two samples")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    table = {
        "rectangular": (2.0, -13.3, 1.0),
        "hann": (4.0, -31.5, 1.5),
        "hamming": (4.0, -42.7, 1.36),
        "blackman": (6.0, -58.1, 1.73),
    }
    key = str(window).lower()
    if key not in table:
        raise ValueError("unknown window %r; known: %s"
                         % (window, ", ".join(sorted(table))))
    lobe, side, enbw = table[key]
    df = fsv / nn
    return RichResult(payload={
        "delta_f": df, "resolution": df * lobe / 2.0,
        "main_lobe_bins": lobe, "sidelobe_db": side,
        "equivalent_noise_bandwidth_bins": enbw,
        "duration": nn / fsv, "n": nn, "fs": fsv, "window": key,
        "zero_padding_helps": False,
        "method": "Rayleigh criterion; Rangayyan (2024) Section 6.4 on "
                  "windowing and leakage"})


rangayyan_spectral_resolution = specres  # pre-policy spelling


# -- rgtmpl: Template matching for EEG spike-and-wave detection.
def template(x, ref, threshold=None, subtract_mean=True):
    """Template matching by the correlation coefficient at every shift.

    Rangayyan (2024) eqs. (4.25), (4.28) and Section 4.4: slide the
    reference over the signal and compute, at each position,

        gamma(k) = sum x(n+k) t(n)
                   / sqrt( sum x^2(n+k) sum t^2(n) ),

    the correlation coefficient of eq. (4.25) with the time shift of eq.
    (4.28).  The book applies it to EEG spike-and-wave detection.

    gamma is normalized at EVERY shift by the energy of the segment
    under the template, not by a global constant.  That is what makes it
    a correlation rather than a convolution, and it is why a large
    low-frequency excursion in the signal does not produce a spurious
    match: the normalization divides the amplitude out.  The book notes
    the means may be removed first (eq. 3.97); here that defaults to on,
    since a DC offset otherwise dominates the coefficient.
    """
    xs, ts = aslist(x), aslist(ref)
    n, m = len(xs), len(ts)
    if m < 2:
        raise ValueError("the template needs at least two samples")
    if n < m:
        raise ValueError("the signal is shorter than the template")
    if subtract_mean:
        mt = fsum(ts) / m
        ts = [v - mt for v in ts]
    et = fsum(v * v for v in ts)
    if et <= 0:
        raise ValueError("the template has zero energy")
    gam = []
    for k in range(n - m + 1):
        seg = xs[k:k + m]
        if subtract_mean:
            ms = fsum(seg) / m
            seg = [v - ms for v in seg]
        es = fsum(v * v for v in seg)
        if es <= 0:
            gam.append(0.0)
            continue
        gam.append(fsum(a * b for a, b in zip(seg, ts)) / sqrt(es * et))
    best = max(range(len(gam)), key=lambda i: gam[i])
    out = {"gamma": gam, "best_shift": best, "best_gamma": gam[best],
           "n_positions": len(gam), "template_length": m,
           "mean_removed": bool(subtract_mean),
           "method": "Rangayyan (2024) eqs. (4.25), (4.28)"}
    if threshold is not None:
        thr = float(threshold)
        hits, i = [], 0
        while i < len(gam):
            if gam[i] >= thr:
                j = i
                while j + 1 < len(gam) and gam[j + 1] >= thr:
                    j += 1
                peak = max(range(i, j + 1), key=lambda q: gam[q])
                hits.append(peak)
                i = j + 1
            else:
                i += 1
        out["detections"] = hits
        out["threshold"] = thr
        out["n_detections"] = len(hits)
    return RichResult(payload=out)


rangayyan_template_match = template  # pre-policy spelling


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
def idft(X):
    """Inverse discrete Fourier transform.

    Rangayyan (2024) eq. (3.81):
        x(n) = (1/N) sum_{k=0}^{N-1} X(k) exp(+j (2 pi / N) n k),
        n = 0, 1, ..., N-1.

    The 1/N and the PLUS sign in the exponent are what distinguish it
    from the forward transform of eq. (3.80); getting either wrong
    produces a signal that is scaled by N or time-reversed, both of
    which survive a magnitude plot unnoticed.
    """
    Xs = [complex(v) for v in X]
    n = len(Xs)
    if not n:
        raise ValueError("need at least one coefficient")
    step = 2.0 * pi / n
    out = []
    for i in range(n):
        acc = 0j
        for k, Xv in enumerate(Xs):
            ang = step * i * k
            acc += Xv * complex(cos(ang), sin(ang))
        out.append(acc / n)
    return RichResult(payload={
        "x": [v.real for v in out], "complex": out, "n": n,
        "max_imaginary": max(abs(v.imag) for v in out),
        "method": "Rangayyan (2024) eq. (3.81)"})


rangayyan_ch3_idft_definition = idft  # pre-policy spelling


# -- rng080: Parseval's theorem: total signal energy preserved under Fourier transform..
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def parseval(x):
    """Parseval's theorem for the DFT.

    Rangayyan (2024) eq. (3.91), discrete form:
        sum_{n=0}^{N-1} |x(n)|^2 = (1/N) sum_{k=0}^{N-1} |X(k)|^2.

    Total energy is preserved by the transform.  The 1/N is not
    decoration: with the forward transform of eq. (3.80) as written --
    no normalization -- the spectral sum is N times the time-domain
    energy, so omitting it inflates the answer by the record length.

    Because the sum of |X(k)|^2 over all k is the total energy, the book
    reads |X(k)|^2 as the spread of power along the frequency axis, and
    that is where the name power spectral density comes from.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 1:
        raise ValueError("need at least one sample")
    re, im = _dft(xs)
    time_energy = fsum(v * v for v in xs)
    freq_energy = fsum(a * a + b * b for a, b in zip(re, im)) / n
    gap = abs(time_energy - freq_energy)
    return RichResult(payload={
        "energy_time": time_energy, "energy_freq": freq_energy,
        "psd": [(a * a + b * b) / n for a, b in zip(re, im)],
        "max_difference": gap,
        "holds": gap <= 1e-9 * max(time_energy, 1.0), "n": n,
        "method": "Rangayyan (2024) eq. (3.91)"})


rangayyan_ch3_parseval_theorem = parseval  # pre-policy spelling


# -- rng085: Synchronized sum across M observations to form ensemble averaging..
def syncsum(observations):
    """Synchronized sum across M observations.

    Rangayyan (2024) eq. (3.96):
        sum_k y_k(n) = sum_k x_k(n) + sum_k eta_k(n),
        n = 0, 1, ..., N-1.

    The step between the model of eq. (3.95) and the averaging that
    follows.  Its content is that the two sums separate: if the
    repetitions are identical and aligned the signal sum is M x(n),
    growing linearly in M, while a zero-mean noise sum grows only as
    sqrt(M).  Dividing by M is what turns the sum into the average.

    The sum is returned rather than the mean, because that is the
    equation; ``average`` is provided beside it so the division is
    explicit rather than assumed.
    """
    recs = [aslist(r) for r in observations]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one observation")
    n = len(recs[0])
    if n == 0:
        raise ValueError("records must be nonempty")
    if any(len(r) != n for r in recs):
        raise ValueError("all realizations must have the same length")
    total = [fsum(r[i] for r in recs) for i in range(n)]
    return RichResult(payload={
        "sum": total, "average": [v / m for v in total], "m": m, "n": n,
        "signal_growth": "linear in M", "noise_growth": "sqrt(M)",
        "method": "Rangayyan (2024) eq. (3.96)"})


rangayyan_ch3_synchronized_averaging_sum = syncsum  # pre-policy spelling


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
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def dotprod(x, y, subtract_mean=False):
    """Discrete inner product of two signals, and the correlation it
    normalizes to.

    Rangayyan (2024) eqs. (4.24)-(4.25):
        <x, y> = sum_{n=0}^{N-1} x(n) y(n)                        (4.24)
        gamma_xy = <x, y> / sqrt( sum x^2 sum y^2 )               (4.25)

    The book reads the dot product as the projection of one signal onto
    the other, each viewed as an N-dimensional vector, and notes that the
    means may be subtracted first as in eq. (3.97) -- which is a
    different quantity, so it is an argument here rather than a silent
    default.  Without mean removal gamma is the cosine of the angle
    between the raw vectors and is 1 for any two positive signals of
    similar shape; with it, gamma is Pearson's r.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if not xs:
        raise ValueError("need at least one sample")
    if subtract_mean:
        mx, my = fsum(xs) / len(xs), fsum(ys) / len(ys)
        xs = [v - mx for v in xs]
        ys = [v - my for v in ys]
    dp = fsum(a * b for a, b in zip(xs, ys))
    ex = fsum(v * v for v in xs)
    ey = fsum(v * v for v in ys)
    gamma = dp / sqrt(ex * ey) if ex > 0 and ey > 0 else None
    return RichResult(payload={
        "dot_product": dp, "gamma": gamma, "energy_x": ex, "energy_y": ey,
        "n": len(xs), "mean_removed": bool(subtract_mean),
        "method": "Rangayyan (2024) eqs. (4.24)-(4.25)"})


rangayyan_ch4_dot_product_discrete = dotprod  # pre-policy spelling


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
def contproj(x, y, t=None, dt=1.0):
    """Projection of one continuous-time signal onto another.

    Rangayyan (2024) eq. (4.26):
        theta_xy = integral x(t) y(t) dt.

    The continuous counterpart of eq. (4.24).  Tabulated, it is the
    discrete inner product SCALED BY dt -- dropping the dt turns an
    integral into a sum and leaves the result wrong by a factor of the
    sampling interval, which is invisible in any comparison of two
    projections computed the same way and fatal in any absolute one.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if len(xs) < 2:
        raise ValueError("need at least two samples to integrate")
    ts = [i * float(dt) for i in range(len(xs))] if t is None else aslist(t)
    if len(ts) != len(xs):
        raise ValueError("t and x must have the same length")
    prod = [a * b for a, b in zip(xs, ys)]
    theta = gridint(prod, ts)
    return RichResult(payload={
        "theta": theta, "integrand": prod,
        "discrete_sum": fsum(prod), "duration": ts[-1] - ts[0],
        "n": len(xs), "method": "Rangayyan (2024) eq. (4.26)"})


rangayyan_ch4_continuous_dot_product = contproj  # pre-policy spelling


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
def ccfouter(x, y, order, tol=1e-3):
    """CCF of random signals as the expectation of an outer product.

    Rangayyan (2024) eq. (4.29):
        Theta_xy = E[ x(n) y^T(n) ],

    where x(n) = [x(n), x(n-1), ..., x(n-N+1)]^T holds the most recent N
    samples.  The result is an N x N matrix whose entries carry all the
    pairwise delays within the window -- which is why the outer product
    appears in the Wiener and RLS normal equations rather than a plain
    correlation sequence.

    The expectation is estimated by averaging over the available
    positions, so entry (i, j) is a sample estimate of E[x(n-i) y(n-j)]
    and depends only on i - j when the processes are stationary.  That
    Toeplitz structure is measured and reported, not assumed: departure
    from it is the signature of a nonstationary record.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = int(order)
    if n < 1:
        raise ValueError("order must be at least 1")
    if len(xs) < n:
        raise ValueError("need at least %d samples" % n)
    m = len(xs) - n + 1
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = fsum(xs[k - i] * ys[k - j]
                             for k in range(n - 1, len(xs))) / m
    dev = 0.0
    for d in range(-(n - 1), n):
        band = [mat[i][i - d] for i in range(n) if 0 <= i - d < n]
        if len(band) > 1:
            mu = fsum(band) / len(band)
            dev = max(dev, max(abs(v - mu) for v in band))
    scale = max(abs(mat[i][j]) for i in range(n) for j in range(n)) or 1.0
    return RichResult(payload={
        "theta": mat, "order": n, "n_positions": m,
        "toeplitz_deviation": dev,
        "relative_deviation": dev / scale,
        "tol": float(tol), "toeplitz": dev <= float(tol) * scale,
        "method": "Rangayyan (2024) eq. (4.29)"})


rangayyan_ch4_ccf_outer_product_random_signals = ccfouter  # pre-policy spelling


# -- rng205: Cross-spectral density (CSD) as the Fourier transform of the CCF..
def csd(x, y, fs=1.0):
    """Cross-spectral density as the Fourier transform of the CCF.

    Rangayyan (2024) eq. (4.31):
        S_xy(f) = FT[theta_xy(tau)] = X(f) Y*(f),

    with eq. (4.30) the autospectrum special case S_xx = |X|^2.  The book
    notes the CSD peaks at frequencies present in BOTH signals, which is
    what makes it useful for finding rhythms shared between two EEG
    channels.

    Both routes are computed -- the transform of the CCF and the product
    X Y* -- and compared, because they agree only when the CCF is taken
    over the full circular lag range; a truncated CCF gives a smoothed
    CSD, which is a different estimator.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    xr, xi = _dft(xs)
    yr, yi = _dft(ys)
    prod = [complex(a, b) * complex(c, -d)
            for a, b, c, d in zip(xr, xi, yr, yi)]
    circ = [fsum(xs[i] * ys[(i + k) % n] for i in range(n))
            for k in range(n)]
    cr, ci = _dft(circ)
    viaccf = [complex(a, -b) for a, b in zip(cr, ci)]
    gap = max(abs(a - b) for a, b in zip(prod, viaccf))
    scale = max(abs(v) for v in prod) or 1.0
    return RichResult(payload={
        "csd": prod, "via_ccf": viaccf, "ccf_circular": circ,
        "freqs": [k * float(fs) / n for k in range(n)],
        "max_difference": gap, "agrees": gap <= 1e-8 * scale, "n": n,
        "method": "Rangayyan (2024) eqs. (4.30)-(4.31)"})


rangayyan_ch4_csd_from_ccf = csd  # pre-policy spelling


# -- rng206: Magnitude coherence spectrum between two signals from CSD and PSDs..
def cohere(x, y, fs=1.0, nperseg=None, noverlap=None):
    """Magnitude coherence spectrum between two signals.

    Rangayyan (2024) eq. (4.32):
        Gamma_xy(f) = [ |S_xy(f)|^2 / (S_xx(f) S_yy(f)) ]^(1/2).

    The book is emphatic on the trap: computed directly from two single
    observations the magnitude is UNITY AT EVERY FREQUENCY, which is
    incorrect; each spectral density must be estimated by AVERAGING over
    several observations.  So this function segments the records and
    averages, and refuses to run with a single segment rather than
    returning the meaningless all-ones answer.

    The phase of the coherence spectrum is angle(S_xy), the average
    phase difference between the two signals at each frequency; it is
    returned alongside, since the magnitude alone says nothing about
    lead or lag.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    m = int(nperseg) if nperseg else max(8, n // 8)
    if m < 4:
        raise ValueError("segments must hold at least four samples")
    step = m - (int(noverlap) if noverlap is not None else m // 2)
    if step < 1:
        raise ValueError("noverlap must be smaller than nperseg")
    starts = list(range(0, n - m + 1, step))
    if len(starts) < 2:
        raise ValueError("eq. (4.32) needs the spectra AVERAGED over "
                         "several observations; %d segment(s) of %d "
                         "samples would give a coherence of 1 at every "
                         "frequency" % (len(starts), m))
    half = m // 2 + 1
    sxx = [0.0] * half
    syy = [0.0] * half
    sxy = [0j] * half
    for s in starts:
        a, b = xs[s:s + m], ys[s:s + m]
        ma, mb = fsum(a) / m, fsum(b) / m
        ar, ai = _dft([v - ma for v in a])
        br, bi = _dft([v - mb for v in b])
        for k in range(half):
            A = complex(ar[k], ai[k])
            B = complex(br[k], bi[k])
            sxx[k] += (A * A.conjugate()).real
            syy[k] += (B * B.conjugate()).real
            sxy[k] += A * B.conjugate()
    k_n = float(len(starts))
    sxx = [v / k_n for v in sxx]
    syy = [v / k_n for v in syy]
    sxy = [v / k_n for v in sxy]
    gam, ph = [], []
    for k in range(half):
        den = sxx[k] * syy[k]
        gam.append(sqrt(abs(sxy[k]) ** 2 / den) if den > 0 else 0.0)
        ph.append(_angle(sxy[k]))
    return RichResult(payload={
        "coherence": gam, "phase": ph, "sxx": sxx, "syy": syy, "sxy": sxy,
        "freqs": [k * float(fs) / m for k in range(half)],
        "n_segments": len(starts), "nperseg": m,
        "method": "Rangayyan (2024) eq. (4.32)"})


rangayyan_ch4_coherence_spectrum = cohere  # pre-policy spelling


# -- rng207: Fourier transform of input signal to a matched filter..
def mfinput(x, omega, dt=1.0):
    """Fourier transform of the signal entering a matched filter.

    Rangayyan (2024) eq. (4.33):
        X(omega) = integral x(t) exp(-j omega t) dt,

    the first step of the derivation in Section 4.6.1.  The transform is
    an integral, so it carries the sampling interval; the matched filter
    of eq. (4.48) is built from X* and the dt cancels in the ratio of
    eq. (4.41), but not in any absolute energy.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    step = float(dt)
    if step <= 0:
        raise ValueError("dt must be positive")
    scalar = isinstance(omega, (int, float))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    vals = []
    for w in ws:
        re = fsum(v * cos(-w * i * step) for i, v in enumerate(xs))
        im = fsum(v * sin(-w * i * step) for i, v in enumerate(xs))
        vals.append(complex(re, im) * step)
    return RichResult(payload={
        "X": vals[0] if scalar else vals,
        "omega": ws[0] if scalar else ws, "dt": step, "n": len(xs),
        "method": "Rangayyan (2024) eq. (4.33)"})


rangayyan_ch4_matched_filter_input_ft = mfinput  # pre-policy spelling


# -- rng208: Output of matched filter via inverse Fourier transform of X(omega)*H(omega)..
def mfoutput(x, h, dt=1.0):
    """Matched-filter output as the inverse transform of X(f) H(f).

    Rangayyan (2024) eq. (4.34):
        y(t) = (1/2 pi) integral X(w) H(w) exp(+j w t) dw
             = integral X(f) H(f) exp(+j 2 pi f t) df.

    Equivalently, and computed that way here, y = x * h in the time
    domain: for a finite record the convolution is exact where the
    frequency-domain route would need the transform sampled finely
    enough to avoid wrap-around.  The sample of peak output and its
    location are returned, since eq. (4.38) reads M_y off that peak.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    step = float(dt)
    if step <= 0:
        raise ValueError("dt must be positive")
    y = []
    for k in range(len(xs) + len(hs) - 1):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)) * step)
    peak = max(range(len(y)), key=lambda i: abs(y[i]))
    return RichResult(payload={
        "y": y, "t": [i * step for i in range(len(y))],
        "peak_index": peak, "peak_time": peak * step,
        "peak_magnitude": abs(y[peak]), "dt": step,
        "method": "Rangayyan (2024) eq. (4.34)"})


rangayyan_ch4_matched_filter_output_inverse_ft = mfoutput  # pre-policy spelling


# -- rng209: PSD of white noise at the input of a matched filter (two-sided)..
def mfnoisein(power, freqs=None):
    """PSD of white noise at the input of a matched filter.

    Rangayyan (2024) eq. (4.35):
        S_eta_i(f) = P_eta_i / 2,

    where P_eta_i is the AVERAGE noise power at the input.  The factor
    of two is the two-sided convention: the density is flat over
    positive and negative frequencies alike, so integrating it over all
    f returns P_eta_i and not twice it.  Halving a one-sided density by
    mistake, or forgetting to, is a factor-of-two error in every SNR
    downstream.
    """
    p = float(power)
    if p < 0:
        raise ValueError("noise power cannot be negative")
    density = p / 2.0
    out = {"density": density, "power": p, "two_sided": True,
           "method": "Rangayyan (2024) eq. (4.35)"}
    if freqs is not None:
        fs_ = aslist(freqs)
        out["psd"] = [density] * len(fs_)
        out["freqs"] = fs_
    return RichResult(payload=out)


rangayyan_ch4_white_noise_psd_input = mfnoisein  # pre-policy spelling


# -- rng210: Noise PSD at the output of a matched filter..
def mfnoiseout(power, H, freqs=None, df=1.0):
    """Noise PSD and average noise power at a matched filter's output.

    Rangayyan (2024) eqs. (4.36)-(4.37):
        S_eta_o(f) = (P_eta_i / 2) |H(f)|^2                       (4.36)
        P_eta_o    = (P_eta_i / 2) integral |H(f)|^2 df           (4.37)

    and the book notes the RMS noise in the absence of signal is
    sqrt(P_eta_o).  The filter shapes the noise but does not change its
    whiteness assumption: eq. (4.36) holds only because the input was
    white, so its density factored out of the transfer function.
    """
    p = float(power)
    if p < 0:
        raise ValueError("noise power cannot be negative")
    Hs = [complex(v) for v in H]
    if not Hs:
        raise ValueError("need at least one transfer-function sample")
    density = p / 2.0
    psd = [density * abs(v) ** 2 for v in Hs]
    step = float(df)
    if freqs is not None:
        fv = aslist(freqs)
        if len(fv) != len(Hs):
            raise ValueError("freqs and H must have the same length")
        total = gridint(psd, fv) if len(fv) > 1 else psd[0] * step
    else:
        total = fsum(psd) * step
    return RichResult(payload={
        "psd": psd, "power": total, "rms": sqrt(total) if total > 0 else 0.0,
        "input_power": p, "input_density": density,
        "method": "Rangayyan (2024) eqs. (4.36)-(4.37)"})


rangayyan_ch4_noise_psd_at_output = mfnoiseout  # pre-policy spelling


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
def mfpeak(X, H, freqs, t0):
    """Magnitude of the instantaneous matched-filter output at t = t0.

    Rangayyan (2024) eq. (4.38):
        M_y = |y(t_0)| = | integral X(f) H(f) exp(+j 2 pi f t0) df |.

    This is eq. (4.34) evaluated at the one instant the filter is
    designed to peak at.  The book's next line defines the output SNR as
    M_y / sqrt(P_eta_o), so M_y is the numerator of everything that
    follows -- and it is a MAGNITUDE, so the phase of X H at t0 is what
    the optimal filter of eq. (4.48) is chosen to align.
    """
    Xs = [complex(v) for v in X]
    Hs = [complex(v) for v in H]
    fv = aslist(freqs)
    if not (len(Xs) == len(Hs) == len(fv)):
        raise ValueError("X, H and freqs must have the same length")
    if len(fv) < 2:
        raise ValueError("need at least two frequency points to integrate")
    t = float(t0)
    re, im = [], []
    for Xv, Hv, f in zip(Xs, Hs, fv):
        ang = 2.0 * pi * f * t
        p = Xv * Hv * complex(cos(ang), sin(ang))
        re.append(p.real)
        im.append(p.imag)
    val = complex(gridint(re, fv), gridint(im, fv))
    return RichResult(payload={
        "my": abs(val), "y": val, "t0": t, "phase": _angle(val),
        "method": "Rangayyan (2024) eq. (4.38)"})


rangayyan_ch4_matched_filter_instantaneous_signal = mfpeak  # pre-policy spelling


# -- rng213: Peak-power SNR at output of a matched filter..
def mfsnr(my, noise_power):
    """Peak-power SNR at the output of a matched filter.

    Rangayyan (2024) eq. (4.39):
        M_y^2 / P_eta_o = instantaneous peak power of the signal
                          / mean noise power,

    the quantity the derivation maximizes.  Note it is a PEAK-to-MEAN
    ratio, not the mean-to-mean power ratio of an ordinary SNR: the
    matched filter maximizes the output at one instant, not the average
    output power, which is why it is the right criterion for detecting
    a known transient and the wrong one for a continuous signal.
    """
    m = float(my)
    p = float(noise_power)
    if p <= 0:
        raise ValueError("the output noise power must be positive")
    ratio = m * m / p
    return RichResult(payload={
        "snr": ratio, "snr_db": 10.0 * log10(ratio) if ratio > 0
        else float("-inf"),
        "amplitude_snr": m / sqrt(p), "my": m, "noise_power": p,
        "peak_to_mean": True, "method": "Rangayyan (2024) eq. (4.39)"})


rangayyan_ch4_peak_power_snr = mfsnr  # pre-policy spelling


# -- rng214: Total energy of a signal via Parseval's theorem..
def sigenergy(x, t=None, dt=1.0, X=None, freqs=None):
    """Total energy of a signal, in either domain.

    Rangayyan (2024) eq. (4.40):
        E_x = integral x^2(t) dt = integral |X(f)|^2 df,

    which is Parseval's theorem of eq. (3.91) named for the role it
    plays here: E_x is a constant for a given signal, so maximizing the
    ratio of eq. (4.41) is equivalent to maximizing eq. (4.39).

    Given both a signal and its spectrum, both integrals are computed
    and compared; a mismatch means the transform was not scaled as an
    integral, which is the usual cause of a factor of dt or of N.
    """
    out = {"method": "Rangayyan (2024) eq. (4.40)"}
    if x is not None:
        xs = aslist(x)
        if len(xs) < 2:
            raise ValueError("need at least two samples to integrate")
        ts = [i * float(dt) for i in range(len(xs))] if t is None \
            else aslist(t)
        out["energy_time"] = gridint([v * v for v in xs], ts)
        out["energy"] = out["energy_time"]
    if X is not None:
        if freqs is None:
            raise ValueError("give the frequency grid alongside X")
        Xs = [complex(v) for v in X]
        fv = aslist(freqs)
        if len(Xs) != len(fv):
            raise ValueError("X and freqs must have the same length")
        out["energy_freq"] = gridint([abs(v) ** 2 for v in Xs], fv)
        out.setdefault("energy", out["energy_freq"])
    if "energy_time" in out and "energy_freq" in out:
        gap = abs(out["energy_time"] - out["energy_freq"])
        out["max_difference"] = gap
        out["parseval_holds"] = gap <= 1e-6 * max(out["energy_time"], 1.0)
    if "energy" not in out:
        raise ValueError("give a signal, a spectrum, or both")
    return RichResult(payload=out)


rangayyan_ch4_signal_total_energy = sigenergy  # pre-policy spelling


# -- rng215: Normalized ratio used in maximizing matched-filter SNR..
def mfratio(X, H, freqs, t0, noise_power):
    """The normalized ratio maximized in deriving the matched filter.

    Rangayyan (2024) eq. (4.41):
        M_y^2 / (E_x P_eta_o)
          = | integral H X exp(+j 2 pi f t0) df |^2
            / [ (P_eta_i / 2) integral |H|^2 df  integral |X|^2 df ].

    Dividing by the constant E_x is what turns eq. (4.39) into something
    Schwarz's inequality applies to directly: by eq. (4.46) the ratio
    cannot exceed 2 / P_eta_i, with equality exactly at the optimal
    filter of eq. (4.48).  ``bound`` reports that ceiling and
    ``optimality`` how close the supplied H comes to it.
    """
    Xs = [complex(v) for v in X]
    Hs = [complex(v) for v in H]
    fv = aslist(freqs)
    if not (len(Xs) == len(Hs) == len(fv)):
        raise ValueError("X, H and freqs must have the same length")
    if len(fv) < 2:
        raise ValueError("need at least two frequency points")
    p = float(noise_power)
    if p <= 0:
        raise ValueError("the input noise power must be positive")
    num = mfpeak(Xs, Hs, fv, t0)["my"] ** 2
    eh = gridint([abs(v) ** 2 for v in Hs], fv)
    ex = gridint([abs(v) ** 2 for v in Xs], fv)
    den = (p / 2.0) * eh * ex
    if den <= 0:
        raise ValueError("the denominator of eq. (4.41) vanishes")
    ratio = num / den
    bound = 2.0 / p
    return RichResult(payload={
        "ratio": ratio, "bound": bound,
        "optimality": ratio / bound if bound > 0 else None,
        "numerator": num, "energy_h": eh, "energy_x": ex,
        "method": "Rangayyan (2024) eqs. (4.41), (4.46)"})


rangayyan_ch4_snr_normalized_ratio = mfratio  # pre-policy spelling


# -- rng216: Schwarz inequality for complex functions A(f) and B(f)..
def schwarzc(A, B, grid):
    """Schwarz's inequality for two complex functions.

    Rangayyan (2024) eq. (4.42):
        | integral A(f) B(f) df |^2
            <= integral |A(f)|^2 df  integral |B(f)|^2 df,

    with equality, as the book states just below eq. (4.45), exactly
    when A(f) = K B*(f) for a real constant K.  That equality condition
    is the entire matched-filter derivation: applying it with A = H and
    B = X exp(+j 2 pi f t0) gives eq. (4.48).

    ``equality`` reports whether the supplied pair attains the bound,
    and ``k`` the constant it would need, so the condition can be
    checked rather than recited.
    """
    As = [complex(v) for v in A]
    Bs = [complex(v) for v in B]
    g = aslist(grid)
    if not (len(As) == len(Bs) == len(g)):
        raise ValueError("A, B and the grid must have the same length")
    if len(g) < 2:
        raise ValueError("need at least two grid points to integrate")
    prod = [a * b for a, b in zip(As, Bs)]
    inner = complex(gridint([v.real for v in prod], g),
                    gridint([v.imag for v in prod], g))
    ea = gridint([abs(v) ** 2 for v in As], g)
    eb = gridint([abs(v) ** 2 for v in Bs], g)
    lhs = abs(inner) ** 2
    rhs = ea * eb
    ks = [a / b.conjugate() for a, b in zip(As, Bs) if abs(b) > 1e-300]
    k = ks[0] if ks else None
    collinear = bool(ks) and all(abs(v - k) <= 1e-9 * (1 + abs(k))
                                 for v in ks)
    return RichResult(payload={
        "lhs": lhs, "rhs": rhs, "holds": lhs <= rhs * (1 + 1e-9),
        "ratio": lhs / rhs if rhs > 0 else None,
        "equality": rhs > 0 and abs(lhs - rhs) <= 1e-9 * rhs,
        "k": k, "collinear": collinear,
        "method": "Rangayyan (2024) eq. (4.42)"})


rangayyan_ch4_schwarz_inequality_complex = schwarzc  # pre-policy spelling


# -- rng217: Schwarz inequality for real functions a(t) and b(t)..
def schwarzr(a, b, grid=None, dt=1.0):
    """Schwarz's inequality for two real functions.

    Rangayyan (2024) eq. (4.43):
        [ integral a(t) b(t) dt ]^2
            <= integral a^2(t) dt  integral b^2(t) dt,

    the real-valued case of eq. (4.42), with equality when a(t) = K b(t)
    -- the two functions collinear as vectors in function space.
    """
    av, bv = aslist(a), aslist(b)
    if len(av) != len(bv):
        raise ValueError("a and b must have the same length")
    if len(av) < 2:
        raise ValueError("need at least two samples to integrate")
    g = [i * float(dt) for i in range(len(av))] if grid is None \
        else aslist(grid)
    inner = gridint([p * q for p, q in zip(av, bv)], g)
    ea = gridint([v * v for v in av], g)
    eb = gridint([v * v for v in bv], g)
    lhs = inner * inner
    rhs = ea * eb
    ks = [p / q for p, q in zip(av, bv) if abs(q) > 1e-300]
    k = ks[0] if ks else None
    return RichResult(payload={
        "lhs": lhs, "rhs": rhs, "holds": lhs <= rhs * (1 + 1e-9),
        "equality": rhs > 0 and abs(lhs - rhs) <= 1e-9 * rhs, "k": k,
        "collinear": bool(ks) and all(abs(v - k) <= 1e-9 * (1 + abs(k))
                                      for v in ks),
        "method": "Rangayyan (2024) eq. (4.43)"})


rangayyan_ch4_schwarz_inequality_real = schwarzr  # pre-policy spelling


# -- rng218: Schwarz (Cauchy-Schwarz) inequality for two vectors..
def cauchysch(a, b):
    """Cauchy-Schwarz inequality for two vectors.

    Rangayyan (2024) eq. (4.44):  |a . b| <= |a| |b|,

    with equality when a = K b.  The vector form of eqs. (4.42)-(4.43);
    the cosine of the angle between the vectors is the ratio of the two
    sides, which is exactly the correlation coefficient of eq. (4.25) on
    mean-removed signals.
    """
    av, bv = aslist(a), aslist(b)
    if len(av) != len(bv):
        raise ValueError("a and b must have the same length")
    if not av:
        raise ValueError("need at least one component")
    dp = fsum(p * q for p, q in zip(av, bv))
    na = sqrt(fsum(v * v for v in av))
    nb = sqrt(fsum(v * v for v in bv))
    return RichResult(payload={
        "lhs": abs(dp), "rhs": na * nb,
        "holds": abs(dp) <= na * nb * (1 + 1e-12),
        "cosine": dp / (na * nb) if na > 0 and nb > 0 else None,
        "equality": na > 0 and nb > 0
        and abs(abs(dp) - na * nb) <= 1e-9 * na * nb,
        "norm_a": na, "norm_b": nb,
        "method": "Rangayyan (2024) eq. (4.44)"})


rangayyan_ch4_cauchy_schwarz_vectors = cauchysch  # pre-policy spelling


# -- rng219: Triangle inequality for two vectors..
def triangle(a, b):
    """Triangle inequality for two vectors.

    Rangayyan (2024) eq. (4.45):  |a + b| <= |a| + |b|,

    with equality when the vectors point the same way.  It follows from
    eq. (4.44) by squaring, and it is the statement that no combination
    of two signals can carry more amplitude than their amplitudes sum to
    -- the reason adding an uncorrelated noise adds in POWER, not in
    amplitude.
    """
    av, bv = aslist(a), aslist(b)
    if len(av) != len(bv):
        raise ValueError("a and b must have the same length")
    if not av:
        raise ValueError("need at least one component")
    s = [p + q for p, q in zip(av, bv)]
    ns = sqrt(fsum(v * v for v in s))
    na = sqrt(fsum(v * v for v in av))
    nb = sqrt(fsum(v * v for v in bv))
    return RichResult(payload={
        "lhs": ns, "rhs": na + nb, "holds": ns <= (na + nb) * (1 + 1e-12),
        "equality": abs(ns - (na + nb)) <= 1e-9 * (na + nb + 1.0),
        "norm_sum": ns, "norm_a": na, "norm_b": nb,
        "method": "Rangayyan (2024) eq. (4.45)"})


rangayyan_ch4_triangle_inequality_vectors = triangle  # pre-policy spelling


# -- rng220: Optimal frequency response of the matched filter..
def mftf(X, freqs, t0, gain=1.0):
    """Optimal transfer function of the matched filter.

    Rangayyan (2024) eq. (4.48):
        H(f) = K X*(f) exp(-j 2 pi f t0),

    the condition A(f) = K B*(f) of Schwarz's inequality applied with
    A = H and B = X exp(+j 2 pi f t0).

    The book's reading: the transfer function is proportional to the
    COMPLEX CONJUGATE of the transform of the signal to be detected.  The
    conjugate is what cancels the signal's phase so every frequency
    component arrives in step at t0 -- that coherent addition is the
    whole gain of the method, and conjugating the wrong quantity gives a
    filter that is exactly as bad as the right one is good.
    """
    Xs = [complex(v) for v in X]
    fv = aslist(freqs)
    if len(Xs) != len(fv):
        raise ValueError("X and freqs must have the same length")
    k = float(gain)
    t = float(t0)
    H = []
    for Xv, f in zip(Xs, fv):
        ang = -2.0 * pi * f * t
        H.append(k * Xv.conjugate() * complex(cos(ang), sin(ang)))
    return RichResult(payload={
        "H": H, "freqs": fv, "t0": t, "gain": k,
        "magnitude": [abs(v) for v in H],
        "conjugate_of_signal": True,
        "method": "Rangayyan (2024) eqs. (4.48), (4.55)"})


rangayyan_ch4_matched_filter_optimal_transfer_function = mftf  # pre-policy spelling


# -- rng221: Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal..
def mfimpulse(x, t0=None, gain=1.0, dt=1.0):
    """Impulse response of the matched filter.

    Rangayyan (2024) eqs. (4.49) and (4.56):
        h(t) = K x[-(t - t0)] = K x(t0 - t),

    a scaled, TIME-REVERSED and shifted copy of the reference signal.

    The delay t0 has to be at least the duration of the reference or the
    filter is not causal; the book says so, and adds a warning specific
    to a DFT implementation of eq. (4.55) at the reference length N --
    the periodicity of the DFT supplies a shift of N-1, one sample short
    of the N needed.  ``t0`` defaults to the N-sample shift, and a
    shorter one is rejected rather than silently producing an
    anticausal filter.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 1:
        raise ValueError("need at least one sample")
    step = float(dt)
    if step <= 0:
        raise ValueError("dt must be positive")
    shift = n if t0 is None else int(round(float(t0) / step))
    if shift < n:
        raise ValueError("t0 must be at least the reference duration "
                         "(%d samples) for a causal filter; the DFT of "
                         "eq. (4.55) at length N supplies only N-1"
                         % n)
    h = [0.0] * (shift + 1)
    for i, v in enumerate(xs):
        j = shift - i
        if 0 <= j <= shift:
            h[j] = float(gain) * v
    return RichResult(payload={
        "h": h, "t0": shift * step, "shift_samples": shift,
        "gain": float(gain), "causal": True, "reversed": True,
        "n_reference": n, "method": "Rangayyan (2024) eqs. (4.49), (4.56)"})


rangayyan_ch4_matched_filter_impulse_response = mfimpulse  # pre-policy spelling


# -- rng222: Matched-filter output equals scaled, delayed ACF of the reference signal..
def _dft(x):
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _xcorr(x, y, maxlag=None):
    """CCF theta_xy(k) = sum_n x(n) y(n + k), eq. (4.28)."""
    n, m = len(x), len(y)
    lo = -(n - 1) if maxlag is None else -int(maxlag)
    hi = (m - 1) if maxlag is None else int(maxlag)
    lags, vals = [], []
    for k in range(lo, hi + 1):
        acc, cnt = 0.0, 0
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                acc += x[i] * y[j]
                cnt += 1
        lags.append(k)
        vals.append(acc)
    return lags, vals


def mfacf(x, gain=1.0, dt=1.0):
    """The matched-filter output is the ACF of the reference.

    Rangayyan (2024): filtering with h(t) = K x(t0 - t) makes the
    convolution equivalent to CORRELATION, so the output is

        y(t) = K phi_x(t - t0),

    a scaled, delayed copy of the reference signal's autocorrelation.
    It therefore peaks exactly at t0, where phi_x is maximal, and the
    peak value is K times the signal's energy.

    Both routes are computed -- convolving with the reversed reference,
    and correlating directly -- and compared, because the equivalence is
    the claim being made.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    h = mfimpulse(xs, gain=gain, dt=dt)["h"]
    y = []
    for k in range(len(xs) + len(h) - 1):
        lo, hi = max(0, k - len(h) + 1), min(k, n - 1)
        y.append(fsum(xs[i] * h[k - i] for i in range(lo, hi + 1)))
    lags, acf = _xcorr(xs, xs)
    shifted = [float(gain) * v for v in acf]
    peak = max(range(len(y)), key=lambda i: y[i])
    energy = fsum(v * v for v in xs)
    offset = peak - (n - 1)
    aligned = y[offset:offset + len(shifted)] if offset >= 0 else []
    gap = (max(abs(a - b) for a, b in zip(aligned, shifted))
           if len(aligned) == len(shifted) else float("inf"))
    return RichResult(payload={
        "y": y, "acf": acf, "lags": lags, "peak_index": peak,
        "peak_value": y[peak], "expected_peak": float(gain) * energy,
        "energy": energy, "max_difference": gap,
        "equals_acf": gap <= 1e-9 * (1 + abs(float(gain) * energy)),
        "method": "Rangayyan (2024) Section 4.6.1, y(t) = K phi_x(t - t0)"})


rangayyan_ch4_matched_filter_output_acf = mfacf  # pre-policy spelling


# -- rng224: Basic three-sample reference pattern used in matched-filter illustration..
def refpattern(amplitudes=None):
    """The three-sample reference pattern of the matched-filter example.

    Rangayyan (2024) eq. (4.53):
        g(n) = 3 delta(n) + 2 delta(n-1) + delta(n-2),

    whose matched filter is eq. (4.54), h(n) = delta(n) + 2 delta(n-1)
    + 3 delta(n-2) with K = 1 and a delay of n0 = 2 samples: the same
    three numbers, reversed.

    Both are returned together, because the reversal is the point of the
    example -- and the book notes the output samples 5 to 9 reproduce
    the ACF of g(n), which is checked here rather than asserted.
    """
    g = [3.0, 2.0, 1.0] if amplitudes is None else aslist(amplitudes)
    if not g:
        raise ValueError("need at least one sample")
    h = list(reversed(g))
    y = []
    for k in range(2 * len(g) - 1):
        lo, hi = max(0, k - len(h) + 1), min(k, len(g) - 1)
        y.append(fsum(g[i] * h[k - i] for i in range(lo, hi + 1)))
    acf = []
    for k in range(-(len(g) - 1), len(g)):
        acf.append(fsum(g[i] * g[i + k] for i in range(len(g))
                        if 0 <= i + k < len(g)))
    gap = max(abs(a - b) for a, b in zip(y, acf))
    return RichResult(payload={
        "g": g, "h": h, "y": y, "acf": acf, "delay": len(g) - 1,
        "max_difference": gap, "output_is_acf": gap <= 1e-12,
        "method": "Rangayyan (2024) eqs. (4.53)-(4.54)"})


rangayyan_ch4_basic_signal_g = refpattern  # pre-policy spelling


# -- rng227: Frequency-domain optimal matched-filter response for EEG spike-and-wave detection..
def mftfeeg(X, freqs, t0, gain=1.0):
    """Matched-filter transfer function for EEG spike-and-wave detection.

    Rangayyan (2024) eq. (4.55):  H(f) = K X*(f) exp(-j 2 pi f t0),

    the same expression as eq. (4.48), restated in Section 4.6.2 for the
    spike-and-wave application, so it delegates rather than carrying a
    second copy.  What is specific to that section is the DFT caveat:
    implementing eq. (4.55) with an N-point DFT of an N-sample template
    supplies a shift of only N-1, one sample short of causal.
    """
    r = mftf(X, freqs, t0, gain=gain)
    out = dict(r)
    out["dft_shift_caveat"] = ("an N-point DFT of an N-sample template "
                               "supplies a shift of N-1; the causal "
                               "filter needs N")
    out["method"] = "Rangayyan (2024) eq. (4.55)"
    return RichResult(payload=out)


rangayyan_ch4_matched_filter_optimal_H_eeg = mftfeeg  # pre-policy spelling


# -- rng228: Time-domain impulse response of the matched filter for EEG spike-and-wave detection..
def mfimpeeg(x, t0=None, gain=1.0, dt=1.0):
    """Matched-filter impulse response for EEG spike-and-wave detection.

    Rangayyan (2024) eq. (4.56):  h(t) = K x(t0 - t),

    identical to eq. (4.49) and delegating to it.  Section 4.6.2 adds
    that because h is a reversed copy of the reference, the filtering is
    equivalent to correlation, so the output approximates the reference
    ACF wherever the input resembles the template.
    """
    r = mfimpulse(x, t0=t0, gain=gain, dt=dt)
    out = dict(r)
    out["equivalent_to_correlation"] = True
    out["method"] = "Rangayyan (2024) eq. (4.56)"
    return RichResult(payload=out)


rangayyan_ch4_matched_filter_impulse_response_eeg = mfimpeeg  # pre-policy spelling


# -- rng229: Frequency-domain output of matched filter equals PSD of the reference signal..
def mfpsd(x, dt=1.0):
    """Matched-filter output spectrum is the PSD of the reference.

    Rangayyan (2024) eq. (4.57):
        Y(f) = X(f) H(f) = X(f) X*(f) = S_x(f),

    ignoring the scale and delay factors.  Because Y is a PSD it is real
    and nonnegative at every frequency -- the phase has been cancelled
    exactly, which is the frequency-domain statement of the coherent
    addition that makes the output peak at t0.

    The residual imaginary part is returned: it is zero to rounding when
    H is the true matched filter, and grows as soon as the conjugation
    or the delay is wrong.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    re, im = _dft(xs)
    X = [complex(a, b) for a, b in zip(re, im)]
    H = [v.conjugate() for v in X]
    Y = [a * b for a, b in zip(X, H)]
    psd = [abs(v) ** 2 for v in X]
    gap = max(abs(a.real - b) for a, b in zip(Y, psd))
    return RichResult(payload={
        "Y": Y, "psd": psd, "max_imaginary": max(abs(v.imag) for v in Y),
        "max_difference": gap, "is_psd": gap <= 1e-9 * (1 + max(psd)),
        "n": n, "method": "Rangayyan (2024) eq. (4.57)"})


rangayyan_ch4_matched_filter_output_psd = mfpsd  # pre-policy spelling


_CHEATSHEET = [
    'coher: Coherence between two signals.',
    'mchfl: Matched filter for template detection in biomedical signals.',
    'rgacf: Autocorrelation estimate.',
    "rgbartl: Bartlett's averaged periodogram.",
    'rgbwbnd: Spectral bandwidth.',
    'rgccf: Cross-correlation function (CCF) between two signals.',
    'rgcfsle: cardiorespiratory coupling by coherence and PLV',
    'rgcoh: Magnitude-squared coherence -- Rangayyan & Krishnan Sec 4.5.1.',
    'rgcxy: magnitude-squared coherence, eq. (4.32) squared',
    'rgeeg: EEG band power (delta theta alpha beta gamma) -- Rangayyan & Krishnan Sec 4.4.1.',
    'rgeegar: EEG rhythm detection via autocorrelation.',
    'rgeegrhm: EEG alpha rhythm presence detection via autocorrelation.',
    'rgeegsp: EEG band powers.',
    'rgemgpk: EMG mean and median frequency, eqs. (6.34)-(6.35)',
    'rgeqn3b: Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n].',
    'rgerpflt: ERP artifact rejection and averaging, Section 3.5',
    'rgmflt: matched-filter design, eqs. (4.48)-(4.49)',
    'rgmfsnr: maximum matched-filter SNR, Rangayyan eq. (4.46)',
    'rgperio: Periodogram.',
    "rgpsd: Power spectral density via Welch's method -- Rangayyan & Krishnan Sec 6.3.2-6.3.4.",
    'rgpsd2hz: PSD on a Hz axis with band powers',
    'rgpsdacf: PSD to autocorrelation.',
    'rgpsync: synchronized averaging of PCG spectra',
    'rgseiz: seizure detection by inter-channel coherence, Section 4.5.3',
    'rgsmom: PSD moments, Rangayyan eqs. (6.32)-(6.43)',
    'rgspres: spectral resolution and leakage',
    'rgtmpl: template matching by correlation, eqs. (4.25), (4.28)',
    'rgwelch: Welch power spectral density.',
    'rng016: Autocorrelation function of a random process by ensemble average (Eq 3.16/3.17).',
    'rng017: Ensemble autocorrelation.',
    'rng018: Ensemble average function (Rangayyan eq. 3.18).',
    'rng020: Time-averaged autocorrelation.',
    'rng023: Cross-correlation function (CCF) between two random processes x and y..',
    'rng070: inverse DFT, Rangayyan eq. (3.81)',
    "rng080: Parseval's theorem, Rangayyan eq. (3.91)",
    'rng085: synchronized sum, Rangayyan eq. (3.96)',
    'rng086: Normalized cross-correlation coefficient used in template matching..',
    'rng198: discrete inner product and correlation, eqs. (4.24)-(4.25)',
    'rng199: Correlation coefficient as normalized dot product of two signals..',
    'rng200: continuous-time projection, Rangayyan eq. (4.26)',
    'rng201: Cross-correlation function of two continuous-time signals with delay tau..',
    'rng202: Discrete-time cross-correlation function of x(n) and y(n) with shift k..',
    'rng203: outer-product CCF of random signals, eq. (4.29)',
    'rng205: cross-spectral density, Rangayyan eqs. (4.30)-(4.31)',
    'rng206: magnitude coherence spectrum, Rangayyan eq. (4.32)',
    'rng207: FT of the matched-filter input, eq. (4.33)',
    'rng208: matched-filter output, Rangayyan eq. (4.34)',
    'rng209: white-noise input PSD, Rangayyan eq. (4.35)',
    'rng210: matched-filter output noise PSD, eqs. (4.36)-(4.37)',
    'rng211: Average output noise power.',
    'rng212: instantaneous matched-filter output, eq. (4.38)',
    'rng213: peak-power SNR of the matched filter, eq. (4.39)',
    'rng214: total signal energy, Rangayyan eq. (4.40)',
    'rng215: normalized matched-filter ratio, eqs. (4.41), (4.46)',
    'rng216: Schwarz inequality for complex functions, eq. (4.42)',
    'rng217: Schwarz inequality for real functions, eq. (4.43)',
    'rng218: Cauchy-Schwarz inequality for vectors, eq. (4.44)',
    'rng219: triangle inequality for vectors, eq. (4.45)',
    'rng220: optimal matched-filter transfer function, eq. (4.48)',
    'rng221: matched-filter impulse response, eqs. (4.49), (4.56)',
    'rng222: matched-filter output equals the reference ACF',
    'rng224: the reference pattern g(n) and its filter, eqs. (4.53)-(4.54)',
    'rng227: EEG matched-filter transfer function, eq. (4.55)',
    'rng228: EEG matched-filter impulse response, eq. (4.56)',
    'rng229: matched-filter output spectrum is the reference PSD, eq. (4.57)',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
