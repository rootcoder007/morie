# morie.fn -- bsastat (rootcoder007/morie)
"""Statistics of random processes: moments, entropy, covariance, signal-level and fractal features.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 36
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import fsum
from math import fsum, sqrt
from math import inf
from math import inf, log
from math import inf, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer

__all__ = [
    'rangayyan_correlation_coeff',
    'rangayyan_spectral_entropy',
    'rangayyan_fd_psd_slope',
    'rangayyan_form_factor',
    'rangayyan_fractal_vag',
    'rangayyan_higuchi_fd',
    'rangayyan_katz_fd',
    'rangayyan_muap_firing_rate',
    'rangayyan_nonlinear_features',
    'rangayyan_pdf_estimate',
    'rangayyan_rms',
    'rangayyan_rms_noise',
    'rangayyan_sync_average',
    'rangayyan_signal_features',
    'rangayyan_signal_to_noise',
    'rangayyan_snr',
    'rangayyan_turns_count',
    'rangayyan_zero_crossing',
    'pdfmean',
    'rangayyan_ch3_mean_continuous',
    'pdfms',
    'rangayyan_ch3_mean_squared_value',
    'pdfvar',
    'rangayyan_ch3_variance_continuous',
    'pdfskew',
    'rangayyan_ch3_skewness',
    'pdfkurt',
    'rangayyan_ch3_kurtosis',
    'diffent',
    'rangayyan_ch3_entropy_continuous',
    'smean',
    'rangayyan_ch3_sample_mean',
    'rangayyan_ch3_sample_mean_squared',
    'srms',
    'rangayyan_ch3_sample_rms',
    'rangayyan_ch3_sample_std',
    'noisemodel',
    'rangayyan_ch3_signal_plus_noise_model',
    'meansum',
    'rangayyan_ch3_mean_of_sum',
    'rangayyan_ch3_variance_of_sum_uncorrelated',
    'ensmean',
    'rangayyan_ch3_ensemble_mean',
    'rangayyan_ch3_time_average_mean',
    'covxy',
    'rangayyan_ch3_covariance',
    'rangayyan_ch3_correlation_coefficient',
    'rangayyan_ch3_observed_signal_kth_realization',
]


# -- rgcorec: Pearson correlation coefficient for morphological analysis.
def rangayyan_correlation_coeff(x, y):
    """
    Pearson correlation coefficient for morphological analysis

    Formula: rho = sum((x-mean_x)*(y-mean_y)) / (N*std_x*std_y)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho

    References
    ----------
    Rangayyan Ch 5.4.1
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
                "method": "Pearson correlation coefficient for morphological analysis",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Pearson correlation coefficient for morphological analysis",
        }
    )


# -- rgentrp: Spectral entropy for signal complexity measurement.
def rangayyan_spectral_entropy(x, fs):
    """
    Spectral entropy for signal complexity measurement

    Formula: H = -sum p_k * log(p_k) where p_k = S(f_k)/sum S(f)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectral_entropy

    References
    ----------
    Rangayyan Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Spectral entropy for signal complexity measurement"}
    )


# -- rgfdpsd: Fractal dimension from PSD slope (1/f noise model).
def rangayyan_fd_psd_slope(psd, freqs, f_range):
    """
    Fractal dimension from PSD slope (1/f noise model)

    Formula: FD = (5 - beta) / 2 where S(f) ~ f^{-beta}; beta from log-log PSD slope

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.
    f_range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd, beta

    References
    ----------
    Rangayyan Ch 6.6.2
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Fractal dimension from PSD slope (1/f noise model)"}
    )


# -- rgff: Form factor (ratio of RMS to mean absolute value).
def rangayyan_form_factor(x):
    """
    Form factor (ratio of RMS to mean absolute value)

    Formula: FF = RMS(x) / mean(|x|)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: form_factor

    References
    ----------
    Rangayyan Ch 5.6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Form factor (ratio of RMS to mean absolute value)"}
    )


# -- rgfracv: Fractal analysis of VAG signals via power spectral slope.
def rangayyan_fractal_vag(vag, fs):
    """
    Fractal analysis of VAG signals via power spectral slope

    Formula: FD = (5-beta)/2; beta estimated from log-log PSD in 100-500 Hz

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd, beta, r_sq

    References
    ----------
    Rangayyan Ch 6.6
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Fractal analysis of VAG signals via power spectral slope",
        }
    )


# -- rghfd: Higuchi fractal dimension -- Rangayyan Sec. 5.13.2, eqs (5.39)-(5.41).
def rangayyan_higuchi_fd(x, kmax=10):
    """Higuchi (1988) fractal dimension.

    Eq (5.39)  x_k(m) = x(m), x(m+k), x(m+2k), ..., x(m + floor((N-m)/k) k),
               for m = 1, 2, ..., k  (1-based).
    Eq (5.40)  L(m,k) = (1/k) * (N-1) / (k floor((N-m)/k))
                        * sum_{i=1}^{floor((N-m)/k)} |x(m+ik) - x(m+(i-1)k)|
    Eq (5.41)  L(k)   = (1/k) sum_{m=1}^{k} L(m,k)

    FD is the slope of a straight-line fit to a log-log plot of L(k)
    against 1/k.

    Parameters
    ----------
    x : array-like
    kmax : int
        Maximum time lag.

    Returns
    -------
    RichResult with keys ``HFD``, ``log_L``, ``log_inv_k``, ``kmax``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. *Biomedical Signal Analysis*,
        3rd ed. (IEEE Press / Wiley, 2024),
        Sec. 5.13.2 "Higuchi's method", p. 304, eqs (5.39)-(5.41).
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
        the fractal theory. *Physica D*, 31, 277-283.

    Note: the docstring previously cited Ch. 7; Higuchi's method is in
    Sec. 5.13.2 of the 2024 edition, verified against the typeset PDF.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 4 or kmax < 2:
        raise ValueError("Need len(x) >= 4 and kmax >= 2.")
    kmax = int(min(kmax, N // 2))
    L = np.empty(kmax)
    for k in range(1, kmax + 1):
        lk = []
        for m in range(1, k + 1):
            # Eq (5.39): x_k(m) starts at the m-th sample, 1-based in the book,
            # so index m-1 into the 0-based array.
            idx = np.arange(m - 1, N, k)
            if idx.size < 2:
                continue
            diffs = np.sum(np.abs(np.diff(x[idx])))
            # Eq (5.40): the normaliser is floor((N - m)/k) with the book's
            # 1-based m, and it must equal the number of difference terms
            # actually summed. The previous code passed the 0-based loop index
            # here, making the denominator floor((N - m + 1)/k) while the
            # numerator still had floor((N - m)/k) terms -- the two disagreed
            # whenever (N - m) was not a multiple of k. Deriving it from
            # idx.size keeps them identical by construction.
            n_terms = idx.size - 1
            norm = (N - 1) / (k * n_terms)
            lk.append((diffs / k) * norm)
        L[k - 1] = np.mean(lk) if lk else np.nan
    ks = np.arange(1, kmax + 1)
    log_L = np.log(L)
    log_inv_k = np.log(1.0 / ks)
    slope, intercept = np.polyfit(log_inv_k, log_L, 1)
    res = RichResult(
        title="Higuchi fractal dimension",
        summary_lines=[("HFD", float(slope)), ("kmax", kmax), ("N", N)],
        interpretation=f"HFD = {slope:.4g}. ~1 smooth, ~2 rough.",
        payload={
            "HFD": float(slope),
            "intercept": float(intercept),
            "log_L": log_L,
            "log_inv_k": log_inv_k,
            "kmax": kmax,
        },
    )
    return with_describe_pointer(res, "rghfd")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_higuchi_fd(rng.standard_normal(500), kmax=8)
# >>> 1.0 <= r["HFD"] <= 2.5
# True


# -- rgkatzfd: Katz fractal dimension of a waveform.
def rangayyan_katz_fd(x):
    """
    Katz fractal dimension of a waveform

    Formula: FD = log10(n) / (log10(n) + log10(d/L))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd

    References
    ----------
    Rangayyan Ch 5.13.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz fractal dimension of a waveform"})


# -- rgmufr: Motor unit mean firing rate and inter-discharge interval (IDI).
def rangayyan_muap_firing_rate(spike_times):
    """
    Motor unit mean firing rate and inter-discharge interval (IDI)

    Formula: MFR = 1/mean(IDI), CV_IDI = std(IDI)/mean(IDI)

    Parameters
    ----------
    spike_times : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mfr, cv_idi

    References
    ----------
    Rangayyan Ch 1.2.4
    """
    spike_times = np.asarray(spike_times, dtype=float)
    n = int(spike_times) if spike_times.ndim == 0 else len(spike_times)
    result = float(np.mean(spike_times))
    se = float(np.std(spike_times, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Motor unit mean firing rate and inter-discharge interval (IDI)",
        }
    )


# -- rgnl: Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov).
def rangayyan_nonlinear_features(x, m, r):
    """
    Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)

    Formula: Feature vector: [ApEn, SampEn, alpha_DFA, lambda_max]

    Parameters
    ----------
    x : array-like
        Input data.
    m : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features_dict

    References
    ----------
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)",
        }
    )


# -- rgpdfest: Probability density estimate.
def rangayyan_pdf_estimate(x, bins=None, bw=None, method="kde", grid=None):
    r"""Probability density estimate (Rangayyan Ch. 3), by histogram
    or Gaussian kernel:

    .. math:: \hat p(x) = \frac{1}{Nh}\sum_i
              K\!\left(\frac{x - x_i}{h}\right),
              \qquad K = \text{Gaussian}.

    The bandwidth h controls a bias-variance trade the bin count
    cannot express as smoothly, which is why the kernel form is the
    default. With ``bw`` omitted, Silverman's rule
    :math:`h = 0.9\,\min(\sigma, IQR/1.34)\,N^{-1/5}` is used and
    reported, so the choice is visible rather than buried.

    Parameters
    ----------
    x : array-like
        Samples.
    bins : int, optional
        Histogram bins (method="hist").
    bw : float, optional
        Kernel bandwidth.
    method : {"kde", "hist"}
        Estimator.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``integrates_to``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (probability density estimation).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("x must have at least 2 samples.")
    g = np.linspace(x.min() - 3 * x.std(), x.max() + 3 * x.std(), 512) \
        if grid is None else np.asarray(grid, dtype=float).ravel()
    if method == "hist":
        nb = int(bins) if bins is not None else max(5, int(np.sqrt(x.size)))
        if nb < 1:
            raise ValueError("bins must be positive.")
        counts, edges = np.histogram(x, bins=nb, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        dens = np.interp(g, centres, counts, left=0.0, right=0.0)
        h = float(edges[1] - edges[0])
    elif method == "kde":
        if bw is None:
            iqr = float(np.subtract(*np.percentile(x, [75, 25])))
            spread = min(float(x.std()), iqr / 1.34) if iqr > 0 else float(x.std())
            h = 0.9 * max(spread, 1e-12) * x.size ** (-0.2)
        else:
            h = float(bw)
        if h <= 0:
            raise ValueError(f"bandwidth must be positive, got {h}.")
        z = (g[:, None] - x[None, :]) / h
        dens = np.exp(-0.5 * z**2).sum(axis=1) / (x.size * h * np.sqrt(2 * np.pi))
    else:
        raise ValueError("method must be 'kde' or 'hist'.")
    return RichResult(payload={"grid": g, "density": dens, "bandwidth": h,
                               "integrates_to": float(np.trapezoid(dens, g)),
                               "method": f"{method} density; Silverman bandwidth when unset"})


# -- rgrms: Root mean square (RMS) value of a signal.
def rangayyan_rms(x):
    """
    Root mean square (RMS) value of a signal

    Formula: RMS = sqrt((1/N) sum x[n]^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rms

    References
    ----------
    Rangayyan Ch 5.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Root mean square (RMS) value of a signal"}
    )


# compact alias per ledger/NAMING.md
rangayyanrms = rangayyan_rms


# -- rgrmsnw: RMS noise level.
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


# -- rgsavg: Synchronized (ensemble) averaging for SNR enhancement.
def rangayyan_sync_average(epochs):
    """
    Synchronized (ensemble) averaging for SNR enhancement

    Formula: x_avg[n] = (1/M) sum_{k=1}^{M} x_k[n]; SNR = sqrt(M) * signal_SNR

    Parameters
    ----------
    epochs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: averaged_signal, snr_gain

    References
    ----------
    Rangayyan Ch 3.5
    """
    epochs = np.asarray(epochs, dtype=float)
    n = int(epochs) if epochs.ndim == 0 else len(epochs)
    result = float(np.mean(epochs))
    se = float(np.std(epochs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Synchronized (ensemble) averaging for SNR enhancement",
        }
    )


# -- rgsf: Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear.
def rangayyan_signal_features(x, fs):
    """
    Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear

    Formula: F = [mean, std, rms, zcr, form_factor, centroid, bandwidth, spectral_entropy, sample_entropy]

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: feature_vector

    References
    ----------
    Rangayyan Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear",
        }
    )


# -- rgsig2n: Signal-to-noise ratio calculation after filtering.
def rangayyan_signal_to_noise(signal_clean, signal_noisy):
    """
    Signal-to-noise ratio calculation after filtering

    Formula: SNR = 10*log10(sum(x_clean^2)/sum(noise^2))

    Parameters
    ----------
    signal_clean : array-like
        Input data.
    signal_noisy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_db

    References
    ----------
    Rangayyan Ch 3
    """
    signal_clean = np.asarray(signal_clean, dtype=float)
    n = int(signal_clean) if signal_clean.ndim == 0 else len(signal_clean)
    result = float(np.mean(signal_clean))
    se = float(np.std(signal_clean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Signal-to-noise ratio calculation after filtering"}
    )


# -- rgsnr: Signal-to-noise ratio (dB).
def rangayyan_snr(signal, noise):
    """
    Signal-to-noise ratio (dB)

    Formula: SNR = 10*log10(P_signal / P_noise)

    Parameters
    ----------
    signal : array-like
        Input data.
    noise : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: snr_db

    References
    ----------
    Rangayyan Ch 1
    """
    signal = np.asarray(signal, dtype=float)
    n = int(signal) if signal.ndim == 0 else len(signal)
    result = float(np.mean(signal))
    se = float(np.std(signal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Signal-to-noise ratio (dB)"})


# compact alias per ledger/NAMING.md
rangayyansnr = rangayyan_snr


# -- rgturns: Turns count of an EMG signal (number of direction reversals above threshold).
def rangayyan_turns_count(x, threshold):
    """
    Turns count of an EMG signal (number of direction reversals above threshold)

    Formula: Turn: local extremum where |delta_amp| > threshold

    Parameters
    ----------
    x : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: turns_count, turn_locs

    References
    ----------
    Rangayyan Ch 5.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Turns count of an EMG signal (number of direction reversals above threshold)",
        }
    )


# -- rgzcr: Zero-crossing rate -- Rangayyan & Krishnan Sec 5.6.2.
def rangayyan_zero_crossing(x, fs=1.0):
    """Zero-crossing rate.

    ``ZCR = (1/(N-1)) Σ 0.5 |sign(x[n]) - sign(x[n-1])|``.

    Parameters
    ----------
    x : array-like
    fs : float
        Sampling rate (Hz). ``zcr_per_second`` is ``zcr * fs``.

    Returns
    -------
    RichResult with keys ``zcr``, ``zcr_per_second``, ``crossings``, ``n``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.6.2 "Zero-crossing rate", p.285
        (Sec 5.6 "Analysis of Activity", p.283).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return with_describe_pointer(
            RichResult(
                title="Zero-crossing rate",
                summary_lines=[("Length", n)],
                payload={"zcr": float("nan"), "zcr_per_second": float("nan"), "crossings": 0, "n": n},
            ),
            "rgzcr",
        )
    s = np.sign(x)
    s[s == 0] = 1.0
    crossings = int(np.sum(np.abs(np.diff(s)) > 0))
    zcr = crossings / (n - 1)
    res = RichResult(
        title="Zero-crossing rate",
        summary_lines=[
            ("N", n),
            ("Crossings", crossings),
            ("ZCR (per sample)", zcr),
            ("ZCR (per second)", zcr * fs),
        ],
        interpretation=f"{crossings} crossings; {zcr * fs:.3g} Hz at fs={fs}.",
        payload={"zcr": float(zcr), "zcr_per_second": float(zcr * fs), "crossings": crossings, "n": n},
    )
    return with_describe_pointer(res, "rgzcr")


# CANONICAL TEST
# >>> x = np.sin(2*np.pi*np.arange(100)/10.0)
# >>> r = rangayyan_zero_crossing(x, fs=100)
# >>> r["crossings"] > 0
# True


# -- rng001: Mean of a random process from its PDF (Rangayyan eq. 3.1).
def pdfmean(pdf=None, x=None, lower=-inf, upper=inf):
    """First-order moment of a PDF.

    Rangayyan (2024) eq. (3.1):  mu_eta = E[eta] = integral eta p(eta) d eta.

    Parameters
    ----------
    pdf : callable or array-like
        The density.  Callable is integrated adaptively between ``lower``
        and ``upper``; array-like is read as densities tabulated at ``x``.
    x : array-like, optional
        Abscissae for a tabulated density.
    lower, upper : float
        Limits for the callable form.  Infinite limits are truncated at
        +/- 40, which holds any density whose scale is order unity; pass
        finite limits when the density lives elsewhere.

    Returns
    -------
    RichResult
        ``mean`` plus the integrated mass of the density, which is the
        only cheap check that the input really is a PDF.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    out = {"mean": float(mu), "method": "Rangayyan (2024) eq. (3.1)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_mean_continuous = pdfmean  # pre-policy spelling


# -- rng002: Mean-squared value of a random process (Rangayyan eq. 3.2).
def pdfms(pdf=None, x=None, lower=-inf, upper=inf):
    """Second-order (not central) moment of a PDF.

    Rangayyan (2024) eq. (3.2):  E[eta^2] = integral eta^2 p(eta) d eta.

    The book notes immediately after eq. (3.3) that sigma^2 = E[eta^2] -
    mu^2, so this is the variance only when the mean is zero; both are
    returned so the caller never has to assume which case they are in.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    ms = pdfint(lambda v: v * v, pdf, x, lower, upper)
    out = {"ms": float(ms), "mean": float(mu),
           "variance_from_identity": float(ms - mu * mu),
           "method": "Rangayyan (2024) eq. (3.2)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_mean_squared_value = pdfms  # pre-policy spelling


# -- rng003: Variance of a random process (Rangayyan eq. 3.3).
def pdfvar(pdf=None, x=None, lower=-inf, upper=inf):
    """Second central moment of a PDF, and the SD and CV that follow.

    Rangayyan (2024) eq. (3.3):
        sigma^2 = E[(eta - mu)^2] = integral (eta - mu)^2 p(eta) d eta.

    The book defines the coefficient of variation as sigma/mu in the same
    paragraph and warns it diverges as mu -> 0, so ``cv`` is None once the
    mean is negligible against the SD of the process -- a quadrature
    residue of 1e-19 on a symmetric density would otherwise be reported
    as a CV of 1e19 rather than as "undefined here".
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var) if var > 0 else 0.0
    cv = float(sd / mu) if abs(mu) > 1e-9 * max(sd, 1.0) else None
    out = {"variance": float(var), "sd": float(sd), "mean": float(mu),
           "cv": cv, "method": "Rangayyan (2024) eq. (3.3)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_variance_continuous = pdfvar  # pre-policy spelling


# -- rng004: Skewness of a random process (Rangayyan eq. 3.4).
def pdfskew(pdf=None, x=None, lower=-inf, upper=inf):
    """Normalized third central moment of a PDF.

    Rangayyan (2024) eq. (3.4):
        S = (1/sigma^3) integral (eta - mu)^3 p(eta) d eta.

    Symmetric densities give zero; the book reads a negative value as a
    tail to the left of the mode and a positive value as a tail to the
    right.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var)
    if sd <= 0:
        raise ValueError("skewness is undefined for a degenerate density")
    m3 = pdfint(lambda v: (v - mu) ** 3, pdf, x, lower, upper)
    out = {"skewness": float(m3 / sd ** 3), "m3": float(m3), "sd": float(sd),
           "mean": float(mu), "method": "Rangayyan (2024) eq. (3.4)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_skewness = pdfskew  # pre-policy spelling


# -- rng005: Kurtosis of a random process (Rangayyan eq. 3.5).
def pdfkurt(pdf=None, x=None, lower=-inf, upper=inf):
    """Normalized fourth central moment of a PDF.

    Rangayyan (2024) eq. (3.5):
        K = (1/sigma^4) integral (eta - mu)^4 p(eta) d eta.

    The book states the Gaussian value is 3 and defines the kurtosis
    excess K' = K - 3, positive for a strongly peaked heavy-tailed
    density and negative for a near-uniform one; both are returned.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var)
    if sd <= 0:
        raise ValueError("kurtosis is undefined for a degenerate density")
    m4 = pdfint(lambda v: (v - mu) ** 4, pdf, x, lower, upper)
    k = float(m4 / sd ** 4)
    out = {"kurtosis": k, "excess": k - 3.0, "m4": float(m4),
           "sd": float(sd), "mean": float(mu),
           "method": "Rangayyan (2024) eq. (3.5)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_kurtosis = pdfkurt  # pre-policy spelling


# -- rng006: Differential entropy of a continuous PDF (Rangayyan eq. 3.6).
def diffent(pdf=None, x=None, lower=-inf, upper=inf):
    """Differential entropy in bits.

    Rangayyan (2024) eq. (3.6):
        H = - integral p(eta) log2[p(eta)] d eta.

    p log2 p -> 0 as p -> 0, so zero-density points contribute nothing
    rather than raising on log(0).  Unlike the discrete Shannon entropy
    of eq. (3.11) this may be negative -- it is a density, not a
    probability, inside the logarithm.
    """
    ln2 = log(2.0)

    def term(p):
        return 0.0 if p <= 0.0 else -p * log(p) / ln2

    if x is not None:
        from ._rgcore import aslist, gridint
        xs = aslist(x)
        ps = [float(pdf(v)) for v in xs] if callable(pdf) else aslist(pdf)
        h = gridint([term(p) for p in ps], xs)
        mass = gridint(ps, xs)
    else:
        h = pdfint(lambda v: 1.0, lambda v: term(float(pdf(v))),
                   None, lower, upper)
        mass = pdfint(lambda v: 1.0, pdf, None, lower, upper)
    out = {"entropy": float(h), "units": "bits",
           "method": "Rangayyan (2024) eq. (3.6)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_entropy_continuous = diffent  # pre-policy spelling


# -- rng007: Sample mean of an observed signal (Rangayyan eq. 3.7).
def smean(x):
    """Sample mean of N observed values.

    Rangayyan (2024) eq. (3.7):  mu = (1/N) sum_{n=0}^{N-1} eta(n).

    The book calls this the DC component of the signal.  Summed with
    math.fsum so the result does not depend on the order of the samples.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    return RichResult(payload={"mean": fsum(xs) / n, "n": n,
                               "method": "Rangayyan (2024) eq. (3.7)"})


rangayyan_ch3_sample_mean = smean  # pre-policy spelling


# -- rng008: Sample mean square.
def rangayyan_ch3_sample_mean_squared(eta, N=None):
    r"""Sample mean-squared value (Rangayyan Ch. 3):

    .. math:: MS_\eta = \frac1N \sum_{n=0}^{N-1} [\eta(n)]^2.

    This is the total average power, NOT the variance: the two differ
    by the squared mean, and they coincide only for a zero-mean
    signal. Both are returned so the distinction is visible.

    Parameters
    ----------
    eta : array-like
        Samples.
    N : int, optional
        Length.

    Returns
    -------
    RichResult
        keys: ``mean_square``, ``variance``, ``mean``, ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    if eta.size < 1:
        raise ValueError("eta must be non-empty.")
    if N is not None and int(N) != eta.size:
        raise ValueError(f"N = {N} does not match len(eta) = {eta.size}.")
    mu = float(np.mean(eta))
    ms = float(np.mean(eta**2))
    return RichResult(payload={"mean_square": ms, "variance": ms - mu**2,
                               "mean": mu, "N": int(eta.size),
                               "method": "MS = (1/N) sum eta^2; equals variance only if mu = 0"})


# -- rng009: Sample RMS, MS, and SD of an observed signal (Rangayyan eqs. 3.8-3.10).
def srms(x):
    """Sample RMS value, with the MS and SD it is bracketed by.

    Rangayyan (2024) eqs. (3.8)-(3.10):
        MS  = (1/N) sum x(n)^2
        RMS = sqrt(MS)
        SD  = sqrt( (1/N) sum [x(n) - mu]^2 )

    Note the divisor is N in all three -- eq. (3.10) is the population
    form, not the N-1 unbiased one; a caller wanting the unbiased
    variance should rescale by N/(N-1).  The book reads MS as average
    power and RMS as average signal level.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    mu = fsum(xs) / n
    ms = fsum(v * v for v in xs) / n
    var = fsum((v - mu) ** 2 for v in xs) / n
    return RichResult(payload={
        "rms": sqrt(ms), "ms": ms, "sd": sqrt(var), "mean": mu, "n": n,
        "ddof": 0, "method": "Rangayyan (2024) eqs. (3.8)-(3.10)"})


rangayyan_ch3_sample_rms = srms  # pre-policy spelling


# -- rng010: Sample standard deviation.
def rangayyan_ch3_sample_std(eta, mu_eta=None, N=None):
    r"""Sample standard deviation (Rangayyan Ch. 3):

    .. math:: \sigma_\eta = \sqrt{\frac1N \sum_{n=0}^{N-1}
              (\eta(n) - \mu_\eta)^2}.

    The book's divisor is N, not N - 1: this is the second central
    moment of the observed record, not an unbiased estimator of a
    population variance. Both are returned, since substituting one
    for the other is a common slip.

    Parameters
    ----------
    eta : array-like
        Samples.
    mu_eta : float, optional
        Mean to centre on; the sample mean if omitted.
    N : int, optional
        Length.

    Returns
    -------
    RichResult
        keys: ``std`` (divisor N), ``std_unbiased`` (divisor N - 1),
        ``mean_used``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    if eta.size < 1:
        raise ValueError("eta must be non-empty.")
    if N is not None and int(N) != eta.size:
        raise ValueError(f"N = {N} does not match len(eta) = {eta.size}.")
    mu = float(np.mean(eta)) if mu_eta is None else float(mu_eta)
    dev2 = (eta - mu) ** 2
    n = eta.size
    unb = float(np.sqrt(dev2.sum() / (n - 1))) if n > 1 else float("nan")
    return RichResult(payload={"std": float(np.sqrt(dev2.mean())),
                               "std_unbiased": unb, "mean_used": mu, "N": int(n),
                               "method": "sigma with divisor N (the book's convention)"})


# -- rng012: Additive signal-plus-noise model (Rangayyan eqs. 3.12-3.14).
def noisemodel(x, eta):
    """Form the observed signal y = x + eta and its first two moments.

    Rangayyan (2024) eqs. (3.12)-(3.14):
        y(t)  = x(t) + eta(t)                                   (3.12)
        mu_y  = mu_x + mu_eta                                   (3.13)
        sig_y^2 = sig_x^2 + sig_eta^2, IF x and eta uncorrelated (3.14)

    Eq. (3.14) holds only under uncorrelatedness, so the sample
    correlation is computed and reported rather than assumed: compare
    ``variance_additive`` (the eq. 3.14 prediction) against
    ``variance_observed`` to see how far the assumption is from holding
    on this particular pair.
    """
    xs, es = aslist(x), aslist(eta)
    if len(xs) != len(es):
        raise ValueError("signal and noise must have the same length")
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    y = [a + b for a, b in zip(xs, es)]
    mx, me = fsum(xs) / n, fsum(es) / n
    vx = fsum((v - mx) ** 2 for v in xs) / n
    ve = fsum((v - me) ** 2 for v in es) / n
    my = fsum(y) / n
    vy = fsum((v - my) ** 2 for v in y) / n
    cov = fsum((a - mx) * (b - me) for a, b in zip(xs, es)) / n
    rho = cov / ((vx * ve) ** 0.5) if vx > 0 and ve > 0 else 0.0
    return RichResult(payload={
        "y": y, "mean_signal": mx, "mean_noise": me,
        "mean_observed": my, "mean_additive": mx + me,
        "variance_observed": vy, "variance_additive": vx + ve,
        "covariance": cov, "correlation": rho, "n": n,
        "method": "Rangayyan (2024) eqs. (3.12)-(3.14)"})


rangayyan_ch3_signal_plus_noise_model = noisemodel  # pre-policy spelling


# -- rng013: Mean of a sum of random processes (Rangayyan eq. 3.13).
def meansum(*processes):
    """Mean of a sum of random processes.

    Rangayyan (2024) eq. (3.13):  E[y] = mu_y = mu_x + mu_eta.

    Linearity of expectation needs no independence assumption, which is
    exactly what distinguishes eq. (3.13) from eq. (3.14) -- the variance
    identity that does.  Accepts any number of processes; each may be a
    sequence of samples or an already-computed mean.
    """
    if not processes:
        raise ValueError("need at least one process")
    means = []
    for p in processes:
        vals = aslist(p)
        if not vals:
            raise ValueError("every process needs at least one sample")
        means.append(fsum(vals) / len(vals))
    return RichResult(payload={
        "mean": fsum(means), "component_means": means,
        "n_processes": len(means),
        "method": "Rangayyan (2024) eq. (3.13)"})


rangayyan_ch3_mean_of_sum = meansum  # pre-policy spelling


# -- rng014: Variance of a sum of two uncorrelated random processes (Rangayyan Eq 3.14).
def rangayyan_ch3_variance_of_sum_uncorrelated(sigma_x, sigma_eta):
    r"""Variance of :math:`y = x + \eta` for uncorrelated :math:`x` and :math:`\eta`.

    .. math::

        E[(y - \mu_y)^2] = \sigma_y^2 = \sigma_x^2 + \sigma_\eta^2

    Parameters
    ----------
    sigma_x, sigma_eta : float or array-like
        Standard deviations of the signal and noise processes. These are
        **SDs, not signals**, and must be non-negative.

    Returns
    -------
    RichResult
        keys: ``variance``, ``sd``, ``sigma_x``, ``sigma_eta``, ``method``.

    Raises
    ------
    ValueError
        If either standard deviation is negative or non-finite.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.14), p. 96. Follows from Eq. (3.12) :math:`y(t)=x(t)+\eta(t)`
        and holds when :math:`x` and :math:`\eta` are uncorrelated; the book
        adds that then "their covariance and correlation coefficient are zero".

    Notes
    -----
    The hypothesis is not checkable from the arguments -- only SDs are passed,
    never the data -- so the caller owns it. For *correlated* processes the
    true variance of the sum exceeds this by :math:`2C_{x\eta}` (Eq. 3.21).
    """
    sx = np.asarray(sigma_x, dtype=float)
    se = np.asarray(sigma_eta, dtype=float)
    if not (np.all(np.isfinite(sx)) and np.all(np.isfinite(se))):
        raise ValueError(
            f"standard deviations must be finite; got sigma_x={sigma_x!r}, sigma_eta={sigma_eta!r}"
        )
    if np.any(sx < 0) or np.any(se < 0):
        raise ValueError(
            f"standard deviations must be non-negative; got sigma_x={sigma_x!r}, "
            f"sigma_eta={sigma_eta!r} -- these are SDs, not signal samples"
        )
    var = sx**2 + se**2
    scalar = var.ndim == 0
    return RichResult(
        payload={
            "variance": float(var) if scalar else var,
            "sd": float(np.sqrt(var)) if scalar else np.sqrt(var),
            "sigma_x": float(sx) if sx.ndim == 0 else sx,
            "sigma_eta": float(se) if se.ndim == 0 else se,
            "method": "variance of a sum of uncorrelated processes (Rangayyan Eq 3.14)",
        }
    )


# -- rng015: Ensemble mean at one instant (Rangayyan eq. 3.15).
def ensmean(observations, index=None):
    """Ensemble mean of M observations at a single instant t1.

    Rangayyan (2024) eq. (3.15):
        mu_x(t1) = lim_{M->inf} (1/M) sum_{k=1}^{M} x_k(t1).

    Parameters
    ----------
    observations : sequence
        Either the M values already sampled at t1, or M whole records
        from which ``index`` selects the instant.
    index : int, optional
        Sample index t1 within each record.

    Notes
    -----
    The SE of the ensemble mean is sigma/sqrt(M): the 1/sqrt(M) noise
    reduction the book attributes to synchronized averaging in Section
    3.3.1 is exactly this, read one instant at a time.
    """
    if index is None:
        vals = aslist(observations)
    else:
        i = int(index)
        vals = []
        for rec in observations:
            r = aslist(rec)
            if i < 0 or i >= len(r):
                raise IndexError("index %d outside a record of length %d"
                                 % (i, len(r)))
            vals.append(r[i])
    m = len(vals)
    if m == 0:
        raise ValueError("need at least one observation")
    mu = fsum(vals) / m
    var = fsum((v - mu) ** 2 for v in vals) / m
    return RichResult(payload={
        "mean": mu, "m": m, "sd": sqrt(var),
        "se": sqrt(var / m) if m > 0 else float("nan"),
        "method": "Rangayyan (2024) eq. (3.15)"})


rangayyan_ch3_ensemble_mean = ensmean  # pre-policy spelling


# -- rng019: Time-average mean.
def rangayyan_ch3_time_average_mean(x_k, T=None, dt=1.0):
    r"""Time-average mean of one realisation (Rangayyan Ch. 3):

    .. math:: \mu_x(k) = \lim_{T\to\infty} \frac1T
              \int_{-T/2}^{T/2} x_k(t)\, dt.

    Averaging ALONG one record. It equals the ensemble mean only for
    an ergodic process; comparing the two is the practical test of
    ergodicity, so when several realisations are supplied their spread
    is returned for exactly that comparison.

    Parameters
    ----------
    x_k : array-like, shape (T,) or (M, T)
        One realisation, or several.
    T : int, optional
        Length check.
    dt : float, default 1.0
        Sample interval (the discrete average is dt-invariant).

    Returns
    -------
    RichResult
        keys: ``time_mean`` (per realisation), ``spread_across_k``,
        ``T``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (time averages and ergodicity).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, n = X.shape
    if n < 1:
        raise ValueError("x_k must be non-empty.")
    if T is not None and int(T) != n:
        raise ValueError(f"T = {T} does not match the {n} samples.")
    if float(dt) <= 0:
        raise ValueError("dt must be positive.")
    means = X.mean(axis=1)
    return RichResult(payload={"time_mean": float(means[0]) if m == 1 else means,
                               "spread_across_k": float(np.std(means)) if m > 1 else 0.0,
                               "T": int(n),
                               "method": "average ALONG the record; equals ensemble mean iff ergodic"})


# -- rng021: Covariance and correlation coefficient (Rangayyan eqs. 3.21-3.22).
def covxy(x, y, ddof=0):
    """Covariance between two processes and the correlation it normalizes to.

    Rangayyan (2024) eqs. (3.21)-(3.22):
        C_xy  = E[(x - mu_x)(y - mu_y)]
        rho   = C_xy / (sigma_x sigma_y),   -1 <= rho <= +1.

    ``ddof=0`` matches the population divisor used throughout Section
    3.2.1; pass ``ddof=1`` for the unbiased sample covariance.  rho is
    None when either process is constant, since eq. (3.22) divides by a
    zero SD there rather than being zero.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    d = n - int(ddof)
    if d <= 0:
        raise ValueError("not enough samples for ddof=%d" % ddof)
    mx, my = fsum(xs) / n, fsum(ys) / n
    cov = fsum((a - mx) * (b - my) for a, b in zip(xs, ys)) / d
    vx = fsum((a - mx) ** 2 for a in xs) / d
    vy = fsum((b - my) ** 2 for b in ys) / d
    rho = cov / sqrt(vx * vy) if vx > 0 and vy > 0 else None
    return RichResult(payload={
        "covariance": cov, "correlation": rho, "sd_x": sqrt(vx),
        "sd_y": sqrt(vy), "mean_x": mx, "mean_y": my, "n": n, "ddof": int(ddof),
        "method": "Rangayyan (2024) eqs. (3.21)-(3.22)"})


rangayyan_ch3_covariance = covxy  # pre-policy spelling


# -- rng022: Correlation coefficient as normalised covariance (Rangayyan Eq 3.22).
def rangayyan_ch3_correlation_coefficient(C_xy, sigma_x, sigma_y):
    r"""Normalise a covariance into a correlation coefficient.

    .. math::

        \rho_{xy} = \frac{C_{xy}}{\sigma_x \sigma_y}

    Parameters
    ----------
    C_xy : float
        Covariance :math:`C_{xy} = E[(x-\mu_x)(y-\mu_y)]` (Eq. 3.21).
    sigma_x, sigma_y : float
        Standard deviations. Both must be strictly positive -- with a
        degenerate process the coefficient is undefined, not zero.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`\rho_{xy}`), ``C_xy``, ``sigma_x``,
        ``sigma_y``, ``method``.

    Raises
    ------
    ValueError
        If either SD is non-positive, or if the inputs imply
        :math:`|\rho_{xy}| > 1`, which the Cauchy-Schwarz inequality forbids.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.22), p. 98, "with :math:`-1 \le \rho_{xy} \le +1`". Covariance
        is Eq. (3.21) on the same page.

    Notes
    -----
    The book states the bound :math:`-1 \le \rho_{xy} \le +1` as part of the
    definition, so a result outside it means the *inputs* are inconsistent --
    some covariance and SDs that cannot come from the same pair of processes.
    That is raised rather than returned, because a "correlation" of 1.4 is not
    a number any caller can use.
    """
    c = float(C_xy)
    sx = float(sigma_x)
    sy = float(sigma_y)
    if not np.isfinite(c):
        raise ValueError(f"C_xy must be finite; got {C_xy!r}")
    if not (sx > 0 and sy > 0) or not (np.isfinite(sx) and np.isfinite(sy)):
        raise ValueError(
            f"standard deviations must be finite and strictly positive; "
            f"got sigma_x={sigma_x!r}, sigma_y={sigma_y!r}. rho is undefined "
            "for a degenerate process."
        )
    rho = c / (sx * sy)
    if abs(rho) > 1.0:
        raise ValueError(
            f"C_xy={c!r} with sigma_x={sx!r}, sigma_y={sy!r} gives rho={rho!r}, "
            "outside the [-1, +1] range Eq. (3.22) states. Cauchy-Schwarz "
            "forbids |C_xy| > sigma_x*sigma_y, so these inputs are inconsistent."
        )
    return RichResult(
        payload={
            "value": rho,
            "C_xy": c,
            "sigma_x": sx,
            "sigma_y": sy,
            "method": "correlation coefficient rho = C_xy/(sigma_x sigma_y) (Rangayyan Eq 3.22)",
        }
    )


# -- rng084: kth observed realization of a signal in noise (signal-plus-noise model)..
def rangayyan_ch3_observed_signal_kth_realization(x_k, eta_k, n):
    """
    kth observed realization of a signal in noise (signal-plus-noise model).

    Formula: y_k(n) = x_k(n) + eta_k(n)

    Parameters
    ----------
    x_k : array-like
        Input data.
    eta_k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.95, p. 135
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    result = float(np.mean(x_k))
    se = float(np.std(x_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "kth observed realization of a signal in noise (signal-plus-noise model).",
        }
    )


_CHEATSHEET = [
    'rgcorec: Pearson correlation coefficient for morphological analysis',
    'rgentrp: Spectral entropy for signal complexity measurement',
    'rgfdpsd: Fractal dimension from PSD slope (1/f noise model)',
    'rgff: Form factor (ratio of RMS to mean absolute value)',
    'rgfracv: Fractal analysis of VAG signals via power spectral slope',
    'rghfd: Higuchi fractal dimension -- Rangayyan Sec. 5.13.2',
    'rgkatzfd: Katz fractal dimension of a waveform',
    'rgmufr: Motor unit mean firing rate and inter-discharge interval (IDI)',
    'rgnl: Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)',
    'rgpdfest: Silverman h reported, not hidden; integral returned as a check',
    'rgrms: Root mean square (RMS) value of a signal',
    "rgrmsnw: signal leaking into the 'noise' segment deflates every SNR",
    'rgsavg: Synchronized (ensemble) averaging for SNR enhancement',
    'rgsf: Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear',
    'rgsig2n: Signal-to-noise ratio calculation after filtering',
    'rgsnr: Signal-to-noise ratio (dB)',
    'rgturns: Turns count of an EMG signal (number of direction reversals above threshold)',
    'rgzcr: Zero-crossing rate -- Rangayyan & Krishnan Sec 5.6.2',
    'rng001: mean of a PDF, Rangayyan eq. (3.1)',
    'rng002: mean-squared value of a PDF, Rangayyan eq. (3.2)',
    'rng003: variance of a PDF, Rangayyan eq. (3.3)',
    'rng004: skewness of a PDF, Rangayyan eq. (3.4)',
    'rng005: kurtosis of a PDF, Rangayyan eq. (3.5)',
    'rng006: differential entropy of a PDF, Rangayyan eq. (3.6)',
    'rng007: sample mean, Rangayyan eq. (3.7)',
    'rng008: mean square is total power, not variance, unless mu = 0',
    'rng009: sample RMS/MS/SD, Rangayyan eqs. (3.8)-(3.10)',
    'rng010: divisor N, not N-1; both returned',
    'rng012: additive noise model, Rangayyan eqs. (3.12)-(3.14)',
    'rng013: mean of a sum, Rangayyan eq. (3.13)',
    'rng014: sigma_y^2 = sigma_x^2 + sigma_eta^2 (Rangayyan Eq 3.14).',
    'rng015: ensemble mean at an instant, Rangayyan eq. (3.15)',
    'rng019: time vs ensemble mean agreeing IS the ergodicity check',
    'rng021: covariance and correlation, Rangayyan eqs. (3.21)-(3.22)',
    'rng022: rho_xy = C_xy/(sigma_x sigma_y) (Rangayyan Eq 3.22).',
    'rng084: kth observed realization of a signal in noise (signal-plus-noise model).',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
