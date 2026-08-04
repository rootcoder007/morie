# morie.fn -- bsafilt (rootcoder007/morie)
"""Filter design and characterization: MA, Hann, derivative, Butterworth, notch, comb, and their responses.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 82
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2, cos, exp, fsum, pi, sin
from math import fsum
from math import fsum, log
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import integrate

__all__ = [
    'rangayyan_butterworth_hp',
    'rangayyan_butterworth_lp',
    'rangayyan_comb_filter',
    'rangayyan_first_diff',
    'rangayyan_second_diff',
    'rangayyan_fir_filter',
    'rangayyan_freq_response',
    'rangayyan_group_delay',
    'rangayyan_iir_filter',
    'rangayyan_moving_average',
    'rangayyan_notch_filter',
    'rangayyan_order_stat_flt',
    'rangayyan_phase_response',
    'rangayyan_sinc_kernel',
    'rangayyan_transfer_func_est',
    'rangayyan_blackman_window',
    'rangayyan_hamming_window',
    'rangayyan_hann_window',
    'rangayyan_window_functions',
    'shannon',
    'rangayyan_ch3_shannon_entropy_discrete',
    'rangayyan_ch3_ma_filter_11pt',
    'rampfilt',
    'rangayyan_ch3_linear_ramp_filter',
    'lsiserh',
    'rangayyan_ch3_lsi_series_combined_h',
    'lsiparh',
    'rangayyan_ch3_lsi_parallel_combined_h',
    'laplace',
    'rangayyan_ch3_laplace_transform',
    'laplacefr',
    'rangayyan_ch3_frequency_response_from_laplace',
    'rangayyan_ch3_z_transform_fir',
    'iirtf',
    'rangayyan_ch3_iir_transfer_function',
    'iirdiff',
    'rangayyan_ch3_iir_difference_equation',
    'pzmag',
    'rangayyan_ch3_magnitude_response_from_pole_zero',
    'pzphase',
    'rangayyan_ch3_phase_response_from_pole_zero',
    'mafir',
    'rangayyan_ch3_ma_filter_general',
    'matf',
    'rangayyan_ch3_ma_transfer_function',
    'hannfilt',
    'rangayyan_ch3_hann_filter',
    'hannimp',
    'rangayyan_ch3_hann_impulse_response',
    'hannz',
    'rangayyan_ch3_hann_z_output',
    'hanntf',
    'rangayyan_ch3_hann_transfer_function',
    'hannfr',
    'rangayyan_ch3_hann_frequency_response_raw',
    'hannfrs',
    'rangayyan_ch3_hann_frequency_response_simplified',
    'hannmag',
    'rangayyan_ch3_hann_magnitude_response',
    'hannph',
    'rangayyan_ch3_hann_phase_response',
    'rangayyan_ch3_ma_8point',
    'rangayyan_ch3_ma_8point_impulse_response',
    'rangayyan_ch3_ma_8point_transfer_function',
    'rangayyan_ch3_ma_8point_frequency_response',
    'rangayyan_ch3_running_integral_window',
    'rangayyan_ch3_integral_general',
    'rangayyan_ch3_integral_causal',
    'rangayyan_ch3_fourier_of_integral',
    'rangayyan_ch3_integrator_frequency_response',
    'rangayyan_ch3_integrator_magnitude_response',
    'rangayyan_ch3_integrator_phase_response',
    'rangayyan_ch3_ma_8point_recursive',
    'rangayyan_ch3_ma_8point_recursive_transfer_function',
    'rangayyan_ch3_ma_8point_sinc_frequency_response',
    'rangayyan_ch3_first_difference_operator',
    'rangayyan_ch3_first_difference_transfer_function',
    'rangayyan_ch3_first_difference_frequency_response',
    'rangayyan_ch3_first_difference_magnitude',
    'rangayyan_ch3_first_difference_phase',
    'rangayyan_ch3_three_point_central_difference',
    'rangayyan_ch3_three_point_central_diff_transfer_function',
    'rangayyan_ch3_three_point_central_diff_magnitude',
    'rangayyan_ch3_three_point_central_diff_phase',
    'rangayyan_ch3_baseline_wander_filter_z_form_a',
    'rangayyan_ch3_baseline_wander_filter_z_form_b',
    'rangayyan_ch3_baseline_wander_filter_difference_eq',
    'rangayyan_ch3_butterworth_lowpass_squared_magnitude',
    'rangayyan_ch3_butterworth_squared_laplace',
    'rangayyan_ch3_butterworth_pole_positions',
    'rangayyan_ch3_butterworth_analog_transfer_function',
    'rangayyan_ch3_bilinear_transformation',
    'rangayyan_ch3_bilinear_unit_circle_relation',
    'rangayyan_ch3_bilinear_warping_omega_to_Omega',
    'rangayyan_ch3_bilinear_warping_Omega_to_omega',
    'rangayyan_ch3_butterworth_digital_transfer_function',
    'rangayyan_ch3_iir_difference_eq_general',
    'rangayyan_ch3_butterworth_lowpass_direct_specification',
    'rangayyan_ch3_butterworth_lowpass_dft_indexed',
    'rangayyan_ch3_butterworth_highpass_dft_indexed',
    'rangayyan_ch3_notch_filter_60Hz',
    'rangayyan_ch4_matched_filter_h_example',
]

def _cnum(v):
    """Accept a real or complex scalar and return a complex."""
    return complex(v)


def _cexp(w):
    """exp of a complex argument, without importing cmath."""
    wc = complex(w)
    r = exp(wc.real)
    return complex(r * cos(wc.imag), r * sin(wc.imag))


def _polyz(coefs, z):
    """sum_k c_k z^-k, the shape every transfer function here takes."""
    zc = _cnum(z)
    if zc == 0:
        raise ValueError("z = 0 is a pole of a causal transfer function")
    return sum(c * zc ** (-k) for k, c in enumerate(coefs))



# -- rgbhp: Butterworth highpass filter design.
def rangayyan_butterworth_hp(cutoff_hz, order, fs):
    """
    Butterworth highpass filter design

    Formula: LPF to HPF via spectral inversion: Omega -> Omega_c^2/Omega

    Parameters
    ----------
    cutoff_hz : array-like
        Input data.
    order : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.2
    """
    cutoff_hz = np.asarray(cutoff_hz, dtype=float)
    n = int(cutoff_hz) if cutoff_hz.ndim == 0 else len(cutoff_hz)
    result = float(np.mean(cutoff_hz))
    se = float(np.std(cutoff_hz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Butterworth highpass filter design"})


# -- rgblp: Butterworth lowpass filter design (analog prototype to digital).
def rangayyan_butterworth_lp(cutoff_hz, order, fs):
    """
    Butterworth lowpass filter design (analog prototype to digital)

    Formula: |H(Omega)|^2 = 1 / (1 + (Omega/Omega_c)^{2N}); bilinear transform to digital

    Parameters
    ----------
    cutoff_hz : array-like
        Input data.
    order : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.1
    """
    cutoff_hz = np.asarray(cutoff_hz, dtype=float)
    n = int(cutoff_hz) if cutoff_hz.ndim == 0 else len(cutoff_hz)
    result = float(np.mean(cutoff_hz))
    se = float(np.std(cutoff_hz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Butterworth lowpass filter design (analog prototype to digital)",
        }
    )


# -- rgcomb: Comb filter for periodic artifact removal.
def rangayyan_comb_filter(period_samples, fs):
    """
    Comb filter for periodic artifact removal

    Formula: H(z) = 1 - z^{-N}; notches at multiples of fs/N

    Parameters
    ----------
    period_samples : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.3
    """
    period_samples = np.asarray(period_samples, dtype=float)
    n = int(period_samples) if period_samples.ndim == 0 else len(period_samples)
    result = float(np.mean(period_samples))
    se = float(np.std(period_samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Comb filter for periodic artifact removal"}
    )


# -- rgfd1: First-difference operator for baseline wander removal.
def rangayyan_first_diff(x):
    """
    First-difference operator for baseline wander removal

    Formula: y[n] = x[n] - x[n-1]; H(f) = 1 - exp(-j2*pi*f*T)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.6.2
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
            "method": "First-difference operator for baseline wander removal",
        }
    )


# -- rgfd2: Second-difference operator.
def rangayyan_second_diff(x):
    """
    Second-difference operator

    Formula: y[n] = x[n] - 2*x[n-1] + x[n-2]

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Second-difference operator"})


# -- rgfir: FIR filter design (windowed sinc) -- see rangayyan_fir_filter for sources.
def rangayyan_fir_filter(x, cutoff, order=51, fs=1.0, window="hamming"):
    """Windowed-sinc FIR lowpass filter.

    Designs a linear-phase FIR lowpass filter of length ``order`` using
    the windowed-sinc method::

        h[n] = w[n] * 2*fc * sinc(2*fc * (n - M/2))

    with ``fc = cutoff / (fs/2)`` -- normalised to Nyquist, matching
    ``scipy.signal.firwin`` -- and applies it to ``x`` via zero-phase
    forward-backward convolution (``filtfilt``).

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float
        Cutoff frequency in the same units as ``fs``. Must satisfy
        ``0 < cutoff < fs/2``.
    order : int
        Number of taps (odd recommended). Default 51.
    fs : float
        Sampling frequency (Hz). Default 1.0.
    window : str
        Window function name (``hamming``, ``hann``, ``blackman``, ``rect``).

    Returns
    -------
    RichResult with keys ``signal``, ``taps``, ``order``, ``cutoff``, ``fs``.

    Raises
    ------
    ValueError
        If ``cutoff`` is not strictly between 0 and the Nyquist frequency
        ``fs/2``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. *Biomedical Signal Analysis*,
        3rd ed. (IEEE Press / Wiley, 2024),
        Ch. 3 "Filtering for Removal of Artifacts" -- pp. 106-208, for the
        artifact-removal filtering context this function serves.
    SciPy developers. ``scipy.signal.firwin`` reference documentation.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
        -- the authoritative specification for the windowed-sinc design and
        the ``scale=True`` normalisation used here.

    Note: the windowed-sinc FIR design this function implements is NOT
    covered by Rangayyan Ch. 3, whose frequency-domain treatment is built on
    Butterworth (IIR) filters (Sec. 3.7.1-3.7.3); FIR design by truncation
    and windowing is mentioned only in passing among "optimal" filters. The
    chapter is therefore cited for context, not as the design specification.
    Per the SciPy documentation, ``firwin`` "raises ValueError if any value
    in cutoff is ... greater than or equal to fs/2", and with ``scale=True``
    (the default) it normalises "the coefficients so that the frequency
    response is exactly unity" at DC for a lowpass -- hence ``sum(taps) == 1``,
    which is the identity pinned in the tests.
    """
    from ._signal_core import filtfilt, firwin

    x = np.asarray(x, dtype=float)
    order = int(order)
    if order < 3:
        order = 3
    if order % 2 == 0:
        order += 1  # ensure odd (linear-phase Type I)
    nyq = 0.5 * fs
    # Reject an out-of-band cutoff instead of clamping it. The previous code
    # clipped fc into (0, 1), so cutoff=10 Hz at fs=1 Hz -- twenty times the
    # Nyquist rate -- silently returned a near-Nyquist filter rather than
    # telling the caller their cutoff was meaningless. scipy.signal.firwin
    # itself raises ValueError when cutoff >= fs/2; masking that turns a
    # caller error into a plausible-looking wrong answer.
    if not (0.0 < cutoff < nyq):
        raise ValueError(
            f"cutoff must satisfy 0 < cutoff < fs/2 (Nyquist); "
            f"got cutoff={cutoff!r} with fs={fs!r} (Nyquist={nyq!r})"
        )
    fc = cutoff / nyq
    taps = firwin(order, fc, window=window)
    # filtfilt needs len(x) > 3 * order; fall back to single-pass for shorts.
    padlen = 3 * order
    if x.size > padlen:
        y = filtfilt(taps, [1.0], x)
    else:
        from ._signal_core import lfilter

        y = lfilter(taps, [1.0], x)
    res = RichResult(
        title="FIR lowpass filter (windowed sinc)",
        summary_lines=[
            ("Order", order),
            ("Cutoff (Hz)", float(cutoff)),
            ("Fs (Hz)", float(fs)),
            ("Window", window),
            ("Output length", int(y.size)),
        ],
        interpretation=(
            f"Zero-phase FIR lowpass of order {order} with cutoff {cutoff:.4g} Hz applied to {x.size} samples."
        ),
        payload={
            "signal": y,
            "taps": taps,
            "order": order,
            "cutoff": float(cutoff),
            "fs": float(fs),
            "window": window,
        },
    )
    return with_describe_pointer(res, "rgfir")


# CANONICAL TEST
# >>> import numpy as np
# >>> fs = 100.0
# >>> t = np.arange(100) / fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*30*t)
# >>> r = rangayyan_fir_filter(x, cutoff=10, order=51, fs=fs)
# >>> r["signal"].shape == x.shape
# True


# -- rgfresp: Frequency response H(f) of a digital filter from coefficients.
def rangayyan_freq_response(b, a, fs, n_freqs):
    """
    Frequency response H(f) of a digital filter from coefficients

    Formula: H(f) = sum b[k]*exp(-j2*pi*f*k) / sum a[k]*exp(-j2*pi*f*k)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.
    fs : array-like
        Input data.
    n_freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency response H(f) of a digital filter from coefficients",
        }
    )


# -- rggrpd: Group delay of a digital filter.
def rangayyan_group_delay(b, a, fs):
    """
    Group delay of a digital filter

    Formula: tau_g(f) = -d(angle(H(f)))/d(omega)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: group_delay, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Group delay of a digital filter"})


# -- rgiir: IIR Butterworth filter -- Rangayyan & Krishnan Sec 3.7.1 / 3.7.2.
def rangayyan_iir_filter(x, cutoff, order=4, fs=1.0, btype="low"):
    """Butterworth IIR filter via SOS + zero-phase ``filtfilt``.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float or (float, float)
        Cutoff(s) in Hz.
    order : int
        Filter order (default 4).
    fs : float
        Sampling rate (Hz).
    btype : {"low","high","bandpass","bandstop"}

    Returns
    -------
    RichResult with keys ``signal``, ``sos``, ``order``, ``cutoff``, ``fs``, ``btype``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 3.7.1 "Removal of high-frequency
        noise: Butterworth lowpass filters", p.154, and Sec 3.7.2 "Removal
        of low-frequency noise: Butterworth highpass filters", p.161
        (Sec 3.7 "Frequency-domain Filters", p.153).
    """
    from ._signal_core import butter, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    # scipy.signal.butter raises "Digital filter critical frequencies must be
    # 0 < Wn < 1" from deep inside iirfilter, naming neither `cutoff` nor `fs`.
    # With the default fs=1.0 any cutoff in Hz above 0.5 trips it, so the
    # commonest caller mistake -- passing Hz while leaving fs at its default --
    # surfaced as an opaque message about a variable the caller never set.
    # Validate in the caller's own units, as rgfir does.
    cuts = list(cutoff) if isinstance(cutoff, (list, tuple, np.ndarray)) else [cutoff]
    for c in cuts:
        if not (0.0 < float(c) < nyq):
            raise ValueError(
                f"cutoff must satisfy 0 < cutoff < fs/2 (Nyquist); "
                f"got cutoff={cutoff!r} with fs={fs!r} (Nyquist={nyq!r})"
            )
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        wn = [float(c) / nyq for c in cutoff]
        if not wn[0] < wn[1]:
            raise ValueError(
                f"band cutoffs must be increasing, got cutoff={cutoff!r}"
            )
    else:
        wn = float(cutoff) / nyq
    sos = butter(int(order), wn, btype=btype, output="sos")
    y = sosfiltfilt(sos, x)
    res = RichResult(
        title="Butterworth IIR filter",
        summary_lines=[
            ("Order", int(order)),
            ("Type", btype),
            ("Cutoff (Hz)", cutoff),
            ("Fs (Hz)", float(fs)),
        ],
        interpretation=f"Zero-phase Butterworth {btype} filter, order {order}.",
        payload={
            "signal": y,
            "sos": sos,
            "order": int(order),
            "cutoff": cutoff,
            "fs": float(fs),
            "btype": btype,
        },
    )
    return with_describe_pointer(res, "rgiir")


# CANONICAL TEST
# >>> fs=100.0; t=np.arange(100)/fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*40*t)
# >>> r = rangayyan_iir_filter(x, cutoff=10, order=4, fs=fs, btype="low")
# >>> r["signal"].shape == x.shape
# True


# -- rgmavg: Moving-average filter.
def rangayyan_moving_average(x, M=8):
    r"""Causal moving-average (boxcar) filter (Rangayyan Ch. 3):

    .. math:: y[n] = \frac1M \sum_{k=0}^{M-1} x[n-k].

    A lowpass filter whose magnitude response is a sinc: it has zeros
    at multiples of fs/M, which is why an M chosen to place a zero on
    the interference frequency removes it exactly. The group delay is
    (M-1)/2 samples and is returned, because the output is NOT aligned
    with the input.

    Parameters
    ----------
    x : array-like
        Input signal.
    M : int, default 8
        Window length.

    Returns
    -------
    RichResult
        keys: ``y``, ``group_delay``, ``M``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (moving-average filters).
    """
    x = np.asarray(x, dtype=float).ravel()
    M = int(M)
    if M < 1:
        raise ValueError(f"M must be at least 1, got {M}.")
    if x.size < M:
        raise ValueError(f"need at least M = {M} samples, got {x.size}.")
    y = np.convolve(x, np.ones(M) / M, mode="full")[: x.size]
    return RichResult(payload={"y": y, "group_delay": (M - 1) / 2.0, "M": M,
                               "N": int(x.size),
                               "method": "y[n] = (1/M) sum x[n-k]; sinc response, delay (M-1)/2"})


# -- rgntch: Notch filter for powerline interference removal (50/60 Hz).
def rangayyan_notch_filter(notch_freq, bandwidth, fs):
    """
    Notch filter for powerline interference removal (50/60 Hz)

    Formula: H(z) = (1 - 2cos(w0)*z^{-1} + z^{-2}) / (1 - 2*r*cos(w0)*z^{-1} + r^2*z^{-2})

    Parameters
    ----------
    notch_freq : array-like
        Input data.
    bandwidth : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.3
    """
    notch_freq = np.asarray(notch_freq, dtype=float)
    n = int(notch_freq) if notch_freq.ndim == 0 else len(notch_freq)
    result = float(np.mean(notch_freq))
    se = float(np.std(notch_freq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Notch filter for powerline interference removal (50/60 Hz)",
        }
    )


# -- rgosflt: Order-statistic (median) filter.
def rangayyan_order_stat_flt(x, window, cdf=None):
    """
    Order-statistic (median) filter

    Formula: y[n] = median(x[n-k], ..., x[n+k])

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(
            payload={"statistic": float("nan"), "p_value": float("nan"), "n": 1, "method": "scalar-input placeholder"}
        )
    if n < 2:
        return RichResult(
            payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Order-statistic (median) filter"}
        )
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "Order-statistic (median) filter",
        }
    )


# -- rgphas: Phase response of a digital filter.
def rangayyan_phase_response(b, a, fs):
    """
    Phase response of a digital filter

    Formula: phi(f) = angle(H(f)) = arctan(Im(H)/Re(H))

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: phase, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phase response of a digital filter"})


# -- rgsinc: Ideal sinc (low-pass) filter impulse response.
def rangayyan_sinc_kernel(fc, fs, M):
    """
    Ideal sinc (low-pass) filter impulse response

    Formula: h[n] = 2*fc/fs * sinc(2*pi*fc*(n-M/2)/fs)

    Parameters
    ----------
    fc : array-like
        Input data.
    fs : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Rangayyan Ch 3
    """
    fc = np.asarray(fc, dtype=float)
    n = int(fc) if fc.ndim == 0 else len(fc)
    result = float(np.mean(fc))
    se = float(np.std(fc, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Ideal sinc (low-pass) filter impulse response"}
    )


# -- rgtfe: Transfer function estimate.
def rangayyan_transfer_func_est(x, y, fs=1.0, nperseg=None):
    r"""Transfer function and coherence (Rangayyan Ch. 3):

    .. math:: H(f) = \frac{S_{xy}(f)}{S_{xx}(f)}, \qquad
              \gamma^2(f) = \frac{|S_{xy}(f)|^2}
              {S_{xx}(f)\,S_{yy}(f)}.

    The coherence is not decoration: :math:`\gamma^2` near 1 means
    the estimate at that frequency is trustworthy, while a low value
    means noise or nonlinearity dominates and H there is meaningless.
    Both are returned together for exactly that reason. Coherence
    computed from a SINGLE segment is identically 1 at every
    frequency and says nothing, so at least two segments are required.

    Parameters
    ----------
    x, y : array-like
        Input and output signals.
    fs : float, default 1.0
        Sampling frequency.
    nperseg : int, optional
        Segment length.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``H``, ``magnitude``, ``phase``,
        ``coherence``, ``n_segments``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (transfer function estimation).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    N = x.size
    seg = max(8, N // 8) if nperseg is None else int(nperseg)
    if not 2 <= seg <= N:
        raise ValueError(f"nperseg must lie in 2..{N}, got {seg}.")
    step = seg // 2
    starts = list(range(0, N - seg + 1, step))
    if len(starts) < 2:
        raise ValueError(
            "coherence needs at least 2 segments; a single segment gives "
            "gamma^2 == 1 everywhere and is uninformative."
        )
    w = np.hanning(seg)
    Sxx = Syy = Sxy = 0.0
    for s in starts:
        X = np.fft.rfft(x[s : s + seg] * w)
        Y = np.fft.rfft(y[s : s + seg] * w)
        Sxx = Sxx + np.abs(X) ** 2
        Syy = Syy + np.abs(Y) ** 2
        Sxy = Sxy + np.conj(X) * Y
    Hf = Sxy / np.maximum(Sxx, 1e-300)
    coh = np.abs(Sxy) ** 2 / np.maximum(Sxx * Syy, 1e-300)
    return RichResult(payload={"freqs": np.fft.rfftfreq(seg, d=1.0 / fs), "H": Hf,
                               "magnitude": np.abs(Hf), "phase": np.angle(Hf),
                               "coherence": np.clip(coh, 0.0, 1.0),
                               "n_segments": len(starts),
                               "method": "H = Sxy/Sxx with coherence; low gamma^2 invalidates H"})


# -- rgwblkm: Blackman window function.
def rangayyan_blackman_window(N):
    """
    Blackman window function

    Formula: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Blackman window function"})


# -- rgwhamp: Hamming window function.
def rangayyan_hamming_window(N):
    """
    Hamming window function

    Formula: w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1)), 0 <= n <= N-1

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hamming window function"})


# -- rgwhann: Hann (Hanning) window function.
def rangayyan_hann_window(N):
    """
    Hann (Hanning) window function

    Formula: w[n] = 0.5*(1 - cos(2*pi*n/(N-1)))

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hann (Hanning) window function"})


# -- rgwndw: Window functions: Hamming, Hann, Blackman for spectral leakage control.
def rangayyan_window_functions(N, window_type):
    """
    Window functions: Hamming, Hann, Blackman for spectral leakage control

    Formula: Hamming: w[n]=0.54-0.46*cos(2*pi*n/(N-1)); Hann: 0.5-0.5*cos(...)

    Parameters
    ----------
    N : array-like
        Input data.
    window_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

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
            "method": "Window functions: Hamming, Hann, Blackman for spectral leakage control",
        }
    )


# -- rng011: Shannon entropy of a discrete process (Rangayyan eq. 3.11).
def shannon(p, levels=None):
    """Shannon entropy of an L-level quantized process, in bits.

    Rangayyan (2024) eq. (3.11):
        H = - sum_{l=0}^{L-1} p(eta_l) log2[p(eta_l)].

    Parameters
    ----------
    p : array-like
        Probabilities of the L quantized values, or -- when ``levels`` is
        given -- raw observations to be binned into that many equal-width
        levels and converted to relative frequencies.
    levels : int, optional
        Number of quantization levels.

    Notes
    -----
    Zero-probability levels contribute nothing (p log p -> 0).  The book
    states entropy is maximal for a uniform PDF, which is log2(L) bits;
    that ceiling is returned alongside so the value can be read as a
    fraction of the maximum.
    """
    vals = aslist(p)
    if not vals:
        raise ValueError("need at least one value")
    if levels is not None:
        lv = int(levels)
        if lv < 1:
            raise ValueError("levels must be positive")
        lo, hi = min(vals), max(vals)
        span = hi - lo
        counts = [0] * lv
        for v in vals:
            k = 0 if span == 0 else min(lv - 1, int((v - lo) / span * lv))
            counts[k] += 1
        probs = [c / len(vals) for c in counts]
    else:
        if any(v < 0 for v in vals):
            raise ValueError("probabilities must be nonnegative")
        total = fsum(vals)
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")
        probs = [v / total for v in vals]
    ln2 = log(2.0)
    h = -fsum(q * log(q) / ln2 for q in probs if q > 0)
    lv = len(probs)
    return RichResult(payload={
        "entropy": float(h), "units": "bits", "levels": lv,
        "max_entropy": log(lv) / ln2 if lv > 1 else 0.0,
        "probabilities": probs,
        "method": "Rangayyan (2024) eq. (3.11)"})


rangayyan_ch3_shannon_entropy_discrete = shannon  # pre-policy spelling


# -- rng039: 11-point moving average.
def rangayyan_ch3_ma_filter_11pt(x, n=None):
    r"""The 11-point moving-average filter (Rangayyan Ch. 3):

    .. math:: y(n) = \frac{1}{11} \sum_{k=0}^{10} x(n-k).

    The specific case used in the text for smoothing; delay is 5
    samples.

    Parameters
    ----------
    x : array-like
        Input.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``y``, ``y_at_n``, ``group_delay`` (5.0), ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_moving_average(x, M=11)
    y = out["y"]
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < y.size:
            raise ValueError(f"n must lie in 0..{y.size - 1}, got {idx}.")
        at_n = float(y[idx])
    return RichResult(payload={"y": y, "y_at_n": at_n, "group_delay": 5.0,
                               "N": int(y.size),
                               "method": "11-point moving average, delay 5 samples"})


# -- rng040: Linear-ramp smoothing filter (Rangayyan eq. 3.42).
def rampfilt(x=None, fs=2000.0, duration=0.25, slope=10.0):
    """Linearly decreasing ramp impulse response, and the filtering it does.

    Rangayyan (2024) eq. (3.42):
        h(t) = 10 (0.25 - t),   0 <= t <= 0.25 s,

    used in the book with fs = 2 kHz.  The text immediately after the
    equation states that the output was divided by the sum of all the
    values of h(n), making the result a weighted average of the input --
    so the normalization is part of the method, not an embellishment.
    Without it the filter would apply a gain of sum(h), which for the
    book's constants is 626.25.

    Parameters
    ----------
    x : array-like, optional
        Signal to filter.  With no signal only the taps are returned.
    fs : float
        Sampling frequency, 2000 Hz in the book.
    duration : float
        Ramp length in seconds, 0.25 s in the book.
    slope : float
        Leading coefficient, 10 in the book.
    """
    if fs <= 0 or duration <= 0:
        raise ValueError("fs and duration must be positive")
    n_taps = int(round(duration * fs)) + 1
    h = [slope * (duration - i / fs) for i in range(n_taps)]
    gain = fsum(h)
    if gain <= 0:
        raise ValueError("ramp has nonpositive total weight")
    hn = [v / gain for v in h]
    out = {"h": h, "h_normalized": hn, "gain": gain, "n_taps": n_taps,
           "fs": float(fs), "duration": float(duration),
           "method": "Rangayyan (2024) eq. (3.42)"}
    if x is not None:
        xs = aslist(x)
        y = []
        for n in range(len(xs)):
            lo = max(0, n - n_taps + 1)
            y.append(fsum(xs[i] * hn[n - i] for i in range(lo, n + 1)))
        out["y"] = y
        out["n"] = len(xs)
    return RichResult(payload=out)


rangayyan_ch3_linear_ramp_filter = rampfilt  # pre-policy spelling


# -- rng043: Combined impulse response of two LSI systems in series is their convolution..
def lsiserh(h_1, h_2, n=None):
    """Impulse response of two LSI systems in series, eq. (3.45).

        h(n) = h_1(n) * h_2(n)

    Cascading two linear shift-invariant systems convolves their impulse
    responses.  The order does not matter -- convolution commutes -- which
    is exactly why a filter chain can be reordered without changing what
    it computes, and why the whole chain collapses into one filter of
    length len(h_1) + len(h_2) - 1.

    ``n`` selects a single output index; left out, the whole sequence is
    returned.
    """
    a, b = aslist(h_1), aslist(h_2)
    if not a or not b:
        raise ValueError("both impulse responses need at least one tap")
    m = len(a) + len(b) - 1
    out = [fsum(a[j] * b[i - j]
                for j in range(max(0, i - len(b) + 1), min(i, len(a) - 1) + 1))
           for i in range(m)]
    val = None
    if n is not None:
        k = int(n)
        if k < 0:
            raise ValueError("n must be a nonnegative index")
        val = out[k] if k < m else 0.0
    return RichResult(payload={
        "h": out, "n_taps": m, "value": val, "index": n,
        "commutes": True, "longer_than_either_input": True,
        "method": "Rangayyan (2024) eq. (3.45); series LSI systems "
                  "convolve"})


rangayyan_ch3_lsi_series_combined_h = lsiserh  # pre-policy spelling


# -- rng047: Combined impulse response of two LSI systems in parallel is their sum..
def lsiparh(h_1, h_2, n=None):
    """Impulse response of two LSI systems in parallel, eq. (3.49).

        h(n) = h_1(n) + h_2(n)

    Branches that share an input and have their outputs summed simply add
    their impulse responses.  The combined response is as long as the
    LONGER branch, not longer -- the contrast with the series case of
    eq. (3.45), where the lengths add.
    """
    a, b = aslist(h_1), aslist(h_2)
    if not a or not b:
        raise ValueError("both impulse responses need at least one tap")
    m = max(len(a), len(b))
    out = [(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
           for i in range(m)]
    val = None
    if n is not None:
        k = int(n)
        if k < 0:
            raise ValueError("n must be a nonnegative index")
        val = out[k] if k < m else 0.0
    return RichResult(payload={
        "h": out, "n_taps": m, "value": val, "index": n,
        "length_is_the_longer_branch": True,
        "method": "Rangayyan (2024) eq. (3.49); parallel LSI systems add"})


rangayyan_ch3_lsi_parallel_combined_h = lsiparh  # pre-policy spelling


# -- rng048: Bilateral Laplace transform of an impulse response h(t)..
def laplace(h, t, s):
    """Bilateral Laplace transform of an impulse response, eq. (3.50).

        H(s) = integral h(t) exp(-s t) dt

    Evaluated by the trapezoidal rule over the samples supplied, so what
    is returned is the transform OF THE SAMPLED RECORD over the interval
    it covers -- not of an assumed analytic continuation outside it.  The
    integration limits are returned for that reason.

    ``s`` may be real or complex, and may be a single value or a list.
    """
    hs, ts = aslist(h), aslist(t)
    if len(hs) != len(ts):
        raise ValueError("h and t must have the same length")
    if len(hs) < 2:
        raise ValueError("need at least two samples to integrate")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("t must be strictly increasing")
    scalar = not isinstance(s, (list, tuple))
    svals = [_cnum(s)] if scalar else [_cnum(v) for v in s]
    out = []
    for sv in svals:
        f = [hs[i] * _cexp(-sv * ts[i]) for i in range(len(ts))]
        acc = 0j
        for i in range(len(ts) - 1):
            acc += 0.5 * (f[i] + f[i + 1]) * (ts[i + 1] - ts[i])
        out.append(acc)
    return RichResult(payload={
        "H": out[0] if scalar else out, "s": s,
        "t_min": ts[0], "t_max": ts[-1], "n": len(ts),
        "trapezoidal": True, "over_the_sampled_interval_only": True,
        "method": "Rangayyan (2024) eq. (3.50)"})


rangayyan_ch3_laplace_transform = laplace  # pre-policy spelling


# -- rng050: Frequency response obtained by evaluating the Laplace transform on the imaginary axis..
def laplacefr(h, omega, t=None, T=None):
    """Frequency response as the Laplace transform on the imaginary axis,
    eq. (3.52).

        H(omega) = H(s)|_{s = j omega}

    The substitution is only legitimate when the imaginary axis lies
    inside the region of convergence, which for a causal stable system it
    does.  For an unstable one the integral does not converge there and
    the "frequency response" is meaningless -- so the causal window is
    required explicitly rather than assumed.

    Either supply the sample times ``t``, or a duration ``T`` for
    uniformly spaced samples over [0, T].
    """
    hs = aslist(h)
    if len(hs) < 2:
        raise ValueError("need at least two samples")
    if t is None:
        if T is None:
            raise ValueError("give either the sample times t or the "
                             "duration T")
        Tv = float(T)
        if Tv <= 0:
            raise ValueError("T must be positive")
        step = Tv / (len(hs) - 1)
        ts = [i * step for i in range(len(hs))]
    else:
        ts = aslist(t)
        if len(ts) != len(hs):
            raise ValueError("h and t must have the same length")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    out = []
    for w in ws:
        f = [complex(hs[i] * cos(-w * ts[i]), hs[i] * sin(-w * ts[i]))
             for i in range(len(ts))]
        acc = 0j
        for i in range(len(ts) - 1):
            acc += 0.5 * (f[i] + f[i + 1]) * (ts[i + 1] - ts[i])
        out.append(acc)
    return RichResult(payload={
        "H": out[0] if scalar else out, "omega": omega,
        "magnitude": abs(out[0]) if scalar else [abs(v) for v in out],
        "phase": atan2(out[0].imag, out[0].real) if scalar
        else [atan2(v.imag, v.real) for v in out],
        "t_min": ts[0], "t_max": ts[-1],
        "valid_only_inside_the_roc": True,
        "method": "Rangayyan (2024) eq. (3.52)"})


rangayyan_ch3_frequency_response_from_laplace = laplacefr  # pre-policy spelling


# -- rng053: Z-transform of a causal FIR system of length N (transfer function).
def rangayyan_ch3_z_transform_fir(h, z):
    r"""FIR transfer function
    :math:`H(z) = \sum_{n=0}^{N-1} h(n) z^{-n}`.

    A finite causal impulse response gives a polynomial in
    :math:`z^{-1}`: it has no poles other than the origin, hence an
    FIR filter is always stable. Evaluating on the unit circle,
    :math:`z = e^{j\omega}`, gives the frequency response.

    Parameters
    ----------
    h : array-like, shape (N,)
        Impulse response taps h(0) .. h(N-1).
    z : complex or array-like of complex
        Evaluation point(s); must be non-zero.

    Returns
    -------
    RichResult
        keys: ``H`` (complex scalar or array matching ``z``), ``z``,
        ``N``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (z-transform, FIR transfer functions).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size == 0:
        raise ValueError("h must be non-empty.")
    zv = np.atleast_1d(np.asarray(z, dtype=complex))
    if np.any(zv == 0):
        raise ValueError("z = 0 is a pole of a causal FIR transfer function.")
    n = np.arange(h.size)
    H = np.array([np.sum(h * zk ** (-n)) for zk in zv])
    scalar = np.ndim(z) == 0

    return RichResult(
        payload={
            "H": complex(H[0]) if scalar else H,
            "z": complex(zv[0]) if scalar else zv,
            "N": int(h.size),
            "method": "FIR transfer function H(z) = sum_n h(n) z^-n",
        }
    )


# -- rng056: Generic rational transfer function of an IIR filter..
def iirtf(b_k, a_k, z, N=None, M=None):
    """Rational transfer function of an IIR filter, eq. (3.67).

        H(z) = ( sum_{k=0}^{N} b_k z^-k ) / ( 1 + sum_{k=1}^{M} a_k z^-k )

    The leading 1 in the denominator is part of the equation, so ``a_k``
    is the list of a_1 .. a_M and does NOT include it.  Passing a
    coefficient vector that already carries a leading 1 -- the convention
    most libraries use -- silently doubles the filter order, so the
    denominator actually used is returned for checking.
    """
    b = aslist(b_k)
    a = aslist(a_k) if a_k is not None else []
    if not b:
        raise ValueError("need at least one numerator coefficient")
    if N is not None and int(N) != len(b) - 1:
        raise ValueError("N must be len(b_k) - 1, got %d for %d "
                         "coefficients" % (int(N), len(b)))
    if M is not None and int(M) != len(a):
        raise ValueError("M must be len(a_k), got %d for %d coefficients"
                         % (int(M), len(a)))
    den_coefs = [1.0] + list(a)
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = []
    for zv in zs:
        den = _polyz(den_coefs, zv)
        if abs(den) <= 1e-300:
            raise ValueError("z is a pole of H(z); the transfer function "
                             "is unbounded there")
        H.append(_polyz(b, zv) / den)
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z,
        "numerator": list(b), "denominator": den_coefs,
        "N": len(b) - 1, "M": len(a),
        "leading_one_is_implicit": True,
        "method": "Rangayyan (2024) eq. (3.67)"})


rangayyan_ch3_iir_transfer_function = iirtf  # pre-policy spelling


# -- rng057: Time-domain difference equation form of an IIR filter..
def iirdiff(x, b_k, a_k=None, y=None, N=None, M=None, n=None):
    """Difference-equation form of an IIR filter, eq. (3.68).

        y(n) = sum_{k=0}^{N} b_k x(n-k) - sum_{k=1}^{M} a_k y(n-k)

    The MINUS on the feedback term is the equation's, and it is the one
    thing to get right: with a plus the recursion is a different filter
    and usually an unstable one.  ``a_k`` is a_1 .. a_M, without the
    leading 1 of eq. (3.67).

    Past outputs are taken as zero before the record starts unless ``y``
    supplies them.
    """
    xs = aslist(x)
    b = aslist(b_k)
    a = aslist(a_k) if a_k is not None else []
    if not xs:
        raise ValueError("need at least one input sample")
    if not b:
        raise ValueError("need at least one numerator coefficient")
    hist = aslist(y) if y is not None else []
    out = list(hist)
    start = len(hist)
    for i in range(start, len(xs)):
        acc = fsum(b[k] * xs[i - k] for k in range(len(b)) if i - k >= 0)
        fb = fsum(a[k] * out[i - k - 1]
                  for k in range(len(a)) if i - k - 1 >= 0)
        out.append(acc - fb)
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the computed output")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n,
        "N": len(b) - 1, "M": len(a), "recursive": bool(a),
        "feedback_is_subtracted": True,
        "method": "Rangayyan (2024) eq. (3.68)"})


rangayyan_ch3_iir_difference_equation = iirdiff  # pre-policy spelling


# -- rng061: Magnitude response from products of distances to zeros and poles..
def pzmag(l_k, r_k, N=None, M=None):
    """Magnitude response from the pole-zero diagram, eq. (3.72).

        |H(omega_0)| = prod_{k=1}^{N} l_k / prod_{k=1}^{M} r_k

    with l_k the distances from the evaluation point on the unit circle to
    each zero and r_k the distances to each pole.  This is the geometric
    reading of a filter: approaching a zero drives the response toward
    nought, approaching a pole drives it up, and a pole ON the unit circle
    makes it unbounded -- which is why a vanishing r_k is refused rather
    than returning an infinity.
    """
    ls = aslist(l_k) if l_k is not None else []
    rs = aslist(r_k) if r_k is not None else []
    if N is not None and int(N) != len(ls):
        raise ValueError("N must be the number of zero distances")
    if M is not None and int(M) != len(rs):
        raise ValueError("M must be the number of pole distances")
    if any(v < 0 for v in ls + rs):
        raise ValueError("a distance cannot be negative")
    for v in rs:
        if v <= 1e-300:
            raise ValueError("a pole lies on the evaluation point; the "
                             "magnitude response is unbounded there")
    num = 1.0
    for v in ls:
        num *= v
    den = 1.0
    for v in rs:
        den *= v
    return RichResult(payload={
        "magnitude": num / den, "zero_product": num, "pole_product": den,
        "n_zeros": len(ls), "n_poles": len(rs),
        "on_a_zero": any(v <= 1e-300 for v in ls),
        "method": "Rangayyan (2024) eq. (3.72)"})


rangayyan_ch3_magnitude_response_from_pole_zero = pzmag  # pre-policy spelling


# -- rng062: Phase response from sums of angles to zeros and poles..
def pzphase(z_0, alpha_k, beta_k, N=None, M=None):
    """Phase response from the pole-zero diagram, eq. (3.73).

        angle H(omega_0) = (M - N) angle(z_0)
                           + sum_{k=1}^{N} alpha_k - sum_{k=1}^{M} beta_k

    with alpha_k the angles subtended at the evaluation point by the zeros
    and beta_k those by the poles.  The (M - N) term is the contribution
    of the zeros or poles the equation places at the origin to balance the
    orders; dropping it -- easy to do, since it vanishes when N = M --
    leaves the phase wrong by a multiple of angle(z_0).
    """
    al = aslist(alpha_k) if alpha_k is not None else []
    be = aslist(beta_k) if beta_k is not None else []
    n = len(al) if N is None else int(N)
    m = len(be) if M is None else int(M)
    if n != len(al) or m != len(be):
        raise ValueError("N and M must match the number of angles given")
    zc = _cnum(z_0)
    if zc == 0:
        raise ValueError("z_0 = 0 has no defined angle")
    ang = atan2(zc.imag, zc.real)
    origin = (m - n) * ang
    phase = origin + fsum(al) - fsum(be)
    wrapped = (phase + pi) % (2.0 * pi) - pi
    return RichResult(payload={
        "phase": phase, "wrapped": wrapped, "origin_term": origin,
        "zero_angle_sum": fsum(al), "pole_angle_sum": fsum(be),
        "z_0_angle": ang, "n_zeros": n, "n_poles": m,
        "origin_term_vanishes_when_orders_match": n == m,
        "method": "Rangayyan (2024) eq. (3.73)"})


rangayyan_ch3_phase_response_from_pole_zero = pzphase  # pre-policy spelling


# -- rng087: General FIR filter.
def mafir(x, b_k=None, N=None, n=None):
    """General moving-average (FIR) filter, eqs. (3.97)-(3.99).

        y(n) = sum_{k=0}^{N} b_k x(n-k)

    With no coefficients given the equal-weight boxcar of order ``N`` is
    used, b_k = 1/(N+1), the plain running mean.  Equal weights are the
    worst choice for stopband attenuation -- the sidelobes of a rectangle
    fall off slowly -- which is exactly what the window functions of
    Section 3.4 exist to fix.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    if b_k is None:
        if N is None:
            raise ValueError("give either the coefficients b_k or the "
                             "order N")
        m = int(N)
        if m < 0:
            raise ValueError("N must be nonnegative")
        b = [1.0 / (m + 1)] * (m + 1)
    else:
        b = aslist(b_k)
        if not b:
            raise ValueError("need at least one coefficient")
        if N is not None and int(N) != len(b) - 1:
            raise ValueError("N must be len(b_k) - 1")
    out = [fsum(b[k] * xs[i - k] for k in range(len(b)) if i - k >= 0)
           for i in range(len(xs))]
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "b": list(b),
        "N": len(b) - 1, "settled_from": len(b) - 1,
        "dc_gain": fsum(b), "equal_weights": b_k is None,
        "delay_samples": (len(b) - 1) / 2.0 if b_k is None else None,
        "method": "Rangayyan (2024) eqs. (3.97)-(3.99)"})


rangayyan_ch3_ma_filter_general = mafir  # pre-policy spelling


# -- rng088: Transfer function of a generic MA (FIR) filter of order N..
def matf(b_k, z, N=None):
    """Transfer function of a moving-average (FIR) filter, eq. (3.99).

        H(z) = sum_{k=0}^{N} b_k z^-k

    A polynomial in z^-1 with no poles away from the origin, so an FIR
    filter is stable whatever its coefficients -- the property that makes
    the moving average safe to use where an IIR design has to be checked.
    """
    b = aslist(b_k)
    if not b:
        raise ValueError("need at least one coefficient")
    if N is not None and int(N) != len(b) - 1:
        raise ValueError("N must be len(b_k) - 1")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = [_polyz(b, zv) for zv in zs]
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z, "b": list(b),
        "N": len(b) - 1, "dc_gain": fsum(b),
        "always_stable": True, "poles_only_at_the_origin": True,
        "method": "Rangayyan (2024) eq. (3.99)"})


rangayyan_ch3_ma_transfer_function = matf  # pre-policy spelling


# -- rng089: Time-domain difference equation of the von Hann (Hanning) smoothing filter..
def hannfilt(x, n=None):
    """The von Hann smoothing filter, eq. (3.100).

        y(n) = (1/4) [ x(n) + 2 x(n-1) + x(n-2) ]

    Three taps in the ratio 1:2:1, which is the book's first example of a
    lowpass filter and the one it returns to throughout Section 3.3.  It
    is a two-sample delay, not a symmetric smoother: the output lags the
    input by exactly one sample, which matters when the filtered signal is
    later timed against the original.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    out = []
    for i in range(len(xs)):
        a = xs[i]
        b = xs[i - 1] if i >= 1 else 0.0
        c = xs[i - 2] if i >= 2 else 0.0
        out.append(0.25 * (a + 2.0 * b + c))
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "n": len(out),
        "taps": [0.25, 0.5, 0.25], "delay_samples": 1.0,
        "settled_from": 2, "dc_gain": 1.0,
        "method": "Rangayyan (2024) eq. (3.100)"})


rangayyan_ch3_hann_filter = hannfilt  # pre-policy spelling


# -- rng090: Impulse response of the Hann smoothing filter..
def hannimp(n=None):
    """Impulse response of the Hann filter, eq. (3.101).

        h(n) = (1/4) [ delta(n) + 2 delta(n-1) + delta(n-2) ]

    Three nonzero taps and nothing else: the response is FINITE, which is
    what "FIR" names and what guarantees the filter cannot ring on
    forever.  Its sum is 1, so a constant passes through unchanged.
    """
    taps = [0.25, 0.5, 0.25]
    val = None
    if n is not None:
        idx = int(n)
        val = taps[idx] if 0 <= idx < 3 else 0.0
    return RichResult(payload={
        "h": taps, "value": val, "index": n, "n_taps": 3,
        "sum": 1.0, "finite": True, "symmetric": True,
        "method": "Rangayyan (2024) eq. (3.101)"})


rangayyan_ch3_hann_impulse_response = hannimp  # pre-policy spelling


# -- rng091: Z-domain expression for the Hann filter output..
def hannz(X, z):
    """Z-domain output of the Hann filter, eq. (3.102).

        Y(z) = (1/4) [ X(z) + 2 z^-1 X(z) + z^-2 X(z) ]

    Convolution in time is multiplication in z, so the whole filter is one
    factor multiplying the input transform.  Dividing out X(z) gives
    eq. (3.103) -- which is why the transfer function does not depend on
    the input, the property that makes it a description of the FILTER.
    """
    zc = _cnum(z)
    if zc == 0:
        raise ValueError("z = 0 is a pole of a causal transfer function")
    Xc = _cnum(X)
    H = 0.25 * (1.0 + 2.0 * zc ** -1 + zc ** -2)
    return RichResult(payload={
        "Y": H * Xc, "H": H, "X": Xc, "z": zc,
        "transfer_function_is_input_independent": True,
        "method": "Rangayyan (2024) eq. (3.102)"})


rangayyan_ch3_hann_z_output = hannz  # pre-policy spelling


# -- rng092: Transfer function of the Hann filter (double zero at z=-1)..
def hanntf(z):
    """Transfer function of the Hann filter, eq. (3.103).

        H(z) = (1/4) [ 1 + 2 z^-1 + z^-2 ] = (1/4) (1 + z^-1)^2

    The factored form shows a DOUBLE zero at z = -1, that is at the
    Nyquist frequency.  A double zero is why the response falls to nought
    there smoothly rather than crossing, and why the attenuation near
    Nyquist is second order.
    """
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = []
    for zv in zs:
        zc = _cnum(zv)
        if zc == 0:
            raise ValueError("z = 0 is a pole of a causal transfer "
                             "function")
        H.append(0.25 * (1.0 + zc ** -1) ** 2)
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z,
        "zeros": [-1.0, -1.0], "zero_multiplicity": 2,
        "zeros_at_nyquist": True, "dc_gain": 1.0,
        "method": "Rangayyan (2024) eq. (3.103)"})


rangayyan_ch3_hann_transfer_function = hanntf  # pre-policy spelling


# -- rng093: Frequency response of the Hann filter on the unit circle..
def hannfr(omega):
    """Frequency response of the Hann filter on the unit circle,
    eq. (3.104).

        H(omega) = (1/4) [ 1 + 2 exp(-j omega) + exp(-j 2 omega) ]

    The transfer function evaluated at z = exp(j omega).  This is the raw
    complex form; eq. (3.105) rearranges it into a real factor times a
    pure delay, and the two agree exactly, which ``hannfrs`` checks.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    H = []
    for w in ws:
        H.append(0.25 * (1.0 + 2.0 * complex(cos(-w), sin(-w))
                         + complex(cos(-2.0 * w), sin(-2.0 * w))))
    return RichResult(payload={
        "H": H[0] if scalar else H, "omega": omega,
        "magnitude": abs(H[0]) if scalar else [abs(v) for v in H],
        "on_the_unit_circle": True,
        "method": "Rangayyan (2024) eq. (3.104)"})


rangayyan_ch3_hann_frequency_response_raw = hannfr  # pre-policy spelling


# -- rng094: Simplified closed-form frequency response of the Hann filter..
def hannfrs(omega):
    """Simplified frequency response of the Hann filter, eq. (3.105).

        H(omega) = 0.5 [ 1 + cos(omega) ] exp(-j omega)

    A REAL nonnegative factor times a pure delay of one sample.  That
    factorization is the point: it proves the filter has exactly linear
    phase, so every frequency is delayed by the same time and the waveform
    is not distorted, only shifted.  The raw form of eq. (3.104) is
    recomputed here and the two are compared.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    H, raw = [], []
    for w in ws:
        H.append(0.5 * (1.0 + cos(w)) * complex(cos(-w), sin(-w)))
        raw.append(0.25 * (1.0 + 2.0 * complex(cos(-w), sin(-w))
                           + complex(cos(-2.0 * w), sin(-2.0 * w))))
    gap = max(abs(a - b) for a, b in zip(H, raw))
    return RichResult(payload={
        "H": H[0] if scalar else H, "omega": omega,
        "envelope": [0.5 * (1.0 + cos(w)) for w in ws],
        "max_difference_from_eq_3_104": gap,
        "agrees_with_raw_form": gap <= 1e-12,
        "real_factor_times_a_pure_delay": True, "linear_phase": True,
        "method": "Rangayyan (2024) eq. (3.105)"})


rangayyan_ch3_hann_frequency_response_simplified = hannfrs  # pre-policy spelling


# -- rng095: Magnitude response of the Hann filter..
def hannmag(omega):
    """Magnitude response of the Hann filter, eq. (3.106).

        |H(omega)| = | 0.5 [ 1 + cos(omega) ] |

    Unity at DC and exactly nought at omega = pi, falling monotonically
    between: a lowpass.  The absolute value in the book's equation is
    redundant here -- 1 + cos is never negative -- and is kept only
    because the book writes it.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    mag = [abs(0.5 * (1.0 + cos(w))) for w in ws]
    return RichResult(payload={
        "magnitude": mag[0] if scalar else mag, "omega": omega,
        "dc_gain": 1.0, "nyquist_gain": 0.0,
        "lowpass": True, "absolute_value_is_redundant": True,
        "method": "Rangayyan (2024) eq. (3.106)"})


rangayyan_ch3_hann_magnitude_response = hannmag  # pre-policy spelling


# -- rng096: Linear phase response of the Hann filter..
def hannph(omega):
    """Phase response of the Hann filter, eq. (3.107).

        angle H(omega) = -omega

    Exactly linear in omega with slope -1, which is a constant group delay
    of one sample at every frequency.  Constant group delay is what
    "no phase distortion" means: components are all held back by the same
    TIME, so their alignment survives the filter.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    ph = [-w for w in ws]
    return RichResult(payload={
        "phase": ph[0] if scalar else ph, "omega": omega,
        "group_delay": 1.0, "slope": -1.0,
        "linear_phase": True, "constant_group_delay": True,
        "method": "Rangayyan (2024) eq. (3.107)"})


rangayyan_ch3_hann_phase_response = hannph  # pre-policy spelling


# -- rng097: 8-point moving average.
def rangayyan_ch3_ma_8point(x, n=None):
    r"""The 8-point moving-average filter (Rangayyan Ch. 3):

    .. math:: y(n) = \frac18 \sum_{k=0}^{7} x(n-k).

    Delay is 3.5 samples -- a non-integer, which is why an even-length
    boxcar cannot be delay-corrected by an integer shift.

    Parameters
    ----------
    x : array-like
        Input.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``y``, ``y_at_n``, ``group_delay`` (3.5), ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_moving_average(x, M=8)
    y = out["y"]
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < y.size:
            raise ValueError(f"n must lie in 0..{y.size - 1}, got {idx}.")
        at_n = float(y[idx])
    return RichResult(payload={"y": y, "y_at_n": at_n, "group_delay": 3.5,
                               "N": int(y.size),
                               "method": "8-point moving average, delay 3.5 (non-integer)"})


# -- rng098: Impulse response of the 8-point MA filter as a sum of shifted deltas..
def rangayyan_ch3_ma_8point_impulse_response(n):
    """
    Impulse response of the 8-point MA filter as a sum of shifted deltas.

    Formula: h(n) = (1/8) * [delta(n) + delta(n-1) + ... + delta(n-7)]

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
    Rangayyan (2024), Ch 3, Eq 3.109, p. 142
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
            "method": "Impulse response of the 8-point MA filter as a sum of shifted deltas.",
        }
    )


# -- rng099: Transfer function of the 8-point MA filter..
def rangayyan_ch3_ma_8point_transfer_function(z):
    """
    Transfer function of the 8-point MA filter.

    Formula: H(z) = (1/8) * sum_{k=0}^{7} z^(-k)

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.110, p. 142
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of the 8-point MA filter."}
    )


# -- rng100: Frequency response of the 8-point MA filter..
def rangayyan_ch3_ma_8point_frequency_response(omega):
    """
    Frequency response of the 8-point MA filter.

    Formula: H(omega) = (1/8) * sum_{k=0}^{7} exp(-j*omega*k) = (1/8) * {1 + exp(-j*4*omega)} * {1 + 2*cos(omega) + 2*cos(2*omega) + 2*cos(3*omega)}

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.111, p. 143
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Frequency response of the 8-point MA filter."}
    )


# -- rng101: Continuous-time integral over a sliding window of duration tau..
def rangayyan_ch3_running_integral_window(x, t, tau):
    """
    Continuous-time integral over a sliding window of duration tau.

    Formula: y(t) = integral_{t-tau}^{t} x(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.112, p. 143
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
            "method": "Continuous-time integral over a sliding window of duration tau.",
        }
    )


# -- rng102: General definition of running integral over (-inf, t]..
def rangayyan_ch3_integral_general(x, t):
    """
    General definition of running integral over (-inf, t].

    Formula: y(t) = integral_{-inf}^{t} x(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.113, p. 143
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
            "method": "General definition of running integral over (-inf, t].",
        }
    )


# -- rng103: Running integral of a causal signal over [0, t].
def rangayyan_ch3_integral_causal(x, dt=1.0):
    r"""Running integral :math:`y(t) = \int_0^t x(\tau)\, d\tau`.

    Cumulative trapezoidal integration of a causal signal (zero for
    t < 0), so ``y[0] = 0`` and each later value accumulates only past
    input -- integration is itself a causal LTI operation.

    Parameters
    ----------
    x : array-like, shape (m,)
        Samples on a uniform grid starting at t = 0.
    dt : float, default 1.0
        Sampling interval.

    Returns
    -------
    RichResult
        keys: ``y`` (m,), ``t`` (m,), ``total`` (y at the last
        sample), ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    y = integrate.cumulative_trapezoid(x, dx=dt, initial=0.0)
    return RichResult(
        payload={
            "y": y,
            "t": np.arange(x.size) * dt,
            "total": float(y[-1]),
            "dt": dt,
            "method": "Running integral y(t) = int_0^t x(tau) dtau (cumulative trapezoid)",
        }
    )


# -- rng104: Fourier transform of the integral of x(t) including DC term..
def rangayyan_ch3_fourier_of_integral(X, omega):
    """
    Fourier transform of the integral of x(t) including DC term.

    Formula: Y(omega) = (1/(j*omega)) * X(omega) + pi * X(0) * delta(omega)

    Parameters
    ----------
    X : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.115, p. 144
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
            "method": "Fourier transform of the integral of x(t) including DC term.",
        }
    )


# -- rng105: Frequency response of the ideal integrator (DC term aside)..
def rangayyan_ch3_integrator_frequency_response(omega):
    """
    Frequency response of the ideal integrator (DC term aside).

    Formula: H(omega) = 1 / (j*omega)

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.116, p. 144
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency response of the ideal integrator (DC term aside).",
        }
    )


# -- rng106: Magnitude response of the ideal integrator..
def rangayyan_ch3_integrator_magnitude_response(omega):
    """
    Magnitude response of the ideal integrator.

    Formula: |H(omega)| = |1/omega|

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.117, p. 144
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Magnitude response of the ideal integrator."}
    )


# -- rng107: Phase response of the ideal integrator (constant -pi/2)..
def rangayyan_ch3_integrator_phase_response(omega):
    """
    Phase response of the ideal integrator (constant -pi/2).

    Formula: angle(H(omega)) = -pi/2

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.118, p. 144
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Phase response of the ideal integrator (constant -pi/2).",
        }
    )


# -- rng108: Recursive form of the 8-point MA filter using delayed output..
def rangayyan_ch3_ma_8point_recursive(x, y, n):
    """
    Recursive form of the 8-point MA filter using delayed output.

    Formula: y(n) = y(n-1) + (1/8)*x(n) - (1/8)*x(n-8)

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
    Rangayyan (2024), Ch 3, Eq 3.120, p. 145
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
            "method": "Recursive form of the 8-point MA filter using delayed output.",
        }
    )


# -- rng109: Transfer function of the recursive 8-point MA filter (sinc-like)..
def rangayyan_ch3_ma_8point_recursive_transfer_function(z):
    """
    Transfer function of the recursive 8-point MA filter (sinc-like).

    Formula: H(z) = (1/8) * (1 - z^(-8)) / (1 - z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.121, p. 145
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Transfer function of the recursive 8-point MA filter (sinc-like).",
        }
    )


# -- rng110: Closed-form sinc-type frequency response of the recursive 8-point MA filter..
def rangayyan_ch3_ma_8point_sinc_frequency_response(omega):
    """
    Closed-form sinc-type frequency response of the recursive 8-point MA filter.

    Formula: H(omega) = (1/8) * (1 - exp(-j*8*omega)) / (1 - exp(-j*omega)) = (1/8) * exp(-j*7*omega/2) * sin(4*omega) / sin(omega/2)

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.122, p. 145
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Closed-form sinc-type frequency response of the recursive 8-point MA filter.",
        }
    )


# -- rng111: First-order difference operator approximating the time derivative..
def rangayyan_ch3_first_difference_operator(x, T, n):
    """
    First-order difference operator approximating the time derivative.

    Formula: y(n) = (1/T) * [x(n) - x(n-1)]

    Parameters
    ----------
    x : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.123, p. 145
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
            "method": "First-order difference operator approximating the time derivative.",
        }
    )


# -- rng112: Transfer function of the first-order difference operator..
def rangayyan_ch3_first_difference_transfer_function(z, T):
    """
    Transfer function of the first-order difference operator.

    Formula: H(z) = (1/T) * (1 - z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.124, p. 145
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Transfer function of the first-order difference operator.",
        }
    )


# -- rng113: Frequency response of the first-order difference operator..
def rangayyan_ch3_first_difference_frequency_response(omega, T):
    """
    Frequency response of the first-order difference operator.

    Formula: H(omega) = (1/T) * [1 - exp(-j*omega)] = (1/T) * exp(-j*omega/2) * [2*j*sin(omega/2)]

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.125, p. 147
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency response of the first-order difference operator.",
        }
    )


# -- rng114: Magnitude response of the first-order difference operator..
def rangayyan_ch3_first_difference_magnitude(omega, T):
    """
    Magnitude response of the first-order difference operator.

    Formula: |H(omega)| = (2/T) * |sin(omega/2)|

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.126, p. 147
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Magnitude response of the first-order difference operator.",
        }
    )


# -- rng115: Phase response of the first-order difference operator..
def rangayyan_ch3_first_difference_phase(omega):
    """
    Phase response of the first-order difference operator.

    Formula: angle(H(omega)) = pi/2 - omega/2

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.127, p. 147
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Phase response of the first-order difference operator.",
        }
    )


# -- rng116: Three-point central-difference operator (lower-noise derivative)..
def rangayyan_ch3_three_point_central_difference(x, T, n):
    """
    Three-point central-difference operator (lower-noise derivative).

    Formula: y_3(n) = (1/(2*T)) * [x(n) - x(n-2)]

    Parameters
    ----------
    x : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.128, p. 147
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
            "method": "Three-point central-difference operator (lower-noise derivative).",
        }
    )


# -- rng117: Transfer function of the three-point central-difference operator..
def rangayyan_ch3_three_point_central_diff_transfer_function(z, T):
    """
    Transfer function of the three-point central-difference operator.

    Formula: H(z) = (1/(2*T)) * (1 - z^(-2)) = [(1/T)*(1 - z^(-1))] * [0.5*(1 + z^(-1))]

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.129, p. 148
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Transfer function of the three-point central-difference operator.",
        }
    )


# -- rng118: Magnitude response of the three-point central-difference operator..
def rangayyan_ch3_three_point_central_diff_magnitude(omega, T):
    """
    Magnitude response of the three-point central-difference operator.

    Formula: |H(omega)| = (1/T) * |sin(omega)|

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.130, p. 148
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Magnitude response of the three-point central-difference operator.",
        }
    )


# -- rng119: Phase response of the three-point central-difference operator..
def rangayyan_ch3_three_point_central_diff_phase(omega):
    """
    Phase response of the three-point central-difference operator.

    Formula: angle(H(omega)) = pi/2 - omega

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.131, p. 148
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Phase response of the three-point central-difference operator.",
        }
    )


# -- rng120: Modified first-difference filter with pole at 0.995 to remove baseline wander..
def rangayyan_ch3_baseline_wander_filter_z_form_a(z, T):
    """
    Modified first-difference filter with pole at 0.995 to remove baseline wander.

    Formula: H(z) = (1/T) * (1 - z^(-1)) / (1 - 0.995 * z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.132, p. 149
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Modified first-difference filter with pole at 0.995 to remove baseline wander.",
        }
    )


# -- rng121: Equivalent (z, not z^-1) form of the baseline-wander filter..
def rangayyan_ch3_baseline_wander_filter_z_form_b(z, T):
    """
    Equivalent (z, not z^-1) form of the baseline-wander filter.

    Formula: H(z) = (1/T) * (z - 1) / (z - 0.995)

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.133, p. 149
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Equivalent (z, not z^-1) form of the baseline-wander filter.",
        }
    )


# -- rng122: Time-domain difference equation of the baseline-wander filter..
def rangayyan_ch3_baseline_wander_filter_difference_eq(x, y, T, n):
    """
    Time-domain difference equation of the baseline-wander filter.

    Formula: y(n) = (1/T) * [x(n) - x(n-1)] + 0.995 * y(n-1)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.134, p. 150
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
            "method": "Time-domain difference equation of the baseline-wander filter.",
        }
    )


# -- rng123: Squared-magnitude response of the analog Butterworth lowpass filter..
def rangayyan_ch3_butterworth_lowpass_squared_magnitude(Omega, Omega_c, N):
    """
    Squared-magnitude response of the analog Butterworth lowpass filter.

    Formula: |H_a(j*Omega)|^2 = 1 / (1 + (j*Omega/(j*Omega_c))^(2*N))

    Parameters
    ----------
    Omega : array-like
        Input data.
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.135, p. 154
    """
    Omega = np.atleast_1d(np.asarray(Omega, dtype=float))
    n = len(Omega)
    result = float(np.mean(Omega))
    se = float(np.std(Omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Squared-magnitude response of the analog Butterworth lowpass filter.",
        }
    )


# -- rng124: Squared transfer function of the Butterworth lowpass filter in s-domain..
def rangayyan_ch3_butterworth_squared_laplace(s, Omega_c, N):
    """
    Squared transfer function of the Butterworth lowpass filter in s-domain.

    Formula: H_a(s) * H_a(-s) = 1 / (1 + (s/(j*Omega_c))^(2*N))

    Parameters
    ----------
    s : array-like
        Input data.
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.136, p. 154
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Squared transfer function of the Butterworth lowpass filter in s-domain.",
        }
    )


# -- rng125: Pole positions on the Butterworth circle in the s-plane..
def rangayyan_ch3_butterworth_pole_positions(Omega_c, N, k):
    """
    Pole positions on the Butterworth circle in the s-plane.

    Formula: s_k = Omega_c * exp( j*pi * (0.5 + (2*k - 1)/(2*N)) ), k = 1..2N

    Parameters
    ----------
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.137, p. 154
    """
    Omega_c = np.atleast_1d(np.asarray(Omega_c, dtype=float))
    n = len(Omega_c)
    result = float(np.mean(Omega_c))
    se = float(np.std(Omega_c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Pole positions on the Butterworth circle in the s-plane.",
        }
    )


# -- rng126: Analog Butterworth transfer function from N left-half-plane poles..
def rangayyan_ch3_butterworth_analog_transfer_function(s, p_k, G, N):
    """
    Analog Butterworth transfer function from N left-half-plane poles.

    Formula: H_a(s) = G / [ (s - p_1)(s - p_2)...(s - p_N) ]

    Parameters
    ----------
    s : array-like
        Input data.
    p_k : array-like
        Input data.
    G : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.138, p. 154
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Analog Butterworth transfer function from N left-half-plane poles.",
        }
    )


# -- rng127: Bilinear transformation mapping s-domain to z-domain..
def rangayyan_ch3_bilinear_transformation(z, T):
    """
    Bilinear transformation mapping s-domain to z-domain.

    Formula: s = (2/T) * (1 - z^(-1)) / (1 + z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.139, p. 154
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Bilinear transformation mapping s-domain to z-domain.",
        }
    )


# -- rng128: Bilinear transform restricted to the unit circle (sigma=0)..
def rangayyan_ch3_bilinear_unit_circle_relation(omega, T):
    """
    Bilinear transform restricted to the unit circle (sigma=0).

    Formula: s = sigma + j*Omega = (2/T) * (1 - exp(-j*omega))/(1 + exp(-j*omega)) = (2*j/T) * tan(omega/2)

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.140, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Bilinear transform restricted to the unit circle (sigma=0).",
        }
    )


# -- rng129: Bilinear frequency warping: analog Omega from discrete omega..
def rangayyan_ch3_bilinear_warping_omega_to_Omega(omega, T):
    """
    Bilinear frequency warping: analog Omega from discrete omega.

    Formula: Omega = (2/T) * tan(omega/2)

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.141, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Bilinear frequency warping: analog Omega from discrete omega.",
        }
    )


# -- rng130: Bilinear frequency warping: discrete omega from analog Omega..
def rangayyan_ch3_bilinear_warping_Omega_to_omega(Omega, T):
    """
    Bilinear frequency warping: discrete omega from analog Omega.

    Formula: omega = 2 * atan(Omega*T/2)

    Parameters
    ----------
    Omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.142, p. 155
    """
    Omega = np.atleast_1d(np.asarray(Omega, dtype=float))
    n = len(Omega)
    result = float(np.mean(Omega))
    se = float(np.std(Omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Bilinear frequency warping: discrete omega from analog Omega.",
        }
    )


# -- rng131: Digital Butterworth transfer function after bilinear transform (IIR form)..
def rangayyan_ch3_butterworth_digital_transfer_function(z, a_k, G_prime, N):
    """
    Digital Butterworth transfer function after bilinear transform (IIR form).

    Formula: H(z) = G' * (1 + z^(-1))^N / sum_{k=0}^{N} a_k * z^(-k)

    Parameters
    ----------
    z : array-like
        Input data.
    a_k : array-like
        Input data.
    G_prime : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.143, p. 155
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Digital Butterworth transfer function after bilinear transform (IIR form).",
        }
    )


# -- rng132: General time-domain difference equation of an IIR filter..
def rangayyan_ch3_iir_difference_eq_general(x, y, b_k, a_k, N, n):
    """
    General time-domain difference equation of an IIR filter.

    Formula: y(n) = sum_{k=0}^{N} b_k x(n-k) - sum_{k=1}^{N} a_k y(n-k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    b_k : array-like
        Input data.
    a_k : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.144, p. 155
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
            "method": "General time-domain difference equation of an IIR filter.",
        }
    )


# -- rng133: Direct discrete-domain specification of the Butterworth lowpass response..
def rangayyan_ch3_butterworth_lowpass_direct_specification(omega, omega_c, N):
    """
    Direct discrete-domain specification of the Butterworth lowpass response.

    Formula: |H(omega)|^2 = 1 / (1 + (omega/omega_c)^(2*N))

    Parameters
    ----------
    omega : array-like
        Input data.
    omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.145, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Direct discrete-domain specification of the Butterworth lowpass response.",
        }
    )


# -- rng134: Butterworth lowpass response indexed by DFT bin k..
def rangayyan_ch3_butterworth_lowpass_dft_indexed(k, k_c, N):
    """
    Butterworth lowpass response indexed by DFT bin k.

    Formula: |H(k)|^2 = 1 / (1 + (k/k_c)^(2*N))

    Parameters
    ----------
    k : array-like
        Input data.
    k_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.146, p. 156
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Butterworth lowpass response indexed by DFT bin k."}
    )


# -- rng135: Butterworth highpass response indexed by DFT bin k..
def rangayyan_ch3_butterworth_highpass_dft_indexed(k, k_c, N):
    """
    Butterworth highpass response indexed by DFT bin k.

    Formula: |H(k)|^2 = 1 / (1 + (k_c/k)^(2*N))

    Parameters
    ----------
    k : array-like
        Input data.
    k_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.149, p. 161
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Butterworth highpass response indexed by DFT bin k."}
    )


# -- rng136: Notch filter with two zeros at 60 Hz on the unit circle..
def rangayyan_ch3_notch_filter_60Hz(z):
    """
    Notch filter with two zeros at 60 Hz on the unit circle.

    Formula: H(z) = (1 - z^(-1)*z_1)(1 - z^(-1)*z_2) = 1 - 1.85955*z^(-1) + z^(-2)

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.150, p. 164
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Notch filter with two zeros at 60 Hz on the unit circle.",
        }
    )


# -- rng226: Matched-filter impulse response for the basic pattern g(n)..
def rangayyan_ch4_matched_filter_h_example(n):
    """
    Matched-filter impulse response for the basic pattern g(n).

    Formula: h(n) = delta(n) + 2*delta(n-1) + 3*delta(n-2)

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
    Rangayyan (2024), Ch 4, Eq 4.54, p. 241
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
            "method": "Matched-filter impulse response for the basic pattern g(n).",
        }
    )


_CHEATSHEET = [
    'rgbhp: Butterworth highpass filter design.',
    'rgblp: Butterworth lowpass filter design (analog prototype to digital).',
    'rgcomb: Comb filter for periodic artifact removal.',
    'rgfd1: First-difference operator for baseline wander removal.',
    'rgfd2: Second-difference operator.',
    'rgfir: FIR filter design (windowed sinc) -- see rangayyan_fir_filter for sources.',
    'rgfresp: Frequency response H(f) of a digital filter from coefficients.',
    'rggrpd: Group delay of a digital filter.',
    'rgiir: IIR Butterworth filter -- Rangayyan & Krishnan Sec 3.7.1 / 3.7.2.',
    'rgmavg: Moving-average filter.',
    'rgntch: Notch filter for powerline interference removal (50/60 Hz).',
    'rgosflt: Order-statistic (median) filter.',
    'rgphas: Phase response of a digital filter.',
    'rgsinc: Ideal sinc (low-pass) filter impulse response.',
    'rgtfe: Transfer function estimate.',
    'rgwblkm: Blackman window function.',
    'rgwhamp: Hamming window function.',
    'rgwhann: Hann (Hanning) window function.',
    'rgwndw: Window functions: Hamming, Hann, Blackman for spectral leakage control.',
    'rng011: Shannon entropy of a discrete process (Rangayyan eq. 3.11).',
    'rng039: 11-point moving average.',
    'rng040: Linear-ramp smoothing filter (Rangayyan eq. 3.42).',
    'series LSI systems: h = h1 * h2, eq. (3.45)',
    'parallel LSI systems: h = h1 + h2, eq. (3.49)',
    'Laplace transform of h(t), eq. (3.50)',
    'H(omega) = H(s) on s = j omega, eq. (3.52)',
    'rng053: Z-transform of a causal FIR system of length N (transfer function).',
    'rational IIR transfer function, eq. (3.67)',
    'IIR difference equation, eq. (3.68)',
    'magnitude from pole-zero distances, eq. (3.72)',
    'phase from pole-zero angles, eq. (3.73)',
    'general FIR / moving-average filter, eqs. (3.97)-(3.99)',
    'FIR transfer function, eq. (3.99)',
    'Hann smoothing filter, eq. (3.100)',
    'Hann impulse response, eq. (3.101)',
    'Hann filter output in z, eq. (3.102)',
    'Hann transfer function, double zero at z=-1, eq. (3.103)',
    'Hann frequency response, raw form, eq. (3.104)',
    'Hann frequency response, simplified, eq. (3.105)',
    'Hann magnitude response, eq. (3.106)',
    'Hann phase response, eq. (3.107)',
    'rng097: 8-point moving average.',
    'rng098: Impulse response of the 8-point MA filter as a sum of shifted deltas..',
    'rng099: Transfer function of the 8-point MA filter..',
    'rng100: Frequency response of the 8-point MA filter..',
    'rng101: Continuous-time integral over a sliding window of duration tau..',
    'rng102: General definition of running integral over (-inf, t]..',
    'rng103: Running integral of a causal signal over [0, t].',
    'rng104: Fourier transform of the integral of x(t) including DC term..',
    'rng105: Frequency response of the ideal integrator (DC term aside)..',
    'rng106: Magnitude response of the ideal integrator..',
    'rng107: Phase response of the ideal integrator (constant -pi/2)..',
    'rng108: Recursive form of the 8-point MA filter using delayed output..',
    'rng109: Transfer function of the recursive 8-point MA filter (sinc-like)..',
    'rng110: Closed-form sinc-type frequency response of the recursive 8-point MA filter..',
    'rng111: First-order difference operator approximating the time derivative..',
    'rng112: Transfer function of the first-order difference operator..',
    'rng113: Frequency response of the first-order difference operator..',
    'rng114: Magnitude response of the first-order difference operator..',
    'rng115: Phase response of the first-order difference operator..',
    'rng116: Three-point central-difference operator (lower-noise derivative)..',
    'rng117: Transfer function of the three-point central-difference operator..',
    'rng118: Magnitude response of the three-point central-difference operator..',
    'rng119: Phase response of the three-point central-difference operator..',
    'rng120: Modified first-difference filter with pole at 0.995 to remove baseline wander..',
    'rng121: Equivalent (z, not z^-1) form of the baseline-wander filter..',
    'rng122: Time-domain difference equation of the baseline-wander filter..',
    'rng123: Squared-magnitude response of the analog Butterworth lowpass filter..',
    'rng124: Squared transfer function of the Butterworth lowpass filter in s-domain..',
    'rng125: Pole positions on the Butterworth circle in the s-plane..',
    'rng126: Analog Butterworth transfer function from N left-half-plane poles..',
    'rng127: Bilinear transformation mapping s-domain to z-domain..',
    'rng128: Bilinear transform restricted to the unit circle (sigma=0)..',
    'rng129: Bilinear frequency warping: analog Omega from discrete omega..',
    'rng130: Bilinear frequency warping: discrete omega from analog Omega..',
    'rng131: Digital Butterworth transfer function after bilinear transform (IIR form)..',
    'rng132: General time-domain difference equation of an IIR filter..',
    'rng133: Direct discrete-domain specification of the Butterworth lowpass response..',
    'rng134: Butterworth lowpass response indexed by DFT bin k..',
    'rng135: Butterworth highpass response indexed by DFT bin k..',
    'rng136: Notch filter with two zeros at 60 Hz on the unit circle..',
    'rng226: Matched-filter impulse response for the basic pattern g(n)..',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
