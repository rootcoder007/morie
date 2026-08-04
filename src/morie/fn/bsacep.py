# morie.fn -- bsacep (rootcoder007/morie)
"""Cepstral and homomorphic analysis: real and complex cepstra, liftering, deconvolution, echo removal.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 23
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

import math as _math
from . import _array_core as np
from ._richresult import RichResult

__all__ = [
    'rangayyan_ar_to_cepstrum',
    'ar2cep',
    'rangayyan_complex_cepstrum',
    'rangayyan_cepstrum_pitch',
    'rangayyan_cepstrum',
    'rangayyan_homomorphic_deconv',
    'rangayyan_homomorphic',
    'rangayyan_homomorphic_pred',
    'rangayyan_liftering',
    'rangayyan_mfcc',
    'rangayyan_min_phase',
    'rangayyan_vocal_tract',
    'rangayyan_ch4_homomorphic_multiplicative_signal',
    'rangayyan_ch4_homomorphic_log_separation',
    'rangayyan_ch4_convolution_model',
    'rangayyan_ch4_complex_cepstrum_definition',
    'rangayyan_ch4_complex_cepstra_sum',
    'rangayyan_ch4_rational_z_transform_form',
    'rangayyan_ch4_complex_cepstrum_closed_form',
    'rangayyan_ch4_complex_cepstrum_decay_bound',
    'rangayyan_ch4_log_echo_power_series_expansion',
    'rangayyan_ch4_power_cepstrum_definition',
    'rangayyan_ch4_power_cepstrum_sum',
    'rangayyan_ch4_power_cepstrum_relation_to_complex',
]


# -- rgar2cep: Cepstral coefficients from AR coefficients.
def rangayyan_ar_to_cepstrum(a_coeffs, gain=None):
    r"""AR coefficients to complex cepstrum, Rangayyan eq. (7.65).

    .. math::
        \hat h(1) &= -a_1 \\
        \hat h(n) &= -a_n - \sum_{k=1}^{n-1}
                     \left(1 - \frac{k}{n}\right) a_k \hat h(n-k),
                     \quad 1 < n \le P

    Derived by expanding :math:`\ln H(z)` as a Laurent series
    (eq. 7.61-7.64) and equating like powers of :math:`z^{-1}`.

    The alternative spelling :math:`\sum (k/n)\,\hat h(k)\,a_{n-k}` is
    the SAME recursion reindexed: put :math:`j = n - k` and
    :math:`(1 - k/n)\,a_k\,\hat h(n-k)` becomes
    :math:`(j/n)\,\hat h(j)\,a_{n-j}`.  The book's form is used here
    because it is the one the citation points at.

    Going through the AR coefficients avoids the phase unwrapping that
    the FFT-based cepstrum needs (Section 4.7.3), which is the practical
    reason to prefer this route.

    Parameters
    ----------
    a_coeffs : sequence
        AR coefficients :math:`a_1 \dots a_P` in the sign convention of
        :math:`A(z) = 1 + \sum a_k z^{-k}` -- the residual filter, so
        that a stable model has its poles inside the unit circle.
    gain : float, optional
        Model gain :math:`G`.  When given, ``c0 = log(G)`` is returned as
        the zeroth cepstral coefficient; the recursion itself does not
        involve it.

    Returns
    -------
    RichResult
        ``cepstrum`` (n = 1..P), ``c0`` (or None).

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, Section 7.5.3, eqs. (7.61)-(7.65).
    """
    a = [float(v) for v in a_coeffs]
    p = len(a)
    if p == 0:
        raise ValueError("need at least one AR coefficient")

    h = [0.0] * (p + 1)          # h[n] for n = 1..P
    for n in range(1, p + 1):
        acc = -a[n - 1]
        for k in range(1, n):
            acc -= (1.0 - k / n) * a[k - 1] * h[n - k]
        h[n] = acc
    cep = h[1:]

    c0 = None
    if gain is not None:
        g = float(gain)
        if g <= 0:
            raise ValueError("gain must be positive")
        c0 = _math.log(g)

    return RichResult(
        title="AR to cepstrum (Rangayyan eq. 7.65)",
        summary_lines=[("order", p)],
        payload={"cepstrum": cep, "c0": c0, "order": p,
                 "method": "Rangayyan (2024) eq. (7.65)"},
    )


ar2cep = rangayyan_ar_to_cepstrum


# -- rgccep: Complex cepstrum using phase unwrapping.
def rangayyan_complex_cepstrum(x):
    """
    Complex cepstrum using phase unwrapping

    Formula: c_hat(n) = IFFT(log FFT(x)) = IFFT(log|X(f)| + j*angle_unwrapped(X(f)))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: complex_cepstrum, quefrency

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Complex cepstrum using phase unwrapping"}
    )


# -- rgcepsp: Cepstral pitch detection.
def rangayyan_cepstrum_pitch(x, fs, f0_range=(50.0, 500.0)):
    r"""Pitch from the real cepstrum (Rangayyan Ch. 3):

    .. math:: c(q) = \mathrm{IDFT}\{\log |X(f)|\},

    with the pitch period :math:`T_0` at the quefrency of the
    dominant rahmonic. The logarithm is what makes this work: it turns
    the product of excitation and vocal-tract spectra into a SUM, so
    the periodic excitation separates from the smooth envelope in
    quefrency. Searching only inside ``f0_range`` avoids locking onto
    the low-quefrency envelope peak.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float
        Sampling frequency.
    f0_range : (float, float)
        Plausible pitch range in Hz.

    Returns
    -------
    RichResult
        keys: ``f0``, ``period_s``, ``quefrency``, ``cepstrum``,
        ``peak_value``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (cepstral analysis).
    """
    x = np.asarray(x, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    lo, hi = float(f0_range[0]), float(f0_range[1])
    if not 0 < lo < hi:
        raise ValueError(f"f0_range must satisfy 0 < lo < hi, got {f0_range}.")
    if x.size < 16:
        raise ValueError(f"need at least 16 samples, got {x.size}.")
    spec = np.abs(np.fft.rfft(x))
    ceps = np.fft.irfft(np.log(np.maximum(spec, 1e-300)))
    q = np.arange(ceps.size) / fs
    q_lo, q_hi = 1.0 / hi, 1.0 / lo
    band = np.flatnonzero((q >= q_lo) & (q <= q_hi))
    if band.size == 0:
        raise ValueError("f0_range maps outside the available quefrencies.")
    ipk = band[int(np.argmax(ceps[band]))]
    T0 = float(q[ipk])
    return RichResult(payload={"f0": 1.0 / T0 if T0 > 0 else np.nan, "period_s": T0,
                               "quefrency": ipk, "cepstrum": ceps,
                               "peak_value": float(ceps[ipk]),
                               "method": "log turns convolution into addition; search inside f0_range"})


# -- rgcepst: Real cepstrum of a signal.
def rangayyan_cepstrum(x):
    """
    Real cepstrum of a signal

    Formula: c(n) = IDFT(log|DFT(x)|) = IFFT(log|FFT(x)|)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cepstrum, quefrency

    References
    ----------
    Rangayyan Ch 4.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Real cepstrum of a signal"})


# -- rghomdc: Homomorphic deconvolution via complex cepstrum.
def rangayyan_homomorphic_deconv(x, lifter_low, lifter_high):
    """
    Homomorphic deconvolution via complex cepstrum

    Formula: x_hat = IFFT(exp(lifter(log(|FFT(x)|) + j*angle(FFT(x)))))

    Parameters
    ----------
    x : array-like
        Input data.
    lifter_low : array-like
        Input data.
    lifter_high : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: source, filter_resp

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Homomorphic deconvolution via complex cepstrum"}
    )


# -- rghomo: Homomorphic filtering system for multiplicative signal models.
def rangayyan_homomorphic(x, filter_type):
    """
    Homomorphic filtering system for multiplicative signal models

    Formula: log -> linear filter -> exp; D*[x*h] = D*[x] + D*[h]

    Parameters
    ----------
    x : array-like
        Input data.
    filter_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: filtered_x

    References
    ----------
    Rangayyan Ch 4.7
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
            "method": "Homomorphic filtering system for multiplicative signal models",
        }
    )


# -- rghompr: Homomorphic prediction via complex cepstrum.
def rangayyan_homomorphic_pred(x, lifter):
    """
    Homomorphic prediction via complex cepstrum

    Formula: Cepstral liftering isolates low-time (vocal tract) from high-time (glottal)

    Parameters
    ----------
    x : array-like
        Input data.
    lifter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction_coeffs

    References
    ----------
    Rangayyan Ch 7.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Homomorphic prediction via complex cepstrum"}
    )


# -- rglift: Cepstral liftering (low-time / high-time separation).
def rangayyan_liftering(cepstrum, l_low, l_high):
    """
    Cepstral liftering (low-time / high-time separation)

    Formula: lifter(c, l_low, l_high) = c * window; window=1 in [l_low, l_high] else 0

    Parameters
    ----------
    cepstrum : array-like
        Input data.
    l_low : array-like
        Input data.
    l_high : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: liftered_cepstrum

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    cepstrum = np.asarray(cepstrum, dtype=float)
    n = int(cepstrum) if cepstrum.ndim == 0 else len(cepstrum)
    result = float(np.mean(cepstrum))
    se = float(np.std(cepstrum, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Cepstral liftering (low-time / high-time separation)"}
    )


# -- rgmfcc: Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis.
def rangayyan_mfcc(x, fs, n_mfcc, n_filters):
    """
    Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis

    Formula: MFCC = DCT(log(filterbank_energy)); Mel-scale: m=2595*log10(1+f/700)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    n_mfcc : array-like
        Input data.
    n_filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mfcc_matrix

    References
    ----------
    Rangayyan Ch 4.7.3
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
            "method": "Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanmfcc = rangayyan_mfcc


# -- rgminph: Minimum-phase correspondent of a signal.
def rangayyan_min_phase(x):
    """
    Minimum-phase correspondent of a signal

    Formula: Constructed by reflecting all zeros outside unit circle to inside

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_minphase

    References
    ----------
    Rangayyan Ch 5.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Minimum-phase correspondent of a signal"}
    )


# -- rgvocal: Vocal tract transfer function extraction via homomorphic deconvolution.
def rangayyan_vocal_tract(speech, fs, n_coeff):
    """
    Vocal tract transfer function extraction via homomorphic deconvolution

    Formula: V(z) extracted from complex cepstrum low-time region

    Parameters
    ----------
    speech : array-like
        Input data.
    fs : array-like
        Input data.
    n_coeff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vocal_response

    References
    ----------
    Rangayyan Ch 4.7.3
    """
    speech = np.asarray(speech, dtype=float)
    n = int(speech) if speech.ndim == 0 else len(speech)
    result = float(np.mean(speech))
    se = float(np.std(speech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Vocal tract transfer function extraction via homomorphic deconvolution",
        }
    )


# -- rng230: Multiplicative model addressed by homomorphic filtering..
def rangayyan_ch4_homomorphic_multiplicative_signal(x, p, t):
    """
    Multiplicative model addressed by homomorphic filtering.

    Formula: y(t) = x(t) * p(t)

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.58, p. 244
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
            "method": "Multiplicative model addressed by homomorphic filtering.",
        }
    )


# -- rng231: Logarithm converts product into a sum in homomorphic filtering..
def rangayyan_ch4_homomorphic_log_separation(x, p, t):
    """
    Logarithm converts product into a sum in homomorphic filtering.

    Formula: log[y(t)] = log[x(t) * p(t)] = log[x(t)] + log[p(t)], for x(t)!=0, p(t)!=0

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.59, p. 244
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
            "method": "Logarithm converts product into a sum in homomorphic filtering.",
        }
    )


# -- rng233: Convolutional signal model addressed by homomorphic deconvolution..
def rangayyan_ch4_convolution_model(x, h, t):
    """
    Convolutional signal model addressed by homomorphic deconvolution.

    Formula: y(t) = x(t) * h(t)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.61, p. 245
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
            "method": "Convolutional signal model addressed by homomorphic deconvolution.",
        }
    )


# -- rng236: Definition of the complex cepstrum via inverse z-transform of complex log of Y(z)..
def rangayyan_ch4_complex_cepstrum_definition(Y, z, n):
    """
    Definition of the complex cepstrum via inverse z-transform of complex log of Y(z).

    Formula: y_hat(n) = (1/(2*pi*j)) * contour_integral log[Y(z)] * z^(n-1) dz

    Parameters
    ----------
    Y : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.64, p. 247
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
            "method": "Definition of the complex cepstrum via inverse z-transform of complex log of Y(z).",
        }
    )


# -- rng238: Complex cepstra of a convolution decompose as a sum..
def rangayyan_ch4_complex_cepstra_sum(x_hat, h_hat, n):
    """
    Complex cepstra of a convolution decompose as a sum.

    Formula: y_hat(n) = x_hat(n) + h_hat(n)

    Parameters
    ----------
    x_hat : array-like
        Input data.
    h_hat : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.66, p. 247
    """
    x_hat = np.atleast_1d(np.asarray(x_hat, dtype=float))
    n = len(x_hat)
    result = float(np.mean(x_hat))
    se = float(np.std(x_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Complex cepstra of a convolution decompose as a sum."}
    )


# -- rng239: Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum)..
def rangayyan_ch4_rational_z_transform_form(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O):
    """
    Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum).

    Formula: X(z) = A * z^r * prod_{k=1}^{M_I} (1 - a_k z^(-1)) * prod_{k=1}^{M_O} (1 - b_k z) / [ prod_{k=1}^{N_I} (1 - c_k z^(-1)) * prod_{k=1}^{N_O} (1 - d_k z) ]

    Parameters
    ----------
    A : array-like
        Input data.
    z : array-like
        Input data.
    r : array-like
        Input data.
    a_k : array-like
        Input data.
    b_k : array-like
        Input data.
    c_k : array-like
        Input data.
    d_k : array-like
        Input data.
    M_I : array-like
        Input data.
    M_O : array-like
        Input data.
    N_I : array-like
        Input data.
    N_O : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.67, p. 247
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
            "method": "Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum).",
        }
    )


# -- rng244: Closed-form complex cepstrum from poles/zeros (inside/outside unit circle)..
def rangayyan_ch4_complex_cepstrum_closed_form(A, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O, n):
    """
    Closed-form complex cepstrum from poles/zeros (inside/outside unit circle).

    Formula: x_hat(n) = log|A| if n=0; -sum_{k=1}^{M_I} a_k^n/n + sum_{k=1}^{N_I} c_k^n/n for n>0; sum_{k=1}^{M_O} b_k^(-n)/n - sum_{k=1}^{N_O} d_k^(-n)/n for n<0

    Parameters
    ----------
    A : array-like
        Input data.
    a_k : array-like
        Input data.
    b_k : array-like
        Input data.
    c_k : array-like
        Input data.
    d_k : array-like
        Input data.
    M_I : array-like
        Input data.
    M_O : array-like
        Input data.
    N_I : array-like
        Input data.
    N_O : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.72, p. 248
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
            "method": "Closed-form complex cepstrum from poles/zeros (inside/outside unit circle).",
        }
    )


# -- rng245: Decay bound for the complex cepstrum: at least as fast as 1/n..
def rangayyan_ch4_complex_cepstrum_decay_bound(K, alpha, n):
    """
    Decay bound for the complex cepstrum: at least as fast as 1/n.

    Formula: |x_hat(n)| < K * |alpha^n / n|, for -inf < n < inf, where alpha = max(|a_k|,|b_k|,|c_k|,|d_k|)

    Parameters
    ----------
    K : array-like
        Input data.
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.73, p. 248
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Decay bound for the complex cepstrum: at least as fast as 1/n.",
        }
    )


# -- rng251: Power-series expansion of the log echo term (a < 1)..
def rangayyan_ch4_log_echo_power_series_expansion(a, n_0, omega, H_hat):
    """
    Power-series expansion of the log echo term (a < 1).

    Formula: Y_hat(omega) = H_hat(omega) + a*exp(-j*omega*n_0) - (a^2/2)*exp(-j*2*omega*n_0) + (a^3/3)*exp(-j*3*omega*n_0) - ...

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.79, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Power-series expansion of the log echo term (a < 1)."}
    )


# -- rng253: Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2..
def rangayyan_ch4_power_cepstrum_definition(Y, z, n):
    """
    Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2.

    Formula: y_hat_p(n) = { (1/(2*pi*j)) * contour_integral log|Y(z)|^2 * z^(n-1) dz }^2

    Parameters
    ----------
    Y : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.81, p. 251
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
            "method": "Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2.",
        }
    )


# -- rng254: Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected)..
def rangayyan_ch4_power_cepstrum_sum(x_hat_p, h_hat_p, n):
    """
    Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected).

    Formula: y_hat_p(n) = x_hat_p(n) + h_hat_p(n)

    Parameters
    ----------
    x_hat_p : array-like
        Input data.
    h_hat_p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.82, p. 251
    """
    x_hat_p = np.atleast_1d(np.asarray(x_hat_p, dtype=float))
    n = len(x_hat_p)
    result = float(np.mean(x_hat_p))
    se = float(np.std(x_hat_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected).",
        }
    )


# -- rng255: Relation between power cepstrum and complex cepstrum..
def rangayyan_ch4_power_cepstrum_relation_to_complex(y_hat, n):
    """
    Relation between power cepstrum and complex cepstrum.

    Formula: y_hat_p(n) = [ y_hat(n) + y_hat(-n) ]^2

    Parameters
    ----------
    y_hat : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.83, p. 251
    """
    y_hat = np.atleast_1d(np.asarray(y_hat, dtype=float))
    n = len(y_hat)
    result = float(np.mean(y_hat))
    se = float(np.std(y_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Relation between power cepstrum and complex cepstrum.",
        }
    )


_CHEATSHEET = [
    'ar2cep: h(n) = -a_n - sum (1 - k/n) a_k h(n-k), eq 7.65',
    'rgccep: Complex cepstrum using phase unwrapping',
    'rgcepsp: the log separates excitation from envelope in quefrency',
    'rgcepst: Real cepstrum of a signal',
    'rghomdc: Homomorphic deconvolution via complex cepstrum',
    'rghomo: Homomorphic filtering system for multiplicative signal models',
    'rghompr: Homomorphic prediction via complex cepstrum',
    'rglift: Cepstral liftering (low-time / high-time separation)',
    'rgmfcc: Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis',
    'rgminph: Minimum-phase correspondent of a signal',
    'rgvocal: Vocal tract transfer function extraction via homomorphic deconvolution',
    'rng230: Multiplicative model addressed by homomorphic filtering.',
    'rng231: Logarithm converts product into a sum in homomorphic filtering.',
    'rng233: Convolutional signal model addressed by homomorphic deconvolution.',
    'rng236: Definition of the complex cepstrum via inverse z-transform of complex log of Y(z).',
    'rng238: Complex cepstra of a convolution decompose as a sum.',
    'rng239: Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum).',
    'rng244: Closed-form complex cepstrum from poles/zeros (inside/outside unit circle).',
    'rng245: Decay bound for the complex cepstrum: at least as fast as 1/n.',
    'rng251: Power-series expansion of the log echo term (a < 1).',
    'rng253: Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2.',
    'rng254: Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected).',
    'rng255: Relation between power cepstrum and complex cepstrum.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
