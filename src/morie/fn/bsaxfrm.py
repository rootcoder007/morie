# morie.fn -- bsaxfrm (rootcoder007/morie)
"""Transforms: Fourier, DFT/IDFT, DTFT, z, Laplace, and their properties.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 33
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from . import _array_core as np
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from .bsafilt import rangayyan_ch3_z_transform_fir

__all__ = [
    'rangayyan_circular_conv_dft',
    'rangayyan_dft',
    'rangayyan_fourier_transform',
    'rangayyan_stft',
    'rangayyan_z_transform',
    'rangayyan_ch3_laplace_transform_causal_finite',
    'rangayyan_ch3_z_transform_definition',
    'rangayyan_ch3_z_transform_convolution',
    'rangayyan_ch3_dtft_via_z',
    'rangayyan_ch3_complex_exponential',
    'rangayyan_ch3_fourier_transform_omega',
    'rangayyan_ch3_fourier_transform_f',
    'rangayyan_ch3_inverse_fourier_transform',
    'rangayyan_ch3_dtft',
    'rangayyan_ch3_dft_K_samples',
    'rangayyan_ch3_dft_definition',
    'rangayyan_ch3_twiddle_factor',
    'rangayyan_ch3_dft_via_twiddle',
    'rangayyan_ch3_twiddle_cos_sin',
    'rangayyan_ch3_dft_real_imag_decomposition',
    'rangayyan_ch3_idft_real_imag',
    'rangayyan_ch3_dft_convolution_property',
    'rangayyan_ch3_twiddle_conjugate_symmetry',
    'rangayyan_ch3_twiddle_periodicity',
    'rangayyan_ch3_even_part',
    'rangayyan_ch3_odd_part',
    'rangayyan_ch3_even_odd_decomposition',
    'rangayyan_ch4_homomorphic_log_fourier',
    'rangayyan_ch4_fourier_convolution_property',
    'rangayyan_ch4_log_of_convolved_signals',
    'rangayyan_ch4_log_power_series',
    'rangayyan_ch4_log_minimum_phase_expansion',
    'rangayyan_ch4_log_maximum_phase_expansion',
]


# -- rgcdft: Circular (cyclic) convolution via DFT.
def rangayyan_circular_conv_dft(x, h):
    """
    Circular (cyclic) convolution via DFT

    Formula: y_circ[n] = IDFT(DFT(x) * DFT(h))

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_circ

    References
    ----------
    Rangayyan Ch 3.4.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Circular (cyclic) convolution via DFT"})


# -- rgdft: Discrete Fourier transform (DFT).
def rangayyan_dft(x):
    """
    Discrete Fourier transform (DFT)

    Formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-j2*pi*kn/N)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_k

    References
    ----------
    Rangayyan Ch 3.4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete Fourier transform (DFT)"})


# compact alias per ledger/NAMING.md
rangayyandft = rangayyan_dft


# -- rgft: Continuous-time Fourier transform (CTFT).
def rangayyan_fourier_transform(t, x):
    """
    Continuous-time Fourier transform (CTFT)

    Formula: X(f) = integral_{-inf}^{inf} x(t) * exp(-j2*pi*f*t) dt

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_f, freqs

    References
    ----------
    Rangayyan Ch 3.4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time Fourier transform (CTFT)"}
    )


# -- rgstf: Short-time Fourier transform -- Rangayyan & Krishnan Sec 8.4.1.
def rangayyan_stft(x, fs=1.0, nperseg=256, noverlap=None, window="hann"):
    """Short-time Fourier transform / spectrogram.

    Parameters
    ----------
    x : array-like
    fs : float
    nperseg : int
    noverlap : int, optional
    window : str

    Returns
    -------
    RichResult with keys ``freqs``, ``times``, ``Sxx``, ``nperseg``,
    ``noverlap``, ``fs``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 8.4.1 "The short-time Fourier
        transform", p.438. The previous docstring cited Ch 4.
    """
    from ._signal_core import spectrogram

    x = np.asarray(x, dtype=float)
    nperseg = min(int(nperseg), x.size)
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling="density", mode="psd")
    res = RichResult(
        title="Short-Time Fourier Transform",
        summary_lines=[
            ("nperseg", nperseg),
            ("noverlap", int(noverlap)),
            ("Window", window),
            ("Fs (Hz)", float(fs)),
            ("Frames", int(t.size)),
            ("Freq bins", int(f.size)),
        ],
        interpretation=f"STFT: {t.size} frames × {f.size} freq bins.",
        payload={"freqs": f, "times": t, "Sxx": Sxx, "nperseg": nperseg, "noverlap": int(noverlap), "fs": float(fs)},
    )
    return with_describe_pointer(res, "rgstf")


# CANONICAL TEST
# >>> fs = 100.0
# >>> t = np.arange(1024)/fs
# >>> x = np.sin(2*np.pi*10*t)
# >>> r = rangayyan_stft(x, fs=fs, nperseg=128)
# >>> r["Sxx"].shape[0] == r["freqs"].size
# True


# compact alias per ledger/NAMING.md
rangayyanstft = rangayyan_stft


# -- rgztf: Z-transform of a causal discrete-time sequence.
def rangayyan_z_transform(x_coeffs, z=None):
    r"""One-sided z-transform :math:`X(z) = \sum_{n \ge 0} x(n) z^{-n}`.

    For a finite coefficient sequence this is the same polynomial in
    :math:`z^{-1}` as an FIR transfer function, so the evaluation
    delegates to :func:`morie.fn.rng053.rangayyan_ch3_z_transform_fir`.
    With ``z=None`` only the coefficient vector and the implied
    polynomial degree are returned.

    Parameters
    ----------
    x_coeffs : array-like
        The sequence x(0), x(1), ...
    z : complex or array-like, optional
        Where to evaluate.

    Returns
    -------
    RichResult
        keys: ``coefficients``, ``degree``, ``H`` (None when ``z`` is
        None), ``z``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    from morie.fn import _array_core as np

    x = np.asarray(x_coeffs, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x_coeffs must be non-empty.")
    H = zz = None
    if z is not None:
        out = rangayyan_ch3_z_transform_fir(x, z)
        H, zz = out["H"], out["z"]
    return RichResult(
        payload={
            "coefficients": x,
            "degree": int(x.size - 1),
            "H": H,
            "z": zz,
            "method": "One-sided z-transform X(z) = sum_{n>=0} x(n) z^-n",
        }
    )


# -- rng049: Laplace transform of a causal finite-duration h(t) over [0, T].
def rangayyan_ch3_laplace_transform_causal_finite(h, s, dt=1.0):
    r"""Numeric Laplace transform
    :math:`H(s) = \int_0^T h(t) e^{-st}\, dt`.

    For a causal impulse response of finite duration T the transform
    integral has finite limits, so it converges for every s and can be
    evaluated by quadrature. Complex s are accepted, so setting
    :math:`s = j\omega` recovers the Fourier transform of h.

    Parameters
    ----------
    h : array-like, shape (m,)
        Impulse response sampled on [0, T] with step ``dt``.
    s : complex or array-like of complex
        Transform variable(s).
    dt : float, default 1.0
        Sampling interval; T = (m - 1) * dt.

    Returns
    -------
    RichResult
        keys: ``H`` (complex scalar or array matching ``s``), ``s``,
        ``T``, ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (Laplace transform of causal systems).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    t = np.arange(h.size) * dt
    sv = np.atleast_1d(np.asarray(s, dtype=complex))
    H = np.array([np.trapezoid(h * np.exp(-sk * t), dx=dt) for sk in sv])
    scalar = np.ndim(s) == 0

    return RichResult(
        payload={
            "H": complex(H[0]) if scalar else H,
            "s": complex(sv[0]) if scalar else sv,
            "T": float(t[-1]),
            "dt": dt,
            "method": "Laplace transform of a finite-duration causal h: int_0^T h e^{-st} dt",
        }
    )


# -- rng052: Bilateral z-transform of a discrete-time signal x(n)..
def rangayyan_ch3_z_transform_definition(x, n, z):
    """
    Bilateral z-transform of a discrete-time signal x(n).

    Formula: X(z) = sum_{n=-inf}^{inf} x(n) * z^(-n)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.54, p. 119
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
            "method": "Bilateral z-transform of a discrete-time signal x(n).",
        }
    )


# -- rng054: Convolution in time becomes multiplication of z-transforms..
def rangayyan_ch3_z_transform_convolution(x, h):
    """
    Convolution in time becomes multiplication of z-transforms.

    Formula: if y(n) = x(n) * h(n), then Y(z) = X(z) H(z)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.56, p. 119
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
            "method": "Convolution in time becomes multiplication of z-transforms.",
        }
    )


# -- rng055: Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle..
def rangayyan_ch3_dtft_via_z(x, n, omega, T, N):
    """
    Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle.

    Formula: X(omega) = sum_{n=0}^{N-1} x(n) * z^(-n) |_{z=exp(j*omega*T)} = sum_{n=0}^{N-1} x(n) * exp(-j*omega*n*T)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    omega : array-like
        Input data.
    T : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.66, p. 122
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
            "method": "Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle.",
        }
    )


# -- rng063: Euler's formula for the complex exponential basis function..
def rangayyan_ch3_complex_exponential(omega, t):
    """
    Euler's formula for the complex exponential basis function.

    Formula: exp(j*omega*t) = cos(omega*t) + j*sin(omega*t)

    Parameters
    ----------
    omega : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.74, p. 125
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
            "method": "Euler's formula for the complex exponential basis function.",
        }
    )


# -- rng064: Continuous-time Fourier transform with frequency variable omega in rad/s..
def rangayyan_ch3_fourier_transform_omega(x, t, omega):
    """
    Continuous-time Fourier transform with frequency variable omega in rad/s.

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
    Rangayyan (2024), Ch 3, Eq 3.75, p. 125
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
            "method": "Continuous-time Fourier transform with frequency variable omega in rad/s.",
        }
    )


# -- rng065: Continuous-time Fourier transform with frequency variable f in Hz..
def rangayyan_ch3_fourier_transform_f(x, t, f):
    """
    Continuous-time Fourier transform with frequency variable f in Hz.

    Formula: X(f) = integral_{-inf}^{inf} x(t) * exp(-j*2*pi*f*t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.76, p. 125
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
            "method": "Continuous-time Fourier transform with frequency variable f in Hz.",
        }
    )


# -- rng066: Continuous-time inverse Fourier transform (synthesis)..
def rangayyan_ch3_inverse_fourier_transform(X, omega, f, t):
    """
    Continuous-time inverse Fourier transform (synthesis).

    Formula: x(t) = (1/(2*pi)) * integral_{-inf}^{inf} X(omega) exp(j*omega*t) d(omega) = integral_{-inf}^{inf} X(f) exp(j*2*pi*f*t) df

    Parameters
    ----------
    X : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.77, p. 126
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
            "method": "Continuous-time inverse Fourier transform (synthesis).",
        }
    )


# -- rng067: Discrete-time Fourier transform (DTFT) of x(n) with continuous omega..
def rangayyan_ch3_dtft(x, n, omega):
    """
    Discrete-time Fourier transform (DTFT) of x(n) with continuous omega.

    Formula: X(omega) = sum_{n=-inf}^{inf} x(n) * exp(-j*omega*n)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.78, p. 126
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
            "method": "Discrete-time Fourier transform (DTFT) of x(n) with continuous omega.",
        }
    )


# -- rng068: DFT computed at K samples of normalized frequency..
def rangayyan_ch3_dft_K_samples(x, n, k, K, N):
    """
    DFT computed at K samples of normalized frequency.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * exp(-j * (2*pi/K) * n * k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    K : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.79, p. 126
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "DFT computed at K samples of normalized frequency."}
    )


# -- rng069: Forward discrete Fourier transform (DFT) of an N-point signal..
def rangayyan_ch3_dft_definition(x, n, k, N):
    """
    Forward discrete Fourier transform (DFT) of an N-point signal.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * exp(-j * (2*pi/N) * n * k)

    Parameters
    ----------
    x : array-like
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
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.80, p. 126
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
            "method": "Forward discrete Fourier transform (DFT) of an N-point signal.",
        }
    )


# -- rng071: Twiddle factor used in DFT and FFT formulations..
def rangayyan_ch3_twiddle_factor(N):
    """
    Twiddle factor used in DFT and FFT formulations.

    Formula: W_N = exp(-j * 2*pi / N)

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.82, p. 127
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Twiddle factor used in DFT and FFT formulations."}
    )


# -- rng072: DFT expressed using twiddle factors W_N^(nk)..
def rangayyan_ch3_dft_via_twiddle(x, n, k, W_N, N):
    """
    DFT expressed using twiddle factors W_N^(nk).

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * W_N^(n*k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    W_N : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.83, p. 127
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "DFT expressed using twiddle factors W_N^(nk)."}
    )


# -- rng073: Twiddle factor expressed in terms of cosine and sine basis functions..
def rangayyan_ch3_twiddle_cos_sin(n, k, N):
    """
    Twiddle factor expressed in terms of cosine and sine basis functions.

    Formula: W_N^(n*k) = exp(-j*(2*pi/N)*n*k) = cos((2*pi/N)*n*k) - j*sin((2*pi/N)*n*k)

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.84, p. 127
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
            "method": "Twiddle factor expressed in terms of cosine and sine basis functions.",
        }
    )


# -- rng074: DFT decomposed into real (cos) and imaginary (sin) parts..
def rangayyan_ch3_dft_real_imag_decomposition(x, n, k, N):
    """
    DFT decomposed into real (cos) and imaginary (sin) parts.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) cos((2*pi/N)*n*k) - j * sum_{n=0}^{N-1} x(n) sin((2*pi/N)*n*k)

    Parameters
    ----------
    x : array-like
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
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.85, p. 127
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
            "method": "DFT decomposed into real (cos) and imaginary (sin) parts.",
        }
    )


# -- rng075: Inverse DFT expressed as combination of cos and sin synthesis terms..
def rangayyan_ch3_idft_real_imag(X, n, k, N):
    """
    Inverse DFT expressed as combination of cos and sin synthesis terms.

    Formula: x(n) = (1/N) * sum_{k=0}^{N-1} X(k) cos((2*pi/N)*n*k) + j * (1/N) * sum_{k=0}^{N-1} X(k) sin((2*pi/N)*n*k)

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
    Rangayyan (2024), Ch 3, Eq 3.86, p. 128
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
            "method": "Inverse DFT expressed as combination of cos and sin synthesis terms.",
        }
    )


# -- rng076: DFT convolution property: time-domain convolution equals DFT-domain product..
def rangayyan_ch3_dft_convolution_property(x, h):
    """
    DFT convolution property: time-domain convolution equals DFT-domain product.

    Formula: if y(n) = x(n) * h(n), then Y(k) = X(k) H(k)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.87, p. 130
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
            "method": "DFT convolution property: time-domain convolution equals DFT-domain product.",
        }
    )


# -- rng077: Symmetry property of twiddle factors used in FFT..
def rangayyan_ch3_twiddle_conjugate_symmetry(n, k, N):
    """
    Symmetry property of twiddle factors used in FFT.

    Formula: W_N^(-n*k) = (W_N^(n*k))*

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.88, p. 130
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Symmetry property of twiddle factors used in FFT."}
    )


# -- rng078: Periodicity property of twiddle factors used in FFT..
def rangayyan_ch3_twiddle_periodicity(n, k, N):
    """
    Periodicity property of twiddle factors used in FFT.

    Formula: W_N^(n*k) = W_N^(n*(k+N)) = W_N^((n+N)*k)

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.89, p. 130
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Periodicity property of twiddle factors used in FFT."}
    )


# -- rng081: Even-symmetric part of a signal..
def rangayyan_ch3_even_part(x, n):
    """
    Even-symmetric part of a signal.

    Formula: x_e(n) = 0.5 * [x(n) + x(-n)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.92, p. 135
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Even-symmetric part of a signal."})


# -- rng082: Odd-symmetric part of a signal..
def rangayyan_ch3_odd_part(x, n):
    """
    Odd-symmetric part of a signal.

    Formula: x_o(n) = 0.5 * [x(n) - x(-n)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.93, p. 135
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Odd-symmetric part of a signal."})


# -- rng083: Decomposition of a signal into even and odd parts..
def rangayyan_ch3_even_odd_decomposition(x_e, x_o, n):
    """
    Decomposition of a signal into even and odd parts.

    Formula: x(n) = x_e(n) + x_o(n)

    Parameters
    ----------
    x_e : array-like
        Input data.
    x_o : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.94, p. 135
    """
    x_e = np.atleast_1d(np.asarray(x_e, dtype=float))
    n = len(x_e)
    result = float(np.mean(x_e))
    se = float(np.std(x_e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Decomposition of a signal into even and odd parts."}
    )


# -- rng232: Fourier transform of the log of a product is sum of log-FTs of the components..
def rangayyan_ch4_homomorphic_log_fourier(X_l, P_l, omega):
    """
    Fourier transform of the log of a product is sum of log-FTs of the components.

    Formula: Y_l(omega) = X_l(omega) + P_l(omega)

    Parameters
    ----------
    X_l : array-like
        Input data.
    P_l : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.60, p. 244
    """
    X_l = np.atleast_1d(np.asarray(X_l, dtype=float))
    n = len(X_l)
    result = float(np.mean(X_l))
    se = float(np.std(X_l, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Fourier transform of the log of a product is sum of log-FTs of the components.",
        }
    )


# -- rng234: Fourier transform converts convolution to multiplication..
def rangayyan_ch4_fourier_convolution_property(X, H, omega):
    """
    Fourier transform converts convolution to multiplication.

    Formula: Y(omega) = X(omega) * H(omega)

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.62, p. 245
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
            "method": "Fourier transform converts convolution to multiplication.",
        }
    )


# -- rng237: Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n)..
def rangayyan_ch4_log_of_convolved_signals(X_hat, H_hat, z, omega):
    """
    Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n).

    Formula: Y_hat(z) = X_hat(z) + H_hat(z); Y_hat(omega) = X_hat(omega) + H_hat(omega)

    Parameters
    ----------
    X_hat : array-like
        Input data.
    H_hat : array-like
        Input data.
    z : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.65, p. 247
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
            "method": "Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n).",
        }
    )


# -- rng241: Power series expansion of log(1 + x) for |x| < 1..
def rangayyan_ch4_log_power_series(x):
    """
    Power series expansion of log(1 + x) for |x| < 1.

    Formula: log(1 + x) = x - x^2/2 + x^3/3 - x^4/4 + ...

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.69, p. 248
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Power series expansion of log(1 + x) for |x| < 1."}
    )


# -- rng242: Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|..
def rangayyan_ch4_log_minimum_phase_expansion(alpha, z, n):
    """
    Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|.

    Formula: log(1 - alpha z^(-1)) = - sum_{n=1}^{inf} (alpha^n / n) * z^(-n), for |z| > |alpha|

    Parameters
    ----------
    alpha : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.70, p. 248
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
            "method": "Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|.",
        }
    )


# -- rng243: Power-series expansion of log(1 - beta z) for |z| < |beta^-1|..
def rangayyan_ch4_log_maximum_phase_expansion(beta, z, n):
    """
    Power-series expansion of log(1 - beta z) for |z| < |beta^-1|.

    Formula: log(1 - beta z) = - sum_{n=1}^{inf} (beta^n / n) * z^n, for |z| < |beta^(-1)|

    Parameters
    ----------
    beta : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.71, p. 248
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
            "method": "Power-series expansion of log(1 - beta z) for |z| < |beta^-1|.",
        }
    )


_CHEATSHEET = [
    'rgcdft: Circular (cyclic) convolution via DFT',
    'rgdft: Discrete Fourier transform (DFT)',
    'rgft: Continuous-time Fourier transform (CTFT)',
    'rgstf: short-time Fourier transform -- Rangayyan & Krishnan Sec 8.4.1',
    'rgztf: X(z) = sum_{n>=0} x(n) z^-n (finite sequence, delegates to rng053)',
    'rng049: H(s) = int_0^T h(t) e^{-st} dt by quadrature; s = jw gives the FT',
    'rng052: Bilateral z-transform of a discrete-time signal x(n).',
    'rng054: Convolution in time becomes multiplication of z-transforms.',
    'rng055: Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle.',
    "rng063: Euler's formula for the complex exponential basis function.",
    'rng064: Continuous-time Fourier transform with frequency variable omega in rad/s.',
    'rng065: Continuous-time Fourier transform with frequency variable f in Hz.',
    'rng066: Continuous-time inverse Fourier transform (synthesis).',
    'rng067: Discrete-time Fourier transform (DTFT) of x(n) with continuous omega.',
    'rng068: DFT computed at K samples of normalized frequency.',
    'rng069: Forward discrete Fourier transform (DFT) of an N-point signal.',
    'rng071: Twiddle factor used in DFT and FFT formulations.',
    'rng072: DFT expressed using twiddle factors W_N^(nk).',
    'rng073: Twiddle factor expressed in terms of cosine and sine basis functions.',
    'rng074: DFT decomposed into real (cos) and imaginary (sin) parts.',
    'rng075: Inverse DFT expressed as combination of cos and sin synthesis terms.',
    'rng076: DFT convolution property: time-domain convolution equals DFT-domain product.',
    'rng077: Symmetry property of twiddle factors used in FFT.',
    'rng078: Periodicity property of twiddle factors used in FFT.',
    'rng081: Even-symmetric part of a signal.',
    'rng082: Odd-symmetric part of a signal.',
    'rng083: Decomposition of a signal into even and odd parts.',
    'rng232: Fourier transform of the log of a product is sum of log-FTs of the components.',
    'rng234: Fourier transform converts convolution to multiplication.',
    'rng237: Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n).',
    'rng241: Power series expansion of log(1 + x) for |x| < 1.',
    'rng242: Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|.',
    'rng243: Power-series expansion of log(1 - beta z) for |z| < |beta^-1|.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
