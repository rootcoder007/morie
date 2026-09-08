# morie.fn -- bsaxfrm (rootcoder007/morie)
"""Transforms: Fourier, DFT/IDFT, DTFT, z, Laplace, and their properties.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 33
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2 as _atan2, cos, fsum, log, pi, sin
from . import _array_core as np
from ._rgcore import aslist, gridint
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from .bsafilt import rangayyan_ch3_z_transform_fir

__all__ = [
    'circconv',
    'rangayyan_circular_conv_dft',
    'dftx',
    'rangayyan_dft',
    'fourier',
    'rangayyan_fourier_transform',
    'rangayyan_stft',
    'rangayyan_z_transform',
    'rangayyan_ch3_laplace_transform_causal_finite',
    'ztrans',
    'rangayyan_ch3_z_transform_definition',
    'ztconv',
    'rangayyan_ch3_z_transform_convolution',
    'dtftz',
    'rangayyan_ch3_dtft_via_z',
    'euler',
    'rangayyan_ch3_complex_exponential',
    'ctft',
    'rangayyan_ch3_fourier_transform_omega',
    'ctftf',
    'rangayyan_ch3_fourier_transform_f',
    'ictft',
    'rangayyan_ch3_inverse_fourier_transform',
    'dtft',
    'rangayyan_ch3_dtft',
    'dftk',
    'rangayyan_ch3_dft_K_samples',
    'dft',
    'rangayyan_ch3_dft_definition',
    'twiddle',
    'rangayyan_ch3_twiddle_factor',
    'dfttw',
    'rangayyan_ch3_dft_via_twiddle',
    'twidcs',
    'rangayyan_ch3_twiddle_cos_sin',
    'dftri',
    'rangayyan_ch3_dft_real_imag_decomposition',
    'idftri',
    'rangayyan_ch3_idft_real_imag',
    'dftconv',
    'rangayyan_ch3_dft_convolution_property',
    'twidconj',
    'rangayyan_ch3_twiddle_conjugate_symmetry',
    'twidper',
    'rangayyan_ch3_twiddle_periodicity',
    'evenpart',
    'rangayyan_ch3_even_part',
    'oddpart',
    'rangayyan_ch3_odd_part',
    'evenodd',
    'rangayyan_ch3_even_odd_decomposition',
    'logft',
    'rangayyan_ch4_homomorphic_log_fourier',
    'ftconv',
    'rangayyan_ch4_fourier_convolution_property',
    'clogsum',
    'rangayyan_ch4_log_of_convolved_signals',
    'logseries',
    'rangayyan_ch4_log_power_series',
    'logminph',
    'rangayyan_ch4_log_minimum_phase_expansion',
    'logmaxph',
    'rangayyan_ch4_log_maximum_phase_expansion',
    'rangayyandft',
]

def _angle(z):
    """Principal argument in (-pi, pi], without importing cmath."""
    return _atan2(z.imag, z.real)



# -- rgcdft: Circular (cyclic) convolution via DFT.
def circconv(x, h, npoints=None):
    """Circular (periodic) convolution, directly and via the DFT.

    Rangayyan (2024) eq. (3.90):
        y_p(n) = sum_{k=0}^{N-1} x_p(k) h_p[(n - k) mod N].

    The book defines this only for periodic signals of the SAME period,
    and warns (Figures 3.40-3.42) that using it in place of linear
    convolution gives wrong results.  Both routes to it are computed --
    the modular sum above, and the inverse DFT of X(k)H(k) -- and their
    agreement is the content of eq. (3.87) at equal lengths.

    Parameters
    ----------
    x, h : array-like
        One period of each signal.
    npoints : int, optional
        Common period N.  Defaults to the longer of the two inputs, with
        the shorter zero-padded; the book requires equal periods, so a
        shorter N than either input is rejected rather than truncated.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    n = max(len(xs), len(hs)) if npoints is None else int(npoints)
    if n < max(len(xs), len(hs)):
        raise ValueError("N must be at least the length of both signals")
    xp = xs + [0.0] * (n - len(xs))
    hp = hs + [0.0] * (n - len(hs))
    direct = [fsum(xp[k] * hp[(i - k) % n] for k in range(n))
              for i in range(n)]
    via = idftri([a * b for a, b in
                  zip(dft(xp)["X"], dft(hp)["X"])])["x"]
    gap = max(abs(a - b) for a, b in zip(direct, via))
    lin_len = len(xs) + len(hs) - 1
    return RichResult(payload={
        "y": direct, "via_dft": via, "N": n, "max_difference": gap,
        "agrees": gap <= 1e-8 * (1 + max(abs(v) for v in direct)),
        "equals_linear": n >= lin_len,
        "linear_length": lin_len,
        "method": "Rangayyan (2024) eq. (3.90)"})


rangayyan_circular_conv_dft = circconv  # pre-policy spelling


# -- rgdft: Discrete Fourier transform (DFT).
def dftx(x, fs=1.0):
    """DFT with the frequency axis in Hz.

    Rangayyan (2024) eq. (3.80) for the transform itself; the frequency
    of bin k is k fs / N, and the book's Figure 3.38 note applies -- for
    even N there are two unique real-valued bins, DC at k = 0 and the
    folding frequency fs/2 at k = N/2, with the remaining N-2 bins in
    complex-conjugate pairs.
    """
    r = dft(x)
    n = r["n"]
    freqs = [k * float(fs) / n for k in range(n)]
    out = dict(r)
    out["freqs"] = freqs
    out["fs"] = float(fs)
    out["folding_frequency"] = float(fs) / 2.0
    out["unique_bins"] = n // 2 + 1
    return RichResult(payload=out)


rangayyan_dft = dftx  # pre-policy spelling


# -- rgft: Continuous-time Fourier transform (CTFT).
def fourier(x, t=None, omega=None, f=None, dt=None):
    """Continuous-time Fourier transform (CTFT).

    Rangayyan (2024) eqs. (3.75)-(3.76).  Alternative spelling of
    :func:`ctft`, kept because "Fourier transform" is what the book's
    Section 3.4.4 calls it; the two share one implementation so they can
    never drift apart.
    """
    return ctft(x, t=t, omega=omega, f=f, dt=dt)


rangayyan_fourier_transform = fourier  # pre-policy spelling


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
def ztrans(x, z=None, n0=0):
    """z-transform of a discrete-time signal, evaluated at given z.

    Rangayyan (2024) eq. (3.54):
        X(z) = sum_{n=-inf}^{inf} x(n) z^(-n),

    with eq. (3.55) as the causal FIR special case, X(z) = sum_{n=0}^{N-1}
    h(n) z^(-n), which is the transfer function of the system.

    Parameters
    ----------
    x : array-like
        Samples x(n0), x(n0+1), ..., x(n0+N-1).
    z : complex or sequence of complex, optional
        Where to evaluate.  With None only the coefficients and the
        implied region information are returned.
    n0 : int
        Index of the first sample.  Nonzero n0 makes the sequence
        two-sided, which is why the sum in eq. (3.54) runs over all n and
        not from zero; the book's eq. (3.55) is the n0 = 0 case.

    Notes
    -----
    A finite-length sequence converges everywhere except possibly at
    z = 0 (positive powers of z^-1) and z = inf (negative powers), so no
    region of convergence has to be supplied.  z = 0 is rejected when the
    sequence has any sample at n > 0, since z^(-n) is then a pole.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    idx = [n0 + i for i in range(len(xs))]
    out = {"coefficients": xs, "n": idx, "causal": n0 >= 0,
           "degree": len(xs) - 1,
           "method": "Rangayyan (2024) eqs. (3.54)-(3.55)"}
    if z is None:
        out["X"] = None
        out["z"] = None
        return RichResult(payload=out)
    scalar = isinstance(z, (int, float, complex))
    zs = [complex(z)] if scalar else [complex(v) for v in z]
    vals = []
    for zv in zs:
        if zv == 0 and any(i > 0 for i in idx):
            raise ValueError("z = 0 is a pole of this sequence")
        vals.append(sum(complex(c) * zv ** (-i) for c, i in zip(xs, idx)))
    out["X"] = vals[0] if scalar else vals
    out["z"] = zs[0] if scalar else zs
    return RichResult(payload=out)


rangayyan_ch3_z_transform_definition = ztrans  # pre-policy spelling


# -- rng054: Convolution in time becomes multiplication of z-transforms..
def ztconv(x, h, z):
    """Convolution in time is multiplication of z-transforms.

    Rangayyan (2024) eq. (3.56):
        if y(n) = x(n) * h(n), then Y(z) = X(z) H(z).

    The property is the reason LSI cascades multiply, so it is worth
    checking rather than assuming: both sides are computed independently
    -- Y(z) from the convolved sequence, X(z)H(z) from the factors -- and
    the largest discrepancy is returned.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both sequences need at least one sample")
    scalar = isinstance(z, (int, float, complex))
    zs = [complex(z)] if scalar else [complex(v) for v in z]
    y = []
    for k in range(len(xs) + len(hs) - 1):
        lo = max(0, k - len(hs) + 1)
        hi = min(k, len(xs) - 1)
        y.append(sum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    lhs, rhs = [], []
    for zv in zs:
        if zv == 0:
            raise ValueError("z = 0 is a pole of a causal sequence")
        X = sum(complex(c) * zv ** (-i) for i, c in enumerate(xs))
        H = sum(complex(c) * zv ** (-i) for i, c in enumerate(hs))
        Y = sum(complex(c) * zv ** (-i) for i, c in enumerate(y))
        lhs.append(Y)
        rhs.append(X * H)
    gap = max(abs(a - b) for a, b in zip(lhs, rhs))
    scale = max(abs(b) for b in rhs) or 1.0
    return RichResult(payload={
        "y": y, "Y": lhs[0] if scalar else lhs,
        "XH": rhs[0] if scalar else rhs,
        "z": zs[0] if scalar else zs,
        "max_difference": gap, "holds": gap <= 1e-9 * scale,
        "method": "Rangayyan (2024) eq. (3.56)"})


rangayyan_ch3_z_transform_convolution = ztconv  # pre-policy spelling


# -- rng055: Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle..
def dtftz(x, omega, fs=None):
    """Fourier transform as the z-transform on the unit circle.

    Rangayyan (2024) eq. (3.66):
        X(omega) = sum_{n=0}^{N-1} x(n) z^(-n) |_{z = exp(j omega T)}
                 = sum_{n=0}^{N-1} x(n) exp(-j omega n T).

    The book then argues T may be dropped by working in normalized
    frequency f/fs in [0, 1]; passing ``fs`` keeps omega in rad/s and
    supplies T = 1/fs, while leaving ``fs`` as None reads omega as
    already normalized (T = 1).  Both the unit-circle point z and the
    transform value are returned, because the whole content of the
    equation is that they are the same computation.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    t_s = 1.0 / float(fs) if fs else 1.0
    scalar = isinstance(omega, (int, float))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    zs, vals = [], []
    for w in ws:
        ang = w * t_s
        zv = complex(cos(ang), sin(ang))          # exp(j omega T)
        zs.append(zv)
        vals.append(sum(complex(c) * zv ** (-n) for n, c in enumerate(xs)))
    return RichResult(payload={
        "X": vals[0] if scalar else vals,
        "z": zs[0] if scalar else zs,
        "omega": ws[0] if scalar else ws,
        "T": t_s, "n": len(xs),
        "on_unit_circle": all(abs(abs(v) - 1.0) < 1e-12 for v in zs),
        "method": "Rangayyan (2024) eq. (3.66)"})


rangayyan_ch3_dtft_via_z = dtftz  # pre-policy spelling


# -- rng063: Euler's formula for the complex exponential basis function..
def euler(omega, t=0.0):
    """Complex exponential basis function of the Fourier transform.

    Rangayyan (2024) eq. (3.74):
        exp(j omega t) = cos(omega t) + j sin(omega t).

    This is the basis function the transform projects onto, which is why
    the book introduces it immediately before eqs. (3.75)-(3.76); the
    real and imaginary parts are returned separately so the projection
    onto the cos and sin components can be read off directly.
    """
    ws = [float(omega)] if isinstance(omega, (int, float)) else \
        [float(v) for v in omega]
    ts = [float(t)] if isinstance(t, (int, float)) else [float(v) for v in t]
    if len(ws) > 1 and len(ts) > 1 and len(ws) != len(ts):
        raise ValueError("omega and t must broadcast: equal lengths or one "
                         "of them scalar")
    n = max(len(ws), len(ts))
    ws = ws * n if len(ws) == 1 else ws
    ts = ts * n if len(ts) == 1 else ts
    ang = [w * tv for w, tv in zip(ws, ts)]
    re = [cos(a) for a in ang]
    im = [sin(a) for a in ang]
    vals = [complex(r, i) for r, i in zip(re, im)]
    one = len(vals) == 1
    return RichResult(payload={
        "value": vals[0] if one else vals,
        "real": re[0] if one else re, "imag": im[0] if one else im,
        "angle": ang[0] if one else ang,
        "unit_modulus": all(abs(abs(v) - 1.0) < 1e-15 for v in vals),
        "method": "Rangayyan (2024) eq. (3.74)"})


rangayyan_ch3_complex_exponential = euler  # pre-policy spelling


# -- rng064: Continuous-time Fourier transform with frequency variable omega in rad/s..
def ctft(x, t=None, omega=None, f=None, dt=None):
    """Continuous-time Fourier transform of a tabulated signal.

    Rangayyan (2024) eqs. (3.75)-(3.76):
        X(omega) = integral x(t) exp(-j omega t) dt          (rad/s)
        X(f)     = integral x(t) exp(-j 2 pi f t) dt         (Hz)

    They are one transform in two frequency variables, omega = 2 pi f, so
    they are one function here: give ``omega`` or give ``f``.  The
    integral is evaluated by the trapezoidal rule over the supplied
    samples, so the limits are the duration of the signal -- which the
    book notes is exactly what the limits become for a finite-duration
    signal.

    Parameters
    ----------
    x : array-like
        Samples of x(t).
    t : array-like, optional
        Sample times.  Defaults to a uniform grid of spacing ``dt``.
    omega : float or sequence, optional
        Frequencies in rad/s.
    f : float or sequence, optional
        Frequencies in Hz.  Exactly one of omega, f must be given.
    dt : float, optional
        Sampling interval when ``t`` is not given (default 1).
    """
    xs = aslist(x)
    if len(xs) < 2:
        raise ValueError("need at least two samples to integrate")
    if (omega is None) == (f is None):
        raise ValueError("give exactly one of omega, f")
    step = 1.0 if dt is None else float(dt)
    ts = [i * step for i in range(len(xs))] if t is None else aslist(t)
    if len(ts) != len(xs):
        raise ValueError("t and x must have the same length")
    if omega is not None:
        scalar = isinstance(omega, (int, float))
        ws = [float(omega)] if scalar else [float(v) for v in omega]
        fs_ = [w / (2.0 * pi) for w in ws]
        variable = "omega"
    else:
        scalar = isinstance(f, (int, float))
        fs_ = [float(f)] if scalar else [float(v) for v in f]
        ws = [2.0 * pi * v for v in fs_]
        variable = "f"
    vals = []
    for w in ws:
        re = [xv * cos(-w * tv) for xv, tv in zip(xs, ts)]
        im = [xv * sin(-w * tv) for xv, tv in zip(xs, ts)]
        vals.append(complex(gridint(re, ts), gridint(im, ts)))
    return RichResult(payload={
        "X": vals[0] if scalar else vals,
        "omega": ws[0] if scalar else ws,
        "f": fs_[0] if scalar else fs_,
        "variable": variable, "duration": ts[-1] - ts[0],
        "method": "Rangayyan (2024) eqs. (3.75)-(3.76)"})


rangayyan_ch3_fourier_transform_omega = ctft  # pre-policy spelling


# -- rng065: Continuous-time Fourier transform with frequency variable f in Hz..
def ctftf(x, f, t=None, dt=None):
    """Continuous-time Fourier transform in Hz.

    Rangayyan (2024) eq. (3.76):
        X(f) = integral x(t) exp(-j 2 pi f t) dt.

    Same transform as eq. (3.75) with omega = 2 pi f, so this is a thin
    spelling of :func:`ctft` rather than a second implementation -- the
    book presents both because the Hz form avoids the 1/(2 pi) factor in
    the inverse transform of eq. (3.77).
    """
    return ctft(x, t=t, f=f, dt=dt)


rangayyan_ch3_fourier_transform_f = ctftf  # pre-policy spelling


# -- rng066: Continuous-time inverse Fourier transform (synthesis)..
def ictft(X, t, omega=None, f=None):
    """Inverse continuous-time Fourier transform (synthesis).

    Rangayyan (2024) eq. (3.77):
        x(t) = (1/(2 pi)) integral X(omega) exp(+j omega t) d omega
             = integral X(f) exp(+j 2 pi f t) df.

    The 1/(2 pi) belongs to the omega form only; the Hz form has no such
    factor, which the book gives as the reason to prefer it.  Getting
    that factor wrong is the usual way an inverse transform comes out
    scaled by 6.28, so the branch is explicit here rather than folded
    into a shared constant.

    Parameters
    ----------
    X : array-like of complex
        Spectrum sampled at ``omega`` or ``f``.
    t : float or sequence
        Times at which to synthesize.
    omega, f : array-like
        Frequency grid of the spectrum, in rad/s or Hz.  Exactly one.
    """
    Xs = [complex(v) for v in X]
    if (omega is None) == (f is None):
        raise ValueError("give exactly one of omega, f")
    if omega is not None:
        grid = [float(v) for v in omega]
        scale = 1.0 / (2.0 * pi)
        ang = lambda w, tv: w * tv
        variable = "omega"
    else:
        grid = [float(v) for v in f]
        scale = 1.0
        ang = lambda w, tv: 2.0 * pi * w * tv
        variable = "f"
    if len(grid) != len(Xs):
        raise ValueError("X and the frequency grid must have equal length")
    if len(grid) < 2:
        raise ValueError("need at least two frequency points to integrate")
    scalar = isinstance(t, (int, float))
    ts = [float(t)] if scalar else [float(v) for v in t]
    out = []
    for tv in ts:
        re = [Xs[i].real * cos(ang(grid[i], tv))
              - Xs[i].imag * sin(ang(grid[i], tv)) for i in range(len(Xs))]
        im = [Xs[i].real * sin(ang(grid[i], tv))
              + Xs[i].imag * cos(ang(grid[i], tv)) for i in range(len(Xs))]
        out.append(complex(scale * gridint(re, grid),
                           scale * gridint(im, grid)))
    return RichResult(payload={
        "x": out[0] if scalar else out, "t": ts[0] if scalar else ts,
        "variable": variable, "scale": scale,
        "method": "Rangayyan (2024) eq. (3.77)"})


rangayyan_ch3_inverse_fourier_transform = ictft  # pre-policy spelling


# -- rng067: Discrete-time Fourier transform (DTFT) of x(n) with continuous omega..
def dtft(x, omega, n0=0):
    """Discrete-time Fourier transform, continuous in frequency.

    Rangayyan (2024) eq. (3.78):
        X(omega) = sum_{n} x(n) exp(-j omega n),

    over the normalized range 0 <= omega <= 2 pi (equivalently
    0 <= f <= 1).  The signal is discrete but the frequency variable is
    not -- that is the whole distinction from the DFT of eq. (3.80),
    which samples this function at N points.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    scalar = isinstance(omega, (int, float))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    vals = []
    for w in ws:
        re = fsum(xv * cos(-w * (n0 + i)) for i, xv in enumerate(xs))
        im = fsum(xv * sin(-w * (n0 + i)) for i, xv in enumerate(xs))
        vals.append(complex(re, im))
    return RichResult(payload={
        "X": vals[0] if scalar else vals,
        "omega": ws[0] if scalar else ws, "n0": int(n0), "n": len(xs),
        "method": "Rangayyan (2024) eq. (3.78)"})


rangayyan_ch3_dtft = dtft  # pre-policy spelling


# -- rng068: DFT computed at K samples of normalized frequency..
def dftk(x, k_points):
    """DFT sampled at K points of the normalized frequency axis.

    Rangayyan (2024) eq. (3.79):
        X(k) = sum_{n=0}^{N-1} x(n) exp(-j (2 pi / K) n k),
        k = 0, 1, ..., K-1.

    The book's point is that K need not equal N: K > N samples the same
    underlying DTFT more finely (the zero-padding case), and the text
    then argues that K = N already suffices for exact recovery, which is
    eq. (3.80).  ``aliased`` flags K < N, where samples fold and the
    signal cannot be recovered.
    """
    xs = aslist(x)
    n = len(xs)
    if not n:
        raise ValueError("need at least one sample")
    kk = int(k_points)
    if kk < 1:
        raise ValueError("K must be positive")
    step = 2.0 * pi / kk
    out = []
    for k in range(kk):
        re = fsum(xv * cos(-step * i * k) for i, xv in enumerate(xs))
        im = fsum(xv * sin(-step * i * k) for i, xv in enumerate(xs))
        out.append(complex(re, im))
    return RichResult(payload={
        "X": out, "K": kk, "n": n, "aliased": kk < n,
        "method": "Rangayyan (2024) eq. (3.79)"})


rangayyan_ch3_dft_K_samples = dftk  # pre-policy spelling


# -- rng069: Forward discrete Fourier transform (DFT) of an N-point signal..
def dft(x):
    """Discrete Fourier transform of an N-point signal.

    Rangayyan (2024) eq. (3.80):
        X(k) = sum_{n=0}^{N-1} x(n) exp(-j (2 pi / N) n k),
        k = 0, 1, ..., N-1.

    Evaluated directly from the definition -- O(N^2), exact at any N, and
    with no requirement that N be a power of two.  Eq. (3.85) splits the
    same sum into its cos and sin parts, which are returned here as
    ``real`` and ``imag`` so the two spellings never disagree.

    For a real-valued signal the book notes X(-k) = X*(k), so the real
    part and magnitude are even-symmetric and the imaginary part and
    phase odd-symmetric; ``conjugate_symmetric`` records whether that
    holds, which is False for a complex input.
    """
    xs = aslist(x)
    n = len(xs)
    if not n:
        raise ValueError("need at least one sample")
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(xv * cos(-step * i * k) for i, xv in enumerate(xs)))
        im.append(fsum(xv * sin(-step * i * k) for i, xv in enumerate(xs)))
    X = [complex(a, b) for a, b in zip(re, im)]
    sym = all(abs(X[k] - X[(n - k) % n].conjugate()) < 1e-9 * (1 + abs(X[k]))
              for k in range(n))
    return RichResult(payload={
        "X": X, "real": re, "imag": im, "n": n,
        "magnitude": [abs(v) for v in X],
        "conjugate_symmetric": sym,
        "method": "Rangayyan (2024) eq. (3.80)"})


rangayyan_ch3_dft_definition = dft  # pre-policy spelling


# -- rng071: Twiddle factor used in DFT and FFT formulations..
def twiddle(npoints, power=1):
    """Twiddle factor W_N and its powers.

    Rangayyan (2024) eq. (3.82):
        W_N = exp(-j 2 pi / N),

    the N-th root of unity that the DFT of eq. (3.83) is written in terms
    of.  The book's Figure 3.34 plots W_8^k for k = 0..7 as phasors on
    the unit circle; ``power`` selects which of those to return.
    """
    n = int(npoints)
    if n < 1:
        raise ValueError("N must be positive")
    scalar = isinstance(power, int)
    ps = [int(power)] if scalar else [int(v) for v in power]
    vals = [complex(cos(-2.0 * pi * p / n), sin(-2.0 * pi * p / n))
            for p in ps]
    return RichResult(payload={
        "W": vals[0] if scalar else vals, "N": n,
        "power": ps[0] if scalar else ps,
        "root_of_unity": abs(vals[0] ** n - 1.0) < 1e-9 if scalar else None,
        "method": "Rangayyan (2024) eq. (3.82)"})


rangayyan_ch3_twiddle_factor = twiddle  # pre-policy spelling


# -- rng072: DFT expressed using twiddle factors W_N^(nk)..
def dfttw(x):
    """DFT written with twiddle factors.

    Rangayyan (2024) eq. (3.83):
        X(k) = sum_{n=0}^{N-1} x(n) W_N^(n k),   W_N = exp(-j 2 pi / N).

    The same transform as eq. (3.80); computing it this way makes the
    root-of-unity structure explicit, which is what the FFT exploits via
    eqs. (3.88)-(3.89).  Computed by accumulating powers of W_N and then
    checked against :func:`dft`, so the two forms are shown to agree
    rather than assumed to.
    """
    xs = aslist(x)
    n = len(xs)
    if not n:
        raise ValueError("need at least one sample")
    w = complex(cos(-2.0 * pi / n), sin(-2.0 * pi / n))
    X = []
    for k in range(n):
        acc = 0j
        wk = 1.0 + 0j
        step = w ** k
        for xv in xs:
            acc += xv * wk
            wk *= step
        X.append(acc)
    direct = dft(xs)["X"]
    gap = max(abs(a - b) for a, b in zip(X, direct))
    return RichResult(payload={
        "X": X, "W": w, "n": n, "max_difference": gap,
        "agrees_with_definition": gap <= 1e-8 * (1 + max(abs(v)
                                                         for v in direct)),
        "method": "Rangayyan (2024) eq. (3.83)"})


rangayyan_ch3_dft_via_twiddle = dfttw  # pre-policy spelling


# -- rng073: Twiddle factor expressed in terms of cosine and sine basis functions..
def twidcs(npoints, n, k):
    """Twiddle factor split into its cos and sin basis functions.

    Rangayyan (2024) eq. (3.84):
        W_N^(n k) = exp(-j (2 pi / N) n k)
                  = cos((2 pi / N) n k) - j sin((2 pi / N) n k).

    Note the minus sign on the sine: the DFT projects onto the
    *conjugated* exponential, so the imaginary part of X(k) is minus the
    dot product of the signal with the sine, not plus it (eq. 3.85).
    That sign is the single most common transcription error in a
    hand-written DFT, so both components are returned separately.
    """
    nn = int(npoints)
    if nn < 1:
        raise ValueError("N must be positive")
    ang = 2.0 * pi * int(n) * int(k) / nn
    c, s = cos(ang), sin(ang)
    return RichResult(payload={
        "W": complex(c, -s), "cos": c, "sin": s, "angle": ang,
        "N": nn, "n": int(n), "k": int(k),
        "method": "Rangayyan (2024) eq. (3.84)"})


rangayyan_ch3_twiddle_cos_sin = twidcs  # pre-policy spelling


# -- rng074: DFT decomposed into real (cos) and imaginary (sin) parts..
def dftri(x):
    """DFT as dot products with the cos and sin basis functions.

    Rangayyan (2024) eq. (3.85):
        X(k) = sum x(n) cos((2 pi/N) n k) - j sum x(n) sin((2 pi/N) n k).

    The book reads the real part as the projection of the signal onto the
    k-th cosine basis function and the imaginary part as its projection
    onto the corresponding sine -- the DFT coefficient measures how much
    of each sinusoid is present.  Those two projections are returned
    separately as ``cos_projection`` and ``sin_projection``; note the
    imaginary part is MINUS the sine projection.
    """
    xs = aslist(x)
    n = len(xs)
    if not n:
        raise ValueError("need at least one sample")
    step = 2.0 * pi / n
    cp, sp = [], []
    for k in range(n):
        cp.append(fsum(xv * cos(step * i * k) for i, xv in enumerate(xs)))
        sp.append(fsum(xv * sin(step * i * k) for i, xv in enumerate(xs)))
    X = [complex(a, -b) for a, b in zip(cp, sp)]
    return RichResult(payload={
        "X": X, "cos_projection": cp, "sin_projection": sp,
        "real": cp, "imag": [-b for b in sp], "n": n,
        "method": "Rangayyan (2024) eq. (3.85)"})


rangayyan_ch3_dft_real_imag_decomposition = dftri  # pre-policy spelling


# -- rng075: Inverse DFT expressed as combination of cos and sin synthesis terms..
def idftri(X):
    """Inverse DFT as a weighted sum of sinusoids.

    Rangayyan (2024) eq. (3.86):
        x(n) = (1/N) sum X(k) cos((2 pi/N) n k)
             + j (1/N) sum X(k) sin((2 pi/N) n k).

    Synthesis: the signal is rebuilt as a weighted combination of
    sinusoids whose weights are the DFT coefficients.  For a spectrum
    that came from a real signal the imaginary parts cancel; the residual
    is returned as ``max_imaginary`` rather than silently discarded,
    since a large residual means the spectrum was not conjugate-symmetric
    and the "real signal" reading does not apply.
    """
    Xs = [complex(v) for v in X]
    n = len(Xs)
    if not n:
        raise ValueError("need at least one coefficient")
    step = 2.0 * pi / n
    out = []
    for i in range(n):
        acc = 0j
        for k, Xk in enumerate(Xs):
            ang = step * i * k
            acc += Xk * complex(cos(ang), sin(ang))
        out.append(acc / n)
    return RichResult(payload={
        "x": [v.real for v in out], "complex": out, "n": n,
        "max_imaginary": max(abs(v.imag) for v in out),
        "method": "Rangayyan (2024) eq. (3.86)"})


rangayyan_ch3_idft_real_imag = idftri  # pre-policy spelling


# -- rng076: DFT convolution property: time-domain convolution equals DFT-domain product..
def dftconv(x, h):
    """DFT convolution property, and the linear-convolution caveat.

    Rangayyan (2024) eq. (3.87):
        if y(n) = x(n) * h(n), then Y(k) = X(k) H(k).

    The book is explicit that the convolution in this relationship is
    PERIODIC, not linear -- every sequence in a DFT relationship is
    periodic.  So multiplying N-point DFTs gives the circular
    convolution of eq. (3.90), and recovering the linear convolution
    requires transforms of length L >= Nx + Nh - 1 with both sequences
    zero-padded to L.  Both are computed here: ``circular`` at length
    max(Nx, Nh) and ``linear`` at the padded length, with the padded
    length reported so the difference is visible rather than a silent
    wrap-around error.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both sequences need at least one sample")
    nx, nh = len(xs), len(hs)
    lin = []
    for k in range(nx + nh - 1):
        lo, hi = max(0, k - nh + 1), min(k, nx - 1)
        lin.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    L = nx + nh - 1
    xp = xs + [0.0] * (L - nx)
    hp = hs + [0.0] * (L - nh)
    Xp, Hp = dft(xp)["X"], dft(hp)["X"]
    rec = idftri([a * b for a, b in zip(Xp, Hp)])["x"]
    n = max(nx, nh)
    xc = xs + [0.0] * (n - nx)
    hc = hs + [0.0] * (n - nh)
    circ = [fsum(xc[i] * hc[(k - i) % n] for i in range(n)) for k in range(n)]
    gap = max(abs(a - b) for a, b in zip(rec, lin))
    return RichResult(payload={
        "linear": lin, "circular": circ, "from_dft": rec,
        "padded_length": L, "n_linear": L, "n_circular": n,
        "max_difference": gap,
        "holds": gap <= 1e-8 * (1 + max(abs(v) for v in lin)),
        "wraps_if_unpadded": n < L,
        "method": "Rangayyan (2024) eq. (3.87)"})


rangayyan_ch3_dft_convolution_property = dftconv  # pre-policy spelling


# -- rng077: Symmetry property of twiddle factors used in FFT..
def twidconj(npoints, n, k):
    """Conjugate symmetry of the twiddle factor.

    Rangayyan (2024) eq. (3.88):
        W_N^(-n k) = (W_N^(n k))*.

    One of the two properties the FFT is built on: a negative power costs
    nothing beyond a sign flip on the imaginary part.  Both sides are
    formed independently and compared.
    """
    nn = int(npoints)
    if nn < 1:
        raise ValueError("N must be positive")
    p = int(n) * int(k)
    lhs = complex(cos(2.0 * pi * p / nn), sin(2.0 * pi * p / nn))
    rhs = complex(cos(-2.0 * pi * p / nn), sin(-2.0 * pi * p / nn)).conjugate()
    return RichResult(payload={
        "negative_power": lhs, "conjugate": rhs,
        "difference": abs(lhs - rhs), "holds": abs(lhs - rhs) < 1e-12,
        "N": nn, "n": int(n), "k": int(k),
        "method": "Rangayyan (2024) eq. (3.88)"})


rangayyan_ch3_twiddle_conjugate_symmetry = twidconj  # pre-policy spelling


# -- rng078: Periodicity property of twiddle factors used in FFT..
def twidper(npoints, n, k):
    """Periodicity of the twiddle factor.

    Rangayyan (2024) eq. (3.89):
        W_N^(n k) = W_N^(n (k + N)) = W_N^((n + N) k).

    The second FFT property: indices may be reduced modulo N, which is
    why the same N-th roots of unity are reused at every stage of the
    decomposition -- and also why every DFT relationship is periodic.
    All three expressions are evaluated separately and compared.
    """
    nn = int(npoints)
    if nn < 1:
        raise ValueError("N must be positive")
    ni, ki = int(n), int(k)

    def w(p):
        return complex(cos(-2.0 * pi * p / nn), sin(-2.0 * pi * p / nn))

    base, shift_k, shift_n = w(ni * ki), w(ni * (ki + nn)), w((ni + nn) * ki)
    gap = max(abs(base - shift_k), abs(base - shift_n))
    return RichResult(payload={
        "base": base, "shift_k": shift_k, "shift_n": shift_n,
        "max_difference": gap, "holds": gap < 1e-9,
        "N": nn, "n": ni, "k": ki,
        "method": "Rangayyan (2024) eq. (3.89)"})


rangayyan_ch3_twiddle_periodicity = twidper  # pre-policy spelling


# -- rng081: Even-symmetric part of a signal..
def evenpart(x, n=None):
    """Even-symmetric part of a signal.

    Rangayyan (2024) eq. (3.92):
        x_e(n) = 0.5 [x(n) + x(-n)].

    x(-n) has to come from somewhere: the index grid must cover both n
    and -n, so either pass ``n`` explicitly or pass a sequence whose
    indices are taken as symmetric about its centre.  Reflecting a
    causal sequence about index 0 instead -- the tempting shortcut --
    computes something else entirely, since x(-n) is then zero for all
    n > 0 and the "even part" collapses to x/2.
    """
    return _evenodd_core(x, n)["even_result"]


rangayyan_ch3_even_part = evenpart  # pre-policy spelling


# -- rng082: Odd-symmetric part of a signal..
def oddpart(x, n=None):
    """Odd-symmetric part of a signal.

    Rangayyan (2024) eq. (3.93):
        x_o(n) = 0.5 [x(n) - x(-n)].

    Odd symmetry forces x_o(0) = 0 at the origin, which is a cheap check
    that the index grid was built correctly.
    """
    return _evenodd_core(x, n)["odd_result"]


rangayyan_ch3_odd_part = oddpart  # pre-policy spelling


# -- rng083: Decomposition of a signal into even and odd parts..
def _evenodd_core(x, n=None):
    xs = aslist(x)
    m = len(xs)
    if not m:
        raise ValueError("need at least one sample")
    if n is None:
        if m % 2 == 0:
            raise ValueError("with no index grid the sequence must have an "
                             "odd length so that n = 0 is a sample; pass n=")
        half = m // 2
        idx = [i - half for i in range(m)]
    else:
        idx = [int(v) for v in n]
        if len(idx) != m:
            raise ValueError("n and x must have the same length")
    table = dict(zip(idx, xs))
    missing = [i for i in idx if -i not in table]
    if missing:
        raise ValueError("index grid is not symmetric: x(-n) is unavailable "
                         "for n = %s" % missing[:5])
    ev = [0.5 * (table[i] + table[-i]) for i in idx]
    od = [0.5 * (table[i] - table[-i]) for i in idx]
    recon = [a + b for a, b in zip(ev, od)]
    err = max(abs(a - b) for a, b in zip(recon, xs))
    common = {"n": idx, "even": ev, "odd": od, "x": xs,
              "reconstruction_error": err}
    return {
        "even_result": RichResult(payload=dict(
            common, method="Rangayyan (2024) eq. (3.92)")),
        "odd_result": RichResult(payload=dict(
            common, method="Rangayyan (2024) eq. (3.93)")),
        "both": RichResult(payload=dict(
            common, method="Rangayyan (2024) eqs. (3.92)-(3.94)")),
    }


def evenodd(x, n=None):
    """Decompose a signal into its even and odd parts.

    Rangayyan (2024) eqs. (3.92)-(3.94):
        x_e(n) = 0.5 [x(n) + x(-n)]
        x_o(n) = 0.5 [x(n) - x(-n)]
        x(n)   = x_e(n) + x_o(n).

    Eq. (3.94) is an identity, so ``reconstruction_error`` is a genuine
    check on the index bookkeeping rather than on the arithmetic: it is
    nonzero only if the grid used for x(-n) does not match the grid used
    for x(n).
    """
    return _evenodd_core(x, n)["both"]


rangayyan_ch3_even_odd_decomposition = evenodd  # pre-policy spelling


# -- rng232: Fourier transform of the log of a product is sum of log-FTs of the components..
def logft(x, p, omega, t=None, dt=None):
    """Fourier transform of the log of a product is a sum of log-spectra.

    Rangayyan (2024) eqs. (4.58)-(4.60), the multiplicative homomorphic
    system:
        y(t)      = x(t) p(t)                                    (4.58)
        log[y(t)] = log[x(t)] + log[p(t)],  x, p nonzero          (4.59)
        Y_l(omega) = X_l(omega) + P_l(omega)                      (4.60)

    with the subscript l marking a transform taken of a log-transformed
    signal.  Eq. (4.59) requires both factors to be nonzero everywhere,
    so zeros are rejected instead of silently producing -inf; and both
    sides of eq. (4.60) are computed independently so the additivity is
    demonstrated on the data given.
    """
    xs, ps = aslist(x), aslist(p)
    if len(xs) != len(ps):
        raise ValueError("x and p must have the same length")
    if any(v == 0 for v in xs) or any(v == 0 for v in ps):
        raise ValueError("eq. (4.59) needs x(t) != 0 and p(t) != 0 for all t")
    if any(v < 0 for v in xs) or any(v < 0 for v in ps):
        raise ValueError("real logarithm needs positive signals; take the "
                         "complex cepstrum route for signed data")
    y = [a * b for a, b in zip(xs, ps)]
    ly = [log(v) for v in y]
    lx = [log(v) for v in xs]
    lp = [log(v) for v in ps]
    Yl = ctft(ly, t=t, omega=omega, dt=dt)["X"]
    Xl = ctft(lx, t=t, omega=omega, dt=dt)["X"]
    Pl = ctft(lp, t=t, omega=omega, dt=dt)["X"]
    scalar = isinstance(omega, (int, float))
    a = [Yl] if scalar else list(Yl)
    b = [Xl] if scalar else list(Xl)
    c = [Pl] if scalar else list(Pl)
    gap = max(abs(u - (v + w)) for u, v, w in zip(a, b, c))
    return RichResult(payload={
        "y": y, "Yl": Yl, "Xl": Xl, "Pl": Pl,
        "max_difference": gap,
        "additive": gap <= 1e-8 * (1 + max(abs(u) for u in a)),
        "method": "Rangayyan (2024) eqs. (4.58)-(4.60)"})


rangayyan_ch4_homomorphic_log_fourier = logft  # pre-policy spelling


# -- rng234: Fourier transform converts convolution to multiplication..
def ftconv(x, h, omega, dt=1.0):
    """Fourier transform turns convolution into multiplication.

    Rangayyan (2024) eqs. (4.61)-(4.62):
        y(t) = x(t) * h(t)      =>      Y(omega) = X(omega) H(omega).

    This is the first step of homomorphic deconvolution: it converts the
    convolution to a product, which eq. (4.63) then converts to a sum by
    taking the complex logarithm.  Both sides are computed from the
    sampled signals -- the convolution is scaled by dt, matching eq.
    (3.30), so the product identity holds in the continuous-time sense
    rather than up to a factor of the sampling interval.
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
    scalar = isinstance(omega, (int, float))
    ws = [float(omega)] if scalar else [float(v) for v in omega]

    def tf(sig):
        vals = []
        for w in ws:
            re = fsum(v * cos(-w * i * step) for i, v in enumerate(sig))
            im = fsum(v * sin(-w * i * step) for i, v in enumerate(sig))
            vals.append(complex(re, im) * step)
        return vals

    Y, X, H = tf(y), tf(xs), tf(hs)
    prod = [a * b for a, b in zip(X, H)]
    gap = max(abs(a - b) for a, b in zip(Y, prod))
    return RichResult(payload={
        "y": y, "Y": Y[0] if scalar else Y, "X": X[0] if scalar else X,
        "H": H[0] if scalar else H, "XH": prod[0] if scalar else prod,
        "max_difference": gap,
        "holds": gap <= 1e-8 * (1 + max(abs(v) for v in prod)),
        "method": "Rangayyan (2024) eqs. (4.61)-(4.62)"})


rangayyan_ch4_fourier_convolution_property = ftconv  # pre-policy spelling


# -- rng237: Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n)..
def clogsum(x, h, z):
    """Complex logarithms of the z-transforms of a convolution add.

    Rangayyan (2024) eq. (4.65):
        Y_hat(z) = X_hat(z) + H_hat(z),
        Y_hat(omega) = X_hat(omega) + H_hat(omega),

    where the hat denotes the complex logarithm, and eq. (4.66) carries
    this to the complex cepstra themselves.  The book's note under eq.
    (4.63) spells out the complex log:
        log[X] = log|X| + j angle(X).

    The catch is the branch: angle() returns a principal value in
    (-pi, pi], so the two sides can differ by an integer multiple of
    2 pi j even when the identity holds.  That is reported as
    ``branch_offset`` (in units of 2 pi) rather than being papered over,
    since it is exactly the phase-unwrapping problem the cepstrum
    literature spends its effort on.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both sequences need at least one sample")
    y = []
    for k in range(len(xs) + len(hs) - 1):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    scalar = isinstance(z, (int, float, complex))
    zs = [complex(z)] if scalar else [complex(v) for v in z]

    def zt(seq, zv):
        return sum(complex(c) * zv ** (-i) for i, c in enumerate(seq))

    Yh, Xh, Hh, off = [], [], [], []
    for zv in zs:
        if zv == 0:
            raise ValueError("z = 0 is a pole of a causal sequence")
        Y, X, H = zt(y, zv), zt(xs, zv), zt(hs, zv)
        if Y == 0 or X == 0 or H == 0:
            raise ValueError("the complex log needs X(z) != 0 and H(z) != 0")
        ly = complex(log(abs(Y)), _angle(Y))
        lx = complex(log(abs(X)), _angle(X))
        lh = complex(log(abs(H)), _angle(H))
        Yh.append(ly)
        Xh.append(lx)
        Hh.append(lh)
        off.append((ly.imag - lx.imag - lh.imag) / (2.0 * pi))
    mag_gap = max(abs(a.real - b.real - c.real)
                  for a, b, c in zip(Yh, Xh, Hh))
    wrap = max(abs(o - round(o)) for o in off)
    return RichResult(payload={
        "y": y, "Y_hat": Yh[0] if scalar else Yh,
        "X_hat": Xh[0] if scalar else Xh,
        "H_hat": Hh[0] if scalar else Hh,
        "magnitude_difference": mag_gap,
        "branch_offset": off[0] if scalar else off,
        "holds_up_to_branch": mag_gap < 1e-9 and wrap < 1e-9,
        "method": "Rangayyan (2024) eqs. (4.63), (4.65)"})


rangayyan_ch4_log_of_convolved_signals = clogsum  # pre-policy spelling


# -- rng241: Power series expansion of log(1 + x) for |x| < 1..
def logseries(x, terms=20):
    """Power series for the logarithm, the identity the cepstrum uses.

    Rangayyan (2024) eq. (4.69):
        log(1 + x) = x - x^2/2 + x^3/3 - x^4/4 + ...,  for |x| < 1.

    The radius of convergence is exactly 1 and the book states the
    condition, so |x| >= 1 is rejected rather than returning a diverging
    partial sum.  The truncation error of an alternating series with
    decreasing terms is bounded by the first omitted term, which is
    returned as ``error_bound``.
    """
    xs = [complex(x)] if isinstance(x, (int, float, complex)) \
        else [complex(v) for v in x]
    k = int(terms)
    if k < 1:
        raise ValueError("terms must be positive")
    bad = [v for v in xs if abs(v) >= 1.0]
    if bad:
        raise ValueError("the series converges only for |x| < 1; got %r"
                         % (bad[0],))
    out, bound = [], []
    for v in xs:
        s = 0j
        p = 1.0 + 0j
        for n in range(1, k + 1):
            p *= v
            s += ((-1) ** (n + 1)) * p / n
        out.append(s)
        bound.append(abs(p * v) / (k + 1))
    exact = [complex(log(abs(1.0 + v)), _angle(1.0 + v)) for v in xs]
    one = len(out) == 1
    return RichResult(payload={
        "value": out[0] if one else out,
        "exact": exact[0] if one else exact,
        "error": max(abs(a - b) for a, b in zip(out, exact)),
        "error_bound": bound[0] if one else bound,
        "terms": k, "method": "Rangayyan (2024) eq. (4.69)"})


rangayyan_ch4_log_power_series = logseries  # pre-policy spelling


# -- rng242: Power-series expansion of log(1 - alpha z^-1) for |z| > |alpha|..
def logminph(alpha, terms=20, z=None):
    """Cepstral expansion of a zero inside the unit circle.

    Rangayyan (2024) eq. (4.70):
        log(1 - alpha z^-1) = - sum_{n=1}^{inf} (alpha^n / n) z^-n,
        valid for |z| > |alpha|.

    Read as a cepstrum, the coefficient of z^-n is the contribution of a
    minimum-phase factor to x_hat(n): -alpha^n/n at POSITIVE quefrency n,
    decaying at least as fast as 1/n.  That one-sidedness is what
    separates minimum-phase from maximum-phase factors (eq. 4.71), and
    it is why the complex cepstrum of a minimum-phase signal is causal.

    Parameters
    ----------
    alpha : complex
        The zero (or pole) location, |alpha| < |z|.
    terms : int
        Number of coefficients to return.
    z : complex, optional
        If given, the truncated series is summed at this z and compared
        against the closed form log(1 - alpha/z).
    """
    a = complex(alpha)
    k = int(terms)
    if k < 1:
        raise ValueError("terms must be positive")
    coeffs = [-(a ** n) / n for n in range(1, k + 1)]
    out = {"coefficients": coeffs, "quefrency": list(range(1, k + 1)),
           "causal": True, "alpha": a,
           "method": "Rangayyan (2024) eq. (4.70)"}
    if z is not None:
        zv = complex(z)
        if abs(zv) <= abs(a):
            raise ValueError("the expansion needs |z| > |alpha|")
        s = sum(c * zv ** (-n) for n, c in enumerate(coeffs, start=1))
        w = 1.0 - a / zv
        exact = complex(log(abs(w)), _angle(w))
        out["value"] = s
        out["exact"] = exact
        out["error"] = abs(s - exact)
        out["z"] = zv
    return RichResult(payload=out)


rangayyan_ch4_log_minimum_phase_expansion = logminph  # pre-policy spelling


# -- rng243: Power-series expansion of log(1 - beta z) for |z| < |beta^-1|..
def logmaxph(beta, terms=20, z=None):
    """Cepstral expansion of a zero outside the unit circle.

    Rangayyan (2024) eq. (4.71):
        log(1 - beta z) = - sum_{n=1}^{inf} (beta^n / n) z^n,
        valid for |z| < |beta^-1|.

    The mirror of eq. (4.70): here the powers of z are POSITIVE, so the
    coefficients sit at negative quefrency and the maximum-phase part of
    the complex cepstrum is anticausal.  A signal with zeros on both
    sides of the unit circle therefore has a two-sided cepstrum, which
    is why homomorphic deconvolution liftering windows are two-sided.
    """
    b = complex(beta)
    k = int(terms)
    if k < 1:
        raise ValueError("terms must be positive")
    coeffs = [-(b ** n) / n for n in range(1, k + 1)]
    out = {"coefficients": coeffs, "quefrency": [-n for n in range(1, k + 1)],
           "causal": False, "beta": b,
           "method": "Rangayyan (2024) eq. (4.71)"}
    if z is not None:
        zv = complex(z)
        if b != 0 and abs(zv) >= 1.0 / abs(b):
            raise ValueError("the expansion needs |z| < 1/|beta|")
        s = sum(c * zv ** n for n, c in enumerate(coeffs, start=1))
        w = 1.0 - b * zv
        exact = complex(log(abs(w)), _angle(w))
        out["value"] = s
        out["exact"] = exact
        out["error"] = abs(s - exact)
        out["z"] = zv
    return RichResult(payload=out)


rangayyan_ch4_log_maximum_phase_expansion = logmaxph  # pre-policy spelling


_CHEATSHEET = [
    'rgcdft: circular convolution via the DFT, Rangayyan eq. (3.90)',
    'rgdft: DFT with a frequency axis in Hz, Rangayyan eq. (3.80)',
    'rgft: continuous-time Fourier transform, eqs. (3.75)-(3.76)',
    'rgstf: Short-time Fourier transform -- Rangayyan & Krishnan Sec 8.4.1.',
    'rgztf: Z-transform of a causal discrete-time sequence.',
    'rng049: Laplace transform of a causal finite-duration h(t) over [0, T].',
    'rng052: z-transform, Rangayyan eqs. (3.54)-(3.55)',
    'rng054: z-domain convolution property, Rangayyan eq. (3.56)',
    'rng055: DTFT as the z-transform on the unit circle, eq. (3.66)',
    'rng063: Euler complex exponential basis, Rangayyan eq. (3.74)',
    'rng064: continuous-time Fourier transform, eqs. (3.75)-(3.76)',
    'rng065: continuous-time Fourier transform in Hz, eq. (3.76)',
    'rng066: inverse continuous-time Fourier transform, eq. (3.77)',
    'rng067: discrete-time Fourier transform, Rangayyan eq. (3.78)',
    'rng068: DFT at K frequency samples, Rangayyan eq. (3.79)',
    'rng069: discrete Fourier transform, Rangayyan eq. (3.80)',
    'rng071: twiddle factor W_N, Rangayyan eq. (3.82)',
    'rng072: DFT via twiddle factors, Rangayyan eq. (3.83)',
    'rng073: twiddle factor in cos/sin form, Rangayyan eq. (3.84)',
    'rng074: DFT as cos/sin projections, Rangayyan eq. (3.85)',
    'rng075: inverse DFT as sinusoid synthesis, Rangayyan eq. (3.86)',
    'rng076: DFT convolution property, Rangayyan eq. (3.87)',
    'rng077: twiddle conjugate symmetry, Rangayyan eq. (3.88)',
    'rng078: twiddle periodicity, Rangayyan eq. (3.89)',
    'rng081: even-symmetric part, Rangayyan eq. (3.92)',
    'rng082: odd-symmetric part, Rangayyan eq. (3.93)',
    'rng083: even/odd decomposition, Rangayyan eqs. (3.92)-(3.94)',
    'rng232: log-spectra add for a product, Rangayyan eqs. (4.58)-(4.60)',
    'rng234: Fourier convolution property, Rangayyan eqs. (4.61)-(4.62)',
    'rng237: complex logs of a convolution add, Rangayyan eq. (4.65)',
    'rng241: log(1+x) power series, Rangayyan eq. (4.69)',
    'rng242: log(1 - alpha z^-1) expansion, Rangayyan eq. (4.70)',
    'rng243: log(1 - beta z) expansion, Rangayyan eq. (4.71)',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)

# Pre-policy run-together spellings.  These were in the lazy
# map but not in the module, so morie.fn.<name> raised
# AttributeError.  Restored rather than dropped, because the
# map is the public flat namespace.
rangayyandft = rangayyan_dft  # pre-policy spelling, kept live
