# morie.fn -- bsafilt (rootcoder007/morie)
"""Filter design and characterization: MA, Hann, derivative, Butterworth, notch, comb, and their responses.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 82
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan, ceil, tan
from math import atan2, cos, exp, fsum, pi, sin
from math import fsum
from math import fsum, log
from math import log10
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import integrate

__all__ = [
    'bwhp',
    'rangayyan_butterworth_hp',
    'bwlp',
    'rangayyan_butterworth_lp',
    'comb',
    'rangayyan_comb_filter',
    'diff1',
    'rangayyan_first_diff',
    'diff2',
    'rangayyan_second_diff',
    'rangayyan_fir_filter',
    'freqresp',
    'rangayyan_freq_response',
    'grpdelay',
    'rangayyan_group_delay',
    'rangayyan_iir_filter',
    'rangayyan_moving_average',
    'notch',
    'rangayyan_notch_filter',
    'osfilt',
    'rangayyan_order_stat_flt',
    'phaseresp',
    'rangayyan_phase_response',
    'sinckern',
    'rangayyan_sinc_kernel',
    'rangayyan_transfer_func_est',
    'blackman',
    'rangayyan_blackman_window',
    'hamming',
    'rangayyan_hamming_window',
    'hannwin',
    'rangayyan_hann_window',
    'windowfn',
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
    'ma8imp',
    'rangayyan_ch3_ma_8point_impulse_response',
    'ma8tf',
    'rangayyan_ch3_ma_8point_transfer_function',
    'ma8fr',
    'rangayyan_ch3_ma_8point_frequency_response',
    'runint',
    'rangayyan_ch3_running_integral_window',
    'runintall',
    'rangayyan_ch3_integral_general',
    'rangayyan_ch3_integral_causal',
    'intft',
    'rangayyan_ch3_fourier_of_integral',
    'intfr',
    'rangayyan_ch3_integrator_frequency_response',
    'intmag',
    'rangayyan_ch3_integrator_magnitude_response',
    'intph',
    'rangayyan_ch3_integrator_phase_response',
    'ma8rec',
    'rangayyan_ch3_ma_8point_recursive',
    'ma8rectf',
    'rangayyan_ch3_ma_8point_recursive_transfer_function',
    'ma8sinc',
    'rangayyan_ch3_ma_8point_sinc_frequency_response',
    'fdiff',
    'rangayyan_ch3_first_difference_operator',
    'fdifftf',
    'rangayyan_ch3_first_difference_transfer_function',
    'fdifffr',
    'rangayyan_ch3_first_difference_frequency_response',
    'fdiffmag',
    'rangayyan_ch3_first_difference_magnitude',
    'fdiffph',
    'rangayyan_ch3_first_difference_phase',
    'cdiff3',
    'rangayyan_ch3_three_point_central_difference',
    'cdiff3tf',
    'rangayyan_ch3_three_point_central_diff_transfer_function',
    'cdiff3mag',
    'rangayyan_ch3_three_point_central_diff_magnitude',
    'cdiff3ph',
    'rangayyan_ch3_three_point_central_diff_phase',
    'bwander',
    'rangayyan_ch3_baseline_wander_filter_z_form_a',
    'bwanderz',
    'rangayyan_ch3_baseline_wander_filter_z_form_b',
    'bwandereq',
    'rangayyan_ch3_baseline_wander_filter_difference_eq',
    'bwsqmag',
    'rangayyan_ch3_butterworth_lowpass_squared_magnitude',
    'bwsqlap',
    'rangayyan_ch3_butterworth_squared_laplace',
    'bwpoles',
    'rangayyan_ch3_butterworth_pole_positions',
    'bwanalog',
    'rangayyan_ch3_butterworth_analog_transfer_function',
    'bilinear',
    'rangayyan_ch3_bilinear_transformation',
    'bilinunit',
    'rangayyan_ch3_bilinear_unit_circle_relation',
    'bilinwarp',
    'rangayyan_ch3_bilinear_warping_omega_to_Omega',
    'bilinunwarp',
    'rangayyan_ch3_bilinear_warping_Omega_to_omega',
    'bwdigital',
    'rangayyan_ch3_butterworth_digital_transfer_function',
    'iirdiffgen',
    'rangayyan_ch3_iir_difference_eq_general',
    'bwdirect',
    'rangayyan_ch3_butterworth_lowpass_direct_specification',
    'bwlpdft',
    'rangayyan_ch3_butterworth_lowpass_dft_indexed',
    'bwhpdft',
    'rangayyan_ch3_butterworth_highpass_dft_indexed',
    'notch60',
    'rangayyan_ch3_notch_filter_60Hz',
    'mfilth',
    'rangayyan_ch4_matched_filter_h_example',
]

def _poly_from_roots(roots):
    """Expand prod (z - r_k) into ascending-power coefficients."""
    coefs = [complex(1.0, 0.0)]
    for r in roots:
        nxt = [complex(0.0, 0.0)] * (len(coefs) + 1)
        for i, c in enumerate(coefs):
            nxt[i] += -complex(r) * c
            nxt[i + 1] += c
        coefs = nxt
    return coefs

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
def bwhp(cutoff_hz, order=4, fs=1000.0, z=None):
    """Butterworth highpass design.

    The lowpass poles are reused -- a Butterworth highpass has the same
    pole radius -- and the N zeros are moved from z = -1 to z = +1, which
    is the lowpass-to-highpass reflection of the unit circle.  The gain is
    then renormalized at NYQUIST rather than at DC, since a highpass has
    no gain at DC to normalize against; normalizing at DC would divide by
    zero.
    """
    fsv = float(fs)
    fcv = float(cutoff_hz)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if not 0 < fcv < fsv / 2.0:
        raise ValueError("the cutoff must lie strictly between 0 and the "
                         "Nyquist frequency %g Hz" % (fsv / 2.0))
    n = int(order)
    lp = bwdigital(N=n, fc=fcv, fs=fsv)
    a = list(lp["a"])
    num = [c.real for c in _poly_from_roots([1.0] * n)]
    b = list(reversed(num))
    # normalize at z = -1, the Nyquist point
    nyq_num = fsum(b[k] * ((-1.0) ** k) for k in range(len(b)))
    nyq_den = fsum(a[k] * ((-1.0) ** k) for k in range(len(a)))
    if abs(nyq_num) <= 1e-300:
        raise ValueError("the numerator vanishes at Nyquist")
    G = nyq_den / nyq_num
    b = [G * v for v in b]
    Hz = None
    if z is not None:
        scalar = not isinstance(z, (list, tuple))
        zs = [z] if scalar else list(z)
        vals = []
        for zv in zs:
            dd = _polyz(a, zv)
            if abs(dd) <= 1e-300:
                raise ValueError("z is a pole of H(z)")
            vals.append(_polyz(b, zv) / dd)
        Hz = vals[0] if scalar else vals
    return RichResult(payload={
        "b": b, "a": a, "gain": G, "H": Hz, "N": n,
        "cutoff_hz": fcv, "fs": fsv, "order": n, "kind": "highpass",
        "zeros_at_plus_one": n, "dc_gain": 0.0, "nyquist_gain": 1.0,
        "prewarped": True, "normalized_at_nyquist": True,
        "method": "Rangayyan (2024) Section 3.7; Butterworth highpass"})


rangayyan_butterworth_hp = bwhp  # pre-policy spelling


# -- rgblp: Butterworth lowpass filter design (analog prototype to digital).
def bwlp(cutoff_hz, order=4, fs=1000.0, z=None):
    """Butterworth lowpass design, analog prototype to digital.

    Runs the book's route end to end: prewarp the cutoff by eq. (3.141),
    place the poles by eq. (3.137), keep the left-half-plane ones by
    eq. (3.138), and apply the bilinear transform of eq. (3.139) to reach
    the digital form of eq. (3.143).

    The prewarping is done here rather than left to the caller.  Without
    it the realized cutoff sits below the one requested, and the error
    grows as the cutoff approaches Nyquist -- at f_s/4 it is already
    several per cent.
    """
    fsv = float(fs)
    fcv = float(cutoff_hz)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if not 0 < fcv < fsv / 2.0:
        raise ValueError("the cutoff must lie strictly between 0 and the "
                         "Nyquist frequency %g Hz" % (fsv / 2.0))
    r = bwdigital(N=int(order), fc=fcv, fs=fsv, z=z)
    out = dict(r)
    out.update({
        "cutoff_hz": fcv, "fs": fsv, "order": int(order),
        "prewarped": True, "kind": "lowpass",
        "method": "Rangayyan (2024) eqs. (3.135)-(3.143); Butterworth "
                  "lowpass via the bilinear transform"})
    return RichResult(payload=out)


rangayyan_butterworth_lp = bwlp  # pre-policy spelling


# -- rgcomb: Comb filter for periodic artifact removal.
def comb(period_samples, fs=1000.0, z=None):
    """Comb filter for periodic artifact removal.

        H(z) = (1/2) ( 1 - z^-N )

    N zeros spaced evenly round the unit circle, so the filter notches
    DC AND every harmonic of f_s/N at once -- which is exactly what
    powerline interference is, a fundamental plus its harmonics, and why
    one comb does the work of a bank of notches.

    The zero at DC is not optional: 1 - z^-N always vanishes at z = 1, so
    a comb removes the mean along with the interference.  If the baseline
    must be kept, the DC zero has to be cancelled by a pole.
    """
    N = int(period_samples)
    if N < 1:
        raise ValueError("the period must be at least one sample")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    b = [0.0] * (N + 1)
    b[0] = 0.5
    b[N] = -0.5
    Hz = None
    if z is not None:
        scalar = not isinstance(z, (list, tuple))
        zs = [z] if scalar else list(z)
        vals = [_polyz(b, zv) for zv in zs]
        Hz = vals[0] if scalar else vals
    notches = [k * fsv / N for k in range(N // 2 + 1)]
    return RichResult(payload={
        "b": b, "a": [1.0], "H": Hz, "period_samples": N, "fs": fsv,
        "notch_frequencies_hz": notches, "n_zeros": N,
        "notch_spacing_hz": fsv / N, "dc_gain": 0.0,
        "removes_dc_as_well": True, "fir": True, "linear_phase": True,
        "method": "Rangayyan (2024) Section 3.7 (comb filter)"})


rangayyan_comb_filter = comb  # pre-policy spelling


# -- rgfd1: First-difference operator for baseline wander removal.
def diff1(x, T=1.0):
    """First difference applied to a record, for baseline removal.

        y(n) = x(n) - x(n-1)

    The operator of eq. (3.123) run over data, with the frequency
    response reported alongside so its highpass character is visible:
    it removes the wandering baseline, but it also boosts high-frequency
    noise, and eq. (3.132) is the book's remedy for that.
    """
    r = fdiff(x, T=T)
    out = dict(r)
    out.update({
        "b": [1.0 / float(T), -1.0 / float(T)], "a": [1.0],
        "zeros": [1.0], "highpass": True,
        "use_bwander_to_avoid_the_noise_boost": True,
        "method": "Rangayyan (2024) eq. (3.123) applied to a record"})
    return RichResult(payload=out)


rangayyan_first_diff = diff1  # pre-policy spelling


# -- rgfd2: Second-difference operator.
def diff2(x, T=1.0, n=None):
    """Second-order difference operator.

        y(n) = x(n) - 2 x(n-1) + x(n-2)

    The book notes the second-order derivative has frequency response
    (jw)(jw) = -w^2, a QUADRATIC rise in gain with frequency, and that it
    "may be realized as a cascade of two" first-order differences.  Both
    the direct form and that cascade are computed here and compared.

    The quadratic rise means it amplifies high-frequency noise far harder
    than the first difference does -- useful only where the wanted
    feature is itself sharp.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    out = []
    for i in range(len(xs)):
        a = xs[i]
        b = xs[i - 1] if i >= 1 else 0.0
        c = xs[i - 2] if i >= 2 else 0.0
        out.append((a - 2.0 * b + c) / (Tv * Tv))
    cascade = fdiff(fdiff(xs, T=Tv)["y"], T=Tv)["y"]
    gap = max(abs(p - q) for p, q in zip(out, cascade))
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "T": Tv,
        "as_cascaded_first_differences": cascade, "max_difference": gap,
        "cascade_agrees": gap <= 1e-9,
        "b": [1.0 / (Tv * Tv), -2.0 / (Tv * Tv), 1.0 / (Tv * Tv)],
        "a": [1.0], "zeros": [1.0, 1.0], "double_zero_at_dc": True,
        "gain_rises_quadratically": True,
        "method": "Rangayyan (2024) Section 3.3.3 (second derivative)"})


rangayyan_second_diff = diff2  # pre-policy spelling


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
def freqresp(b, a=None, fs=1000.0, n_freqs=512):
    """Frequency response of a digital filter from its coefficients.

        H(f) = sum_k b_k exp(-j 2 pi f k / f_s)
               / sum_k a_k exp(-j 2 pi f k / f_s)

    Evaluated on a uniform grid from DC to NYQUIST inclusive -- the
    one-sided response, since for real coefficients the other half is the
    conjugate mirror and carries nothing new.

    ``a`` follows the eq. (3.67) convention with a_0 = 1 included, which
    is the form ``bwlp`` and ``bwhp`` return.
    """
    bs = aslist(b)
    if not bs:
        raise ValueError("need at least one numerator coefficient")
    az = aslist(a) if a is not None else [1.0]
    if not az:
        raise ValueError("the denominator needs at least one coefficient")
    if abs(az[0]) <= 1e-300:
        raise ValueError("a_0 must not be zero")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    m = int(n_freqs)
    if m < 2:
        raise ValueError("need at least two frequency points")
    freqs, H = [], []
    for i in range(m):
        f = 0.5 * fsv * i / (m - 1)
        w = 2.0 * pi * f / fsv
        num = sum(bs[k] * complex(cos(-w * k), sin(-w * k))
                  for k in range(len(bs)))
        den = sum(az[k] * complex(cos(-w * k), sin(-w * k))
                  for k in range(len(az)))
        if abs(den) <= 1e-300:
            raise ValueError("the denominator vanishes at f = %g Hz; the "
                             "filter has a pole on the unit circle" % f)
        freqs.append(f)
        H.append(num / den)
    mag = [abs(v) for v in H]
    return RichResult(payload={
        "f": freqs, "H": H, "magnitude": mag,
        "magnitude_db": [20.0 * log10(v) if v > 0 else float("-inf")
                         for v in mag],
        "phase": [atan2(v.imag, v.real) for v in H],
        "fs": fsv, "n_freqs": m, "one_sided": True,
        "includes_nyquist": True,
        "method": "Rangayyan (2024) Section 3.5 (frequency response)"})


rangayyan_freq_response = freqresp  # pre-policy spelling


# -- rggrpd: Group delay of a digital filter.
def grpdelay(b, a=None, fs=1000.0, n_freqs=512):
    """Group delay of a digital filter.

        tau_g(f) = - d(phase) / d(omega)

    Computed from the COEFFICIENTS rather than by differentiating a
    numerical phase curve:

        tau_g = Re[ sum_k k b_k e^-jwk / sum_k b_k e^-jwk ]
                - Re[ sum_k k a_k e^-jwk / sum_k a_k e^-jwk ]

    Differentiating the phase is the obvious route and it is wrong at any
    zero on the unit circle, where the phase steps by PI.  That step is
    real, not a branch-cut artifact, so unwrapping -- which only ever
    removes multiples of 2 pi -- leaves it in place and the derivative
    reports a spike.  A three-point moving average has such a zero at
    w = 2 pi / 3, and differentiating its phase gives a mean group delay
    near zero instead of the correct 1.

    The delay is undefined wherever the numerator or denominator
    vanishes; those points are returned as None and marked, not filled
    in.
    """
    bs = aslist(b)
    if not bs:
        raise ValueError("need at least one numerator coefficient")
    az = aslist(a) if a is not None else [1.0]
    if not az:
        raise ValueError("the denominator needs at least one coefficient")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    m = int(n_freqs)
    if m < 2:
        raise ValueError("need at least two frequency points")

    def ratio(coefs, w):
        num = sum(k * coefs[k] * complex(cos(-w * k), sin(-w * k))
                  for k in range(len(coefs)))
        den = sum(coefs[k] * complex(cos(-w * k), sin(-w * k))
                  for k in range(len(coefs)))
        return num, den

    freqs, tau, defined = [], [], []
    for i in range(m):
        f = 0.5 * fsv * i / (m - 1)
        w = 2.0 * pi * f / fsv
        bn, bd = ratio(bs, w)
        an, ad = ratio(az, w)
        freqs.append(f)
        if abs(bd) <= 1e-12 or abs(ad) <= 1e-12:
            tau.append(None)
            defined.append(False)
        else:
            tau.append((bn / bd).real - (an / ad).real)
            defined.append(True)
    good = [v for v in tau if v is not None]
    if not good:
        raise ValueError("the response vanishes at every frequency "
                         "evaluated; the group delay is undefined")
    mu = fsum(good) / len(good)
    spread = max(abs(v - mu) for v in good)
    return RichResult(payload={
        "f": freqs, "group_delay": tau, "fs": fsv,
        "mean": mu, "max_deviation": spread,
        "approximately_constant": spread <= 1e-9 * max(1.0, abs(mu)),
        "defined": defined,
        "n_undefined": sum(1 for v in defined if not v),
        "from_the_coefficients": True,
        "phase_differentiation_breaks_at_unit_circle_zeros": True,
        "method": "Rangayyan (2024) Section 3.5 (group delay)"})


rangayyan_group_delay = grpdelay  # pre-policy spelling


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
def notch(notch_freq, bandwidth=None, fs=1000.0, r=None, z=None):
    """Notch filter with two zeros and two poles.

        H(z) = G (1 - 2cos(w_0) z^-1 + z^-2)
                 / (1 - 2 r cos(w_0) z^-1 + r^2 z^-2)

    The zeros sit ON the unit circle at the interference frequency and
    the poles just inside at the same angle, radius r.  The poles are
    what make the notch NARROW: without them the two zeros pull the
    response down over a wide band, taking signal with them.

    The bandwidth follows r as bw ~ (1 - r) f_s / pi, and either may be
    given.  r must stay strictly inside the unit circle -- at r = 1 the
    poles cancel the zeros and the filter does nothing at all.
    """
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    f0 = float(notch_freq)
    if not 0 < f0 < fsv / 2.0:
        raise ValueError("the notch frequency must lie strictly between 0 "
                         "and the Nyquist frequency")
    if (bandwidth is None) == (r is None):
        raise ValueError("give either the bandwidth or the pole radius r, "
                         "not both and not neither")
    if bandwidth is not None:
        bw = float(bandwidth)
        if bw <= 0:
            raise ValueError("the bandwidth must be positive")
        rv = 1.0 - pi * bw / fsv
        if rv <= 0:
            raise ValueError("that bandwidth needs a pole radius <= 0; ask "
                             "for a narrower notch")
    else:
        rv = float(r)
        bw = (1.0 - rv) * fsv / pi
    if not 0 < rv < 1:
        raise ValueError("the pole radius must satisfy 0 < r < 1; at r = 1 "
                         "the poles cancel the zeros")
    w0 = 2.0 * pi * f0 / fsv
    bb = [1.0, -2.0 * cos(w0), 1.0]
    aa = [1.0, -2.0 * rv * cos(w0), rv * rv]
    dcn = fsum(bb)
    dcd = fsum(aa)
    G = (dcd / dcn) if abs(dcn) > 1e-300 else 1.0
    bb = [G * v for v in bb]
    Hz = None
    if z is not None:
        scalar = not isinstance(z, (list, tuple))
        zs = [z] if scalar else list(z)
        vals = []
        for zv in zs:
            dd = _polyz(aa, zv)
            if abs(dd) <= 1e-300:
                raise ValueError("z is a pole of H(z)")
            vals.append(_polyz(bb, zv) / dd)
        Hz = vals[0] if scalar else vals
    zc = complex(cos(w0), sin(w0))
    return RichResult(payload={
        "b": bb, "a": aa, "gain": G, "H": Hz,
        "f0": f0, "fs": fsv, "r": rv, "bandwidth_hz": bw,
        "omega_0": w0,
        "zeros": [zc, zc.conjugate()],
        "poles": [rv * zc, rv * zc.conjugate()],
        "gain_at_the_notch": abs(_polyz(bb, zc)) / abs(_polyz(aa, zc)),
        "dc_gain": 1.0, "iir": True,
        "poles_narrow_the_notch": True,
        "method": "Rangayyan (2024) Section 3.7 (notch filter with poles)"})


rangayyan_notch_filter = notch  # pre-policy spelling


# -- rgosflt: Order-statistic (median) filter.
def osfilt(x, window, kind="median", alpha=0.0, weights=None, order=None):
    """Order-statistic filters, Rangayyan (2024) Section 3.8.

    Rank the samples in a moving window and take one entry, or a
    combination of entries, as the output:

      "min"      first entry; removes high-valued impulsive noise
      "max"      last entry; removes low-valued impulsive noise
      "minmax"   the min filter followed by the max filter
      "median"   the middle entry -- the book's "most popular and
                 commonly used" order-statistic filter
      "trimmed"  mean of the list after dropping the lowest and highest
                 alpha x 100 per cent, 0 <= alpha < 0.5
      "l"        L-filter, a weighted combination of the whole ranked
                 list; suitable weights reproduce any of the above
      "order"    the ith entry outright, ``order`` counting from 1

    All of these are NONLINEAR, so, as the book notes, none of them can
    be analysed with the Fourier transform: there is no frequency
    response to report and none is returned.

    The window is centred and odd-length, and the edges are handled by
    symmetric reflection so the output is the same length as the input
    and no artificial step is introduced at either end.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    w = int(window)
    if w < 1:
        raise ValueError("the window must hold at least one sample")
    if w % 2 == 0:
        raise ValueError("the window must be odd so it can be centred, "
                         "got %d" % w)
    if w > n:
        raise ValueError("the window is longer than the record")
    kinds = ("min", "max", "minmax", "median", "trimmed", "l", "order")
    if kind not in kinds:
        raise ValueError("kind must be one of %s, got %r"
                         % (", ".join(kinds), kind))
    av = float(alpha)
    if kind == "trimmed" and not 0.0 <= av < 0.5:
        raise ValueError("the book writes 0 <= alpha < 0.5; at 0.5 the "
                         "whole list is trimmed away, got %g" % av)
    if kind == "l":
        if weights is None:
            raise ValueError("the L-filter needs one weight per rank")
        wts = aslist(weights)
        if len(wts) != w:
            raise ValueError("the L-filter needs %d weights, one per "
                             "rank, got %d" % (w, len(wts)))
        tot = fsum(wts)
        if abs(tot) <= 1e-300:
            raise ValueError("the L-filter weights sum to zero")
    if kind == "order":
        if order is None:
            raise ValueError("kind='order' needs the rank to take")
        i_ord = int(order)
        if not 1 <= i_ord <= w:
            raise ValueError("order must lie in 1..%d, got %d"
                             % (w, i_ord))

    half = w // 2

    def padded(seq):
        # whole-sample symmetric reflection: the edge value is repeated,
        # so a monotone run passes through a median filter untouched.
        # Half-sample reflection (dropping the edge) shifts the ends by a
        # sample, which shows up as a spurious step in the output.
        left = list(reversed(seq[:half]))
        right = list(reversed(seq[len(seq) - half:]))
        return left + list(seq) + right

    def rank_pass(seq, take):
        pad = padded(seq)
        return [take(sorted(pad[i:i + w])) for i in range(len(seq))]

    if kind == "min":
        out = rank_pass(xs, lambda r: r[0])
    elif kind == "max":
        out = rank_pass(xs, lambda r: r[-1])
    elif kind == "minmax":
        out = rank_pass(rank_pass(xs, lambda r: r[0]), lambda r: r[-1])
    elif kind == "median":
        out = rank_pass(xs, lambda r: r[half])
    elif kind == "order":
        out = rank_pass(xs, lambda r: r[i_ord - 1])
    elif kind == "trimmed":
        drop = int(av * w)
        if 2 * drop >= w:
            drop = (w - 1) // 2
        out = rank_pass(xs, lambda r: fsum(r[drop:w - drop])
                        / (w - 2 * drop))
    else:
        out = rank_pass(xs, lambda r: fsum(a * b for a, b in zip(wts, r))
                        / tot)

    return RichResult(payload={
        "y": out, "n": len(out), "window": w, "kind": kind,
        "alpha": av if kind == "trimmed" else None,
        "trimmed_each_end": int(av * w) if kind == "trimmed" else None,
        "order": i_ord if kind == "order" else None,
        "nonlinear": True, "no_frequency_response": True,
        "edges": "symmetric reflection",
        "method": "Rangayyan (2024) Section 3.8 (order-statistic "
                  "filters)"})


rangayyan_order_stat_flt = osfilt  # pre-policy spelling


# -- rgphas: Phase response of a digital filter.
def phaseresp(b, a=None, fs=1000.0, n_freqs=512, unwrap=True):
    """Phase response of a digital filter.

        phi(f) = angle H(f)

    The principal value jumps by 2 pi wherever it crosses the branch cut,
    which is an artifact of the arctangent and not of the filter.  It is
    unwrapped by default, because a wrapped phase makes a linear-phase
    filter look nonlinear and makes the group delay meaningless.  Both
    the wrapped and unwrapped curves are returned.
    """
    r = freqresp(b, a=a, fs=fs, n_freqs=n_freqs)
    wrapped = list(r["phase"])
    mag = r["magnitude"]
    scale = max(mag) if mag else 0.0
    # Where H vanishes the phase is undefined -- atan2(0, 0) returns 0,
    # which is not a phase.  Those points are marked and skipped by the
    # unwrap, which otherwise carries the bogus value into every later
    # sample.  The Hann filter of eq. (3.100) hits this at Nyquist.
    defined = [v > 1e-9 * scale for v in mag] if scale > 0 \
        else [False] * len(mag)
    unw, last = [], None
    for i in range(len(wrapped)):
        if not defined[i]:
            unw.append(unw[-1] if unw else wrapped[i])
            continue
        if last is None:
            unw.append(wrapped[i])
        else:
            d = wrapped[i] - wrapped[last]
            while d > pi:
                d -= 2.0 * pi
            while d < -pi:
                d += 2.0 * pi
            unw.append(unw[last] + d)
        last = i
    return RichResult(payload={
        "f": r["f"], "phase": unw if unwrap else wrapped,
        "wrapped": wrapped, "unwrapped": unw, "unwrap": bool(unwrap),
        "fs": r["fs"], "defined": defined,
        "n_undefined": sum(1 for v in defined if not v),
        "phase_undefined_where_the_response_vanishes": True,
        "wrapping_is_an_arctangent_artifact": True,
        "method": "Rangayyan (2024) Section 3.5 (phase response)"})


rangayyan_phase_response = phaseresp  # pre-policy spelling


# -- rgsinc: Ideal sinc (low-pass) filter impulse response.
def sinckern(fc, fs=1000.0, M=64, window=None):
    """Ideal lowpass (sinc) impulse response.

        h(n) = 2 (f_c/f_s) sinc( 2 (f_c/f_s) (n - M/2) )

    The inverse transform of a rectangular passband, truncated to M + 1
    taps and delayed by M/2 to make it causal.  The truncation is the
    catch: cutting a sinc off abruptly is multiplying it by a rectangle,
    whose transform has slowly decaying sidelobes, so the realized
    stopband ripples.  That is Gibbs' phenomenon and it does not improve
    with M -- only the ripples' width shrinks, not their height.

    Supplying a ``window`` tapers the truncation and fixes it, which is
    what Section 3.4's windows are for.
    """
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    fcv = float(fc)
    if not 0 < fcv < fsv / 2.0:
        raise ValueError("the cutoff must lie strictly between 0 and the "
                         "Nyquist frequency")
    m = int(M)
    if m < 1:
        raise ValueError("M must be at least 1")
    ratio = 2.0 * fcv / fsv
    h = []
    for n in range(m + 1):
        t = n - m / 2.0
        if abs(t) <= 1e-12:
            h.append(ratio)
        else:
            h.append(ratio * sin(pi * ratio * t) / (pi * ratio * t))
    win = None
    if window is not None:
        win = windowfn(m + 1, window)["w"]
        h = [a * b for a, b in zip(h, win)]
    total = fsum(h)
    if abs(total) > 1e-300:
        h = [v / total for v in h]
    return RichResult(payload={
        "h": h, "n_taps": m + 1, "fc": fcv, "fs": fsv, "M": m,
        "window": window, "window_values": win,
        "delay_samples": m / 2.0, "dc_gain": 1.0,
        "truncation_causes_gibbs_ripple": window is None,
        "ripple_height_does_not_shrink_with_M": True,
        "method": "Rangayyan (2024) Section 3.4 (windowed sinc)"})


rangayyan_sinc_kernel = sinckern  # pre-policy spelling


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
def blackman(N):
    """Blackman window.

        w(n) = 0.42 - 0.5 cos(2 pi n/(N-1)) + 0.08 cos(4 pi n/(N-1))

    A third cosine term buys much deeper sidelobes than the Hamming --
    about -58 dB -- at the cost of a main lobe half again as wide.  That
    is the standing trade: resolution against leakage, and no window
    escapes it.
    """
    n = int(N)
    if n < 1:
        raise ValueError("N must be at least 1")
    if n == 1:
        return RichResult(payload={
            "w": [1.0], "N": 1, "sum": 1.0, "endpoints": [1.0, 1.0],
            "method": "Rangayyan (2024) Section 3.4 (Blackman window)"})
    w = [0.42 - 0.5 * cos(2.0 * pi * i / (n - 1))
         + 0.08 * cos(4.0 * pi * i / (n - 1)) for i in range(n)]
    return RichResult(payload={
        "w": w, "N": n, "sum": fsum(w),
        "endpoints": [w[0], w[-1]],
        "coherent_gain": fsum(w) / n,
        "widest_main_lobe_of_the_three": True,
        "resolution_traded_for_leakage": True,
        "symmetric": all(abs(w[i] - w[n - 1 - i]) < 1e-12
                         for i in range(n)),
        "method": "Rangayyan (2024) Section 3.4 (Blackman window)"})


rangayyan_blackman_window = blackman  # pre-policy spelling


# -- rgwhamp: Hamming window function.
def hamming(N):
    """Hamming window.

        w(n) = 0.54 - 0.46 cos( 2 pi n / (N-1) ),   0 <= n <= N-1

    The 0.54/0.46 split is chosen to cancel the largest sidelobe of the
    rectangle, which buys about -43 dB of sidelobe suppression at the
    cost of a wider main lobe than the Hann.  It does NOT reach zero at
    the ends -- w(0) = w(N-1) = 0.08 -- which is the difference from the
    Hann and matters when windows are overlapped and added.
    """
    n = int(N)
    if n < 1:
        raise ValueError("N must be at least 1")
    if n == 1:
        return RichResult(payload={
            "w": [1.0], "N": 1, "sum": 1.0, "endpoints": [1.0, 1.0],
            "reaches_zero_at_the_ends": False,
            "method": "Rangayyan (2024) Section 3.4 (Hamming window)"})
    w = [0.54 - 0.46 * cos(2.0 * pi * i / (n - 1)) for i in range(n)]
    return RichResult(payload={
        "w": w, "N": n, "sum": fsum(w),
        "endpoints": [w[0], w[-1]],
        "reaches_zero_at_the_ends": False,
        "coherent_gain": fsum(w) / n,
        "symmetric": all(abs(w[i] - w[n - 1 - i]) < 1e-12
                         for i in range(n)),
        "method": "Rangayyan (2024) Section 3.4 (Hamming window)"})


rangayyan_hamming_window = hamming  # pre-policy spelling


# -- rgwhann: Hann (Hanning) window function.
def hannwin(N):
    """Hann (Hanning) window.

        w(n) = 0.5 [ 1 - cos( 2 pi n / (N-1) ) ]

    Reaches exactly zero at both ends, so overlapped Hann windows add to
    a constant at 50 per cent overlap -- the property that makes it the
    default for overlap-add analysis.  Its sidelobes fall off faster than
    the Hamming's even though the first one is higher.

    Not to be confused with the Hann FILTER of eq. (3.100), which is a
    three-tap 1:2:1 smoother; this is a taper applied to a data segment.
    """
    n = int(N)
    if n < 1:
        raise ValueError("N must be at least 1")
    if n == 1:
        return RichResult(payload={
            "w": [1.0], "N": 1, "sum": 1.0, "endpoints": [1.0, 1.0],
            "reaches_zero_at_the_ends": False,
            "method": "Rangayyan (2024) Section 3.4 (Hann window)"})
    w = [0.5 * (1.0 - cos(2.0 * pi * i / (n - 1))) for i in range(n)]
    return RichResult(payload={
        "w": w, "N": n, "sum": fsum(w),
        "endpoints": [w[0], w[-1]],
        "reaches_zero_at_the_ends": True,
        "coherent_gain": fsum(w) / n,
        "not_the_hann_filter_of_eq_3_100": True,
        "symmetric": all(abs(w[i] - w[n - 1 - i]) < 1e-12
                         for i in range(n)),
        "method": "Rangayyan (2024) Section 3.4 (Hann window)"})


rangayyan_hann_window = hannwin  # pre-policy spelling


# -- rgwndw: Window functions: Hamming, Hann, Blackman for spectral leakage control.
def windowfn(N, window_type="hamming"):
    """Window functions for spectral leakage control, Section 3.4.

    One entry point for the rectangular, Hann, Hamming and Blackman
    windows.  Truncating a record is multiplying it by a rectangle, and
    the rectangle's transform has sidelobes that leak energy from strong
    components into neighbouring bins; a tapered window trades a wider
    main lobe for lower sidelobes.

    The rectangular window is included because it IS the default -- doing
    nothing is choosing it -- and naming it makes that choice explicit.
    """
    n = int(N)
    if n < 1:
        raise ValueError("N must be at least 1")
    kinds = ("rectangular", "hann", "hamming", "blackman")
    if window_type not in kinds:
        raise ValueError("window_type must be one of %s, got %r"
                         % (", ".join(kinds), window_type))
    if window_type == "rectangular":
        w = [1.0] * n
        r = {"w": w, "N": n, "sum": float(n), "endpoints": [1.0, 1.0],
             "coherent_gain": 1.0, "symmetric": True}
    elif window_type == "hann":
        r = dict(hannwin(n))
    elif window_type == "hamming":
        r = dict(hamming(n))
    else:
        r = dict(blackman(n))
    r["window_type"] = window_type
    r["doing_nothing_is_the_rectangular_window"] = True
    r["method"] = "Rangayyan (2024) Section 3.4 (window functions)"
    return RichResult(payload=r)


rangayyan_window_functions = windowfn  # pre-policy spelling


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
def ma8imp(n=None):
    """Impulse response of the 8-point MA filter, eq. (3.109).

        h(n) = (1/8) [ delta(n) + delta(n-1) + ... + delta(n-7) ]

    Eight equal taps.  Equal weighting is what makes the stopband
    attenuation poor: the book notes the filter gives no more than about
    -20 dB at most frequencies, which is why its result on a noisy ECG
    (Figure 3.53) is still visibly noisy.
    """
    taps = [0.125] * 8
    val = None
    if n is not None:
        idx = int(n)
        val = taps[idx] if 0 <= idx < 8 else 0.0
    return RichResult(payload={
        "h": taps, "value": val, "index": n, "n_taps": 8,
        "sum": 1.0, "finite": True, "equal_weights": True,
        "attenuation_is_poor": True,
        "method": "Rangayyan (2024) eq. (3.109)"})


rangayyan_ch3_ma_8point_impulse_response = ma8imp  # pre-policy spelling


# -- rng099: Transfer function of the 8-point MA filter..
def ma8tf(z):
    """Transfer function of the 8-point MA filter, eq. (3.110).

        H(z) = (1/8) sum_{k=0}^{7} z^-k

    Seven zeros spaced evenly round the unit circle, at every multiple of
    f_s/8 except DC.  For f_s = 1000 Hz the book puts them at 125, 250,
    375 and 500 Hz (with the negative-frequency conjugates), which is
    what the notches in Figure 3.50 are.
    """
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = [_polyz([0.125] * 8, zv) for zv in zs]
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z, "n_taps": 8,
        "n_zeros": 7, "zeros_at_multiples_of_fs_over_8": True,
        "dc_gain": 1.0, "always_stable": True,
        "method": "Rangayyan (2024) eq. (3.110)"})


rangayyan_ch3_ma_8point_transfer_function = ma8tf  # pre-policy spelling


# -- rng100: Frequency response of the 8-point MA filter..
def ma8fr(omega):
    """Frequency response of the 8-point MA filter, eq. (3.111).

        H(w) = (1/8) sum_{k=0}^{7} exp(-j w k)
             = (1/8) { 1 + exp(-j4w) [ 1 + 2cos(w) + 2cos(2w)
                                       + 2cos(3w) ] }

    The book's factored form is EXACT: the bracket is the sum over lags
    -3..3, and multiplying it by exp(-j4w) shifts that to lags 1..7,
    which with the leading 1 is the whole sum.  (The placeholder
    docstring rendered it as a product of two brackets; that is a
    different function and does not agree with the sum.)  Both forms are
    computed here and compared.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    direct, factored = [], []
    for w in ws:
        direct.append(0.125 * sum(complex(cos(-w * k), sin(-w * k))
                                  for k in range(8)))
        brack = 1.0 + 2.0 * cos(w) + 2.0 * cos(2.0 * w) + 2.0 * cos(3.0 * w)
        factored.append(0.125 * (1.0 + complex(cos(-4.0 * w),
                                               sin(-4.0 * w)) * brack))
    gap = max(abs(a - b) for a, b in zip(direct, factored))
    return RichResult(payload={
        "H": direct[0] if scalar else direct,
        "factored": factored[0] if scalar else factored,
        "omega": omega,
        "magnitude": abs(direct[0]) if scalar else [abs(v) for v in direct],
        "max_difference": gap, "factored_form_agrees": gap <= 1e-12,
        "bracket_is_inside_the_product": True,
        "method": "Rangayyan (2024) eq. (3.111)"})


rangayyan_ch3_ma_8point_frequency_response = ma8fr  # pre-policy spelling


# -- rng101: Continuous-time integral over a sliding window of duration tau..
def runint(x, t, tau):
    """Running integral over a sliding window, eq. (3.112).

        y(t) = integral_{t-tau}^{t} x(t) dt

    The continuous-time counterpart of the moving-average sum: eq. (3.108)
    is this integral discretized.  Evaluated by the trapezoidal rule on
    the samples given, with the window clipped at the start of the record
    -- the leading samples cover less than tau, and how many is reported
    rather than left to be discovered from the output.
    """
    xs, ts = aslist(x), aslist(t)
    if len(xs) != len(ts):
        raise ValueError("x and t must have the same length")
    if len(xs) < 2:
        raise ValueError("need at least two samples to integrate")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("t must be strictly increasing")
    tv = float(tau)
    if tv <= 0:
        raise ValueError("tau must be positive")
    out, clipped = [], 0
    for i in range(len(ts)):
        lo = ts[i] - tv
        if lo < ts[0]:
            lo = ts[0]
            clipped += 1
        acc = 0.0
        for j in range(i):
            a, b = ts[j], ts[j + 1]
            if b <= lo:
                continue
            fa, fb = xs[j], xs[j + 1]
            if a < lo:                       # partial panel at the edge
                fa = fa + (fb - fa) * (lo - a) / (b - a)
                a = lo
            acc += 0.5 * (fa + fb) * (b - a)
        out.append(acc)
    return RichResult(payload={
        "y": out, "n": len(out), "tau": tv, "clipped_windows": clipped,
        "trapezoidal": True,
        "continuous_counterpart_of_the_ma_filter": True,
        "method": "Rangayyan (2024) eq. (3.112)"})


rangayyan_ch3_running_integral_window = runint  # pre-policy spelling


# -- rng102: General definition of running integral over (-inf, t]..
def runintall(x, t):
    """Running integral from minus infinity, eq. (3.113).

        y(t) = integral_{-inf}^{t} x(t) dt

    The general definition.  Over a finite record the lower limit is the
    first sample, so what is returned is the cumulative integral from the
    START OF THE RECORD -- any mass before it is unobserved and cannot be
    recovered, which is why the constant of integration is arbitrary and
    is reported as such.

    The discrete counterpart accumulates every sample and has transfer
    function 1/(1 - z^-1) -- a pole ON the unit circle at DC, so it has
    no bounded frequency response and any offset in the input walks off
    without limit.  The book notes such an operation is seldom used in
    filtering; the windowed form of eq. (3.112) is used instead.
    """
    xs, ts = aslist(x), aslist(t)
    if len(xs) != len(ts):
        raise ValueError("x and t must have the same length")
    if len(xs) < 2:
        raise ValueError("need at least two samples to integrate")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("t must be strictly increasing")
    out, acc = [0.0], 0.0
    for i in range(len(ts) - 1):
        acc += 0.5 * (xs[i] + xs[i + 1]) * (ts[i + 1] - ts[i])
        out.append(acc)
    return RichResult(payload={
        "y": out, "n": len(out), "total": acc,
        "lower_limit": ts[0],
        "constant_of_integration_is_arbitrary": True,
        "discrete_pole_on_the_unit_circle": True,
        "seldom_used_for_filtering": True,
        "method": "Rangayyan (2024) eq. (3.113)"})


rangayyan_ch3_integral_general = runintall  # pre-policy spelling


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
def intft(X, omega, X0=None):
    """Fourier transform of an integral, eq. (3.115).

        Y(w) = (1/(jw)) X(w) + pi X(0) delta(w)

    The delta term is not decoration: it carries the DC content, which
    the 1/(jw) factor cannot represent because it blows up there.  A
    caller who drops it -- as eq. (3.116) deliberately does, "keeping
    aside the second term related to DC" -- is computing the response of
    the integrator to everything EXCEPT the mean, so a signal with an
    offset will not be reconstructed.

    The delta is returned as its weight, pi X(0), rather than evaluated:
    a delta has no value at a point.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    Xs = [_cnum(X)] * len(ws) if not isinstance(X, (list, tuple)) \
        else [_cnum(v) for v in X]
    if len(Xs) != len(ws):
        raise ValueError("X and omega must have the same length")
    dc = _cnum(X0) if X0 is not None else None
    out, at_dc = [], []
    for w, xv in zip(ws, Xs):
        if abs(w) <= 1e-300:
            out.append(None)                # 1/(jw) is unbounded at w = 0
            at_dc.append(True)
        else:
            out.append(xv / complex(0.0, w))
            at_dc.append(False)
    return RichResult(payload={
        "Y": out[0] if scalar else out, "omega": omega,
        "delta_weight": (pi * dc) if dc is not None else None,
        "at_dc": at_dc[0] if scalar else at_dc,
        "dc_term_carried_by_the_delta": True,
        "undefined_at_zero_without_the_delta": True,
        "method": "Rangayyan (2024) eq. (3.115)"})


rangayyan_ch3_fourier_of_integral = intft  # pre-policy spelling


# -- rng105: Frequency response of the ideal integrator (DC term aside)..
def intfr(omega):
    """Frequency response of the ideal integrator, eq. (3.116).

        H(w) = 1 / (jw)

    The DC term of eq. (3.115) is set aside, as the book does.  The gain
    falls as the frequency rises, so the integrator is a lowpass -- and
    it is unbounded at w = 0, which is refused here rather than returned
    as an infinity.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    if any(abs(w) <= 1e-300 for w in ws):
        raise ValueError("H(w) = 1/(jw) is unbounded at w = 0; the DC "
                         "content sits in the delta term of eq. (3.115)")
    H = [1.0 / complex(0.0, w) for w in ws]
    return RichResult(payload={
        "H": H[0] if scalar else H, "omega": omega,
        "lowpass": True, "dc_term_set_aside": True,
        "gain_falls_nonlinearly_with_frequency": True,
        "method": "Rangayyan (2024) eq. (3.116)"})


rangayyan_ch3_integrator_frequency_response = intfr  # pre-policy spelling


# -- rng106: Magnitude response of the ideal integrator..
def intmag(omega):
    """Magnitude response of the ideal integrator, eq. (3.117).

        |H(w)| = 1 / |w|

    The book prints 1/w, which is right for w > 0 and is how the response
    is plotted; a magnitude cannot be negative, so the absolute value is
    taken here and the printed form is noted.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    if any(abs(w) <= 1e-300 for w in ws):
        raise ValueError("the magnitude is unbounded at w = 0")
    mag = [1.0 / abs(w) for w in ws]
    return RichResult(payload={
        "magnitude": mag[0] if scalar else mag, "omega": omega,
        "book_prints_one_over_omega": True,
        "absolute_value_needed_for_negative_omega": True,
        "method": "Rangayyan (2024) eq. (3.117)"})


rangayyan_ch3_integrator_magnitude_response = intmag  # pre-policy spelling


# -- rng107: Phase response of the ideal integrator (constant -pi/2)..
def intph(omega):
    """Phase response of the ideal integrator, eq. (3.118).

        angle H(w) = -pi/2

    Constant, because 1/(jw) is a fixed quarter-turn whatever the
    frequency.  A constant phase is NOT a constant delay: the group delay
    is the derivative of the phase, which here is zero, so the integrator
    delays nothing while shifting everything by a quarter cycle.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    if any(abs(w) <= 1e-300 for w in ws):
        raise ValueError("the phase is undefined at w = 0")
    ph = [-pi / 2.0 if w > 0 else pi / 2.0 for w in ws]
    return RichResult(payload={
        "phase": ph[0] if scalar else ph, "omega": omega,
        "constant": True, "group_delay": 0.0,
        "constant_phase_is_not_constant_delay": True,
        "sign_flips_for_negative_omega": True,
        "method": "Rangayyan (2024) eq. (3.118)"})


rangayyan_ch3_integrator_phase_response = intph  # pre-policy spelling


# -- rng108: Recursive form of the 8-point MA filter using delayed output..
def ma8rec(x, n=None):
    """Recursive form of the 8-point MA filter, eq. (3.120).

        y(n) = y(n-1) + (1/8) x(n) - (1/8) x(n-8)

    Obtained by subtracting eq. (3.119) from eq. (3.108).  It computes
    the same output as the direct form with two additions per sample
    instead of eight, and it "clearly depicts the integration aspect of
    the filter" -- the running sum is carried in y(n-1) and only the two
    samples entering and leaving the window are touched.

    The cost is that error accumulates: the recursion never forgets, so a
    single perturbation in y persists for the whole record, where the
    direct form would flush it after eight samples.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    out, acc = [], 0.0
    for i in range(len(xs)):
        acc += 0.125 * xs[i]
        if i >= 8:
            acc -= 0.125 * xs[i - 8]
        out.append(acc)
    direct = [fsum(xs[i - k] for k in range(8) if i - k >= 0) / 8.0
              for i in range(len(xs))]
    gap = max(abs(a - b) for a, b in zip(out, direct))
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "direct_form": direct,
        "max_difference": gap, "agrees_with_direct_form": gap <= 1e-9,
        "additions_per_sample": 2, "direct_form_additions": 8,
        "error_accumulates": True,
        "method": "Rangayyan (2024) eq. (3.120)"})


rangayyan_ch3_ma_8point_recursive = ma8rec  # pre-policy spelling


# -- rng109: Transfer function of the recursive 8-point MA filter (sinc-like)..
def ma8rectf(z):
    """Transfer function of the recursive 8-point MA filter, eq. (3.121).

        H(z) = (1/8) (1 - z^-8) / (1 - z^-1)

    A pole at z = 1 and a zero at z = 1 that cancel it, so the filter is
    still FIR despite the recursive implementation -- the eight zeros of
    the numerator include the one at DC that removes the pole.  At z = 1
    the ratio is 0/0 and the limit is the DC gain, 1; that value is
    returned instead of a division by zero.
    """
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = []
    for zv in zs:
        zc = _cnum(zv)
        if zc == 0:
            raise ValueError("z = 0 is a pole of a causal transfer "
                             "function")
        den = 1.0 - zc ** -1
        if abs(den) <= 1e-12:
            H.append(complex(1.0, 0.0))      # the removable singularity
        else:
            H.append(0.125 * (1.0 - zc ** -8) / den)
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z,
        "pole_at_dc_cancelled_by_a_zero": True,
        "still_fir": True, "dc_gain": 1.0,
        "removable_singularity_at_z_equals_one": True,
        "method": "Rangayyan (2024) eq. (3.121)"})


rangayyan_ch3_ma_8point_recursive_transfer_function = ma8rectf  # pre-policy spelling


# -- rng110: Closed-form sinc-type frequency response of the recursive 8-point MA filter..
def ma8sinc(omega):
    """Sinc-type frequency response of the 8-point MA filter, eq. (3.122).

        H(w) = (1/8) (1 - exp(-j8w)) / (1 - exp(-jw))
             = (1/8) exp(-j 7w/2) sin(4w) / sin(w/2)

    The Dirichlet-kernel identity: a real sinc-like envelope times a pure
    delay of 7/2 samples.  The book states this "is equivalent to that in
    Equation 3.111", and both are recomputed here and compared.  At w = 0
    the ratio is 0/0 with limit 1, which is returned rather than raising.

    The delay of 3.5 samples is not an integer, which is the price of an
    even-length filter: the output falls between input samples.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    closed, direct = [], []
    for w in ws:
        s2 = sin(w / 2.0)
        if abs(s2) <= 1e-12:
            closed.append(complex(1.0, 0.0))
        else:
            closed.append(0.125 * complex(cos(-3.5 * w), sin(-3.5 * w))
                          * sin(4.0 * w) / s2)
        direct.append(0.125 * sum(complex(cos(-w * k), sin(-w * k))
                                  for k in range(8)))
    gap = max(abs(a - b) for a, b in zip(closed, direct))
    return RichResult(payload={
        "H": closed[0] if scalar else closed, "omega": omega,
        "direct_sum": direct[0] if scalar else direct,
        "max_difference": gap, "agrees_with_eq_3_111": gap <= 1e-9,
        "group_delay": 3.5, "delay_is_not_an_integer": True,
        "method": "Rangayyan (2024) eq. (3.122)"})


rangayyan_ch3_ma_8point_sinc_frequency_response = ma8sinc  # pre-policy spelling


# -- rng111: First-order difference operator approximating the time derivative..
def fdiff(x, T=1.0, n=None):
    """First-order difference operator, eq. (3.123).

        y(n) = (1/T) [ x(n) - x(n-1) ]

    The basic DSP derivative.  The 1/T is not cosmetic: without it the
    output is a change PER SAMPLE, not a rate of change per unit time, so
    the numbers depend on the sampling rate.  The book is explicit that
    the scale factor "is required in order to obtain the rate of change
    of the signal with respect to the true time".

    It is a highpass: it removes DC and boosts high frequencies, which is
    also its weakness -- it amplifies noise, and eq. (3.128) is the
    book's remedy.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    out = [(xs[i] - (xs[i - 1] if i >= 1 else 0.0)) / Tv
           for i in range(len(xs))]
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "T": Tv,
        "scale_factor_gives_true_time_rate": True,
        "highpass": True, "amplifies_noise": True,
        "removes_dc": True,
        "method": "Rangayyan (2024) eq. (3.123)"})


rangayyan_ch3_first_difference_operator = fdiff  # pre-policy spelling


# -- rng112: Transfer function of the first-order difference operator..
def fdifftf(z, T=1.0):
    """Transfer function of the first difference, eq. (3.124).

        H(z) = (1/T) (1 - z^-1)

    One zero, at z = 1, which is the DC point -- that single zero is the
    whole of the operator's highpass character.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = [_polyz([1.0 / Tv, -1.0 / Tv], zv) for zv in zs]
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z, "T": Tv,
        "zeros": [1.0], "zero_at_dc": True, "dc_gain": 0.0,
        "method": "Rangayyan (2024) eq. (3.124)"})


rangayyan_ch3_first_difference_transfer_function = fdifftf  # pre-policy spelling


# -- rng113: Frequency response of the first-order difference operator..
def fdifffr(omega, T=1.0):
    """Frequency response of the first difference, eq. (3.125).

        H(w) = (1/T) [1 - exp(-jw)]
             = (1/T) exp(-j w/2) [ 2j sin(w/2) ]

    The second form separates a half-sample delay from a real gain, and
    the factor of j is what puts the phase a quarter turn ahead -- the
    +pi/2 in eq. (3.127).  Both forms are computed and compared.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    raw, split = [], []
    for w in ws:
        raw.append((1.0 - complex(cos(-w), sin(-w))) / Tv)
        split.append(complex(cos(-w / 2.0), sin(-w / 2.0))
                     * complex(0.0, 2.0 * sin(w / 2.0)) / Tv)
    gap = max(abs(a - b) for a, b in zip(raw, split))
    return RichResult(payload={
        "H": raw[0] if scalar else raw, "omega": omega, "T": Tv,
        "split_form": split[0] if scalar else split,
        "max_difference": gap, "forms_agree": gap <= 1e-12,
        "half_sample_delay": 0.5,
        "method": "Rangayyan (2024) eq. (3.125)"})


rangayyan_ch3_first_difference_frequency_response = fdifffr  # pre-policy spelling


# -- rng114: Magnitude response of the first-order difference operator..
def fdiffmag(omega, T=1.0):
    """Magnitude response of the first difference, eq. (3.126).

        |H(w)| = (2/T) |sin(w/2)|

    Nought at DC and largest at Nyquist, rising roughly in proportion to
    frequency over the low end -- which is why the book plots it on a
    linear scale, "in order to illustrate better its proportionality to
    frequency", and why the operator amplifies high-frequency noise.

    The book prints (2/T) sin(w/2) without the bars, which is right on
    0 <= w <= pi, the range plotted; the absolute value is needed for w
    outside it.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    mag = [2.0 * abs(sin(w / 2.0)) / Tv for w in ws]
    return RichResult(payload={
        "magnitude": mag[0] if scalar else mag, "omega": omega, "T": Tv,
        "dc_gain": 0.0, "nyquist_gain": 2.0 / Tv,
        "roughly_proportional_to_frequency": True,
        "book_omits_the_absolute_value": True,
        "method": "Rangayyan (2024) eq. (3.126)"})


rangayyan_ch3_first_difference_magnitude = fdiffmag  # pre-policy spelling


# -- rng115: Phase response of the first-order difference operator..
def fdiffph(omega):
    """Phase response of the first difference, eq. (3.127).

        angle H(w) = pi/2 - w/2

    Linear with slope -1/2, so the group delay is half a sample, plus the
    constant quarter turn contributed by the j of eq. (3.125).  A
    half-sample delay cannot be undone by shifting samples, which is one
    reason the three-point central difference of eq. (3.128) -- whose
    delay is a whole sample -- is easier to align with the original.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    ph = [pi / 2.0 - w / 2.0 for w in ws]
    return RichResult(payload={
        "phase": ph[0] if scalar else ph, "omega": omega,
        "group_delay": 0.5, "slope": -0.5,
        "quarter_turn_offset": pi / 2.0, "linear_phase": True,
        "method": "Rangayyan (2024) eq. (3.127)"})


rangayyan_ch3_first_difference_phase = fdiffph  # pre-policy spelling


# -- rng116: Three-point central-difference operator (lower-noise derivative)..
def cdiff3(x, T=1.0, n=None):
    """Three-point central-difference operator, eq. (3.128).

        y_3(n) = (1/2) [ y(n) + y(n-1) ]
               = (1/(2T)) [ x(n) - x(n-2) ]

    Averaging two successive first differences, which is what controls
    the noise amplification of eq. (3.123).  The book warns the price is
    accuracy: the approximation to d/dt "is poor after about f_s/10", so
    it is a better differentiator only over the low tenth of the band.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    out = [(xs[i] - (xs[i - 2] if i >= 2 else 0.0)) / (2.0 * Tv)
           for i in range(len(xs))]
    # the book's derivation: the mean of two successive first differences
    d1 = fdiff(xs, T=Tv)["y"]
    avg = [0.5 * (d1[i] + (d1[i - 1] if i >= 1 else 0.0))
           for i in range(len(xs))]
    gap = max(abs(a - b) for a, b in zip(out, avg))
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "T": Tv,
        "as_averaged_first_differences": avg, "max_difference": gap,
        "derivation_agrees": gap <= 1e-9,
        "controls_noise_amplification": True,
        "poor_above_fs_over_10": True,
        "method": "Rangayyan (2024) eq. (3.128)"})


rangayyan_ch3_three_point_central_difference = cdiff3  # pre-policy spelling


# -- rng117: Transfer function of the three-point central-difference operator..
def cdiff3tf(z, T=1.0):
    """Transfer function of the three-point central difference,
    eq. (3.129).

        H(z) = (1/(2T)) (1 - z^-2)
             = [ (1/T)(1 - z^-1) ] [ (1/2)(1 + z^-1) ]

    The factored form is the point the book draws out: the operator IS a
    first-order difference in series with a two-point moving average, so
    it may be built as that cascade.  Zeros at z = 1 and z = -1 make it a
    bandpass -- a highpass and a lowpass in series -- with the zero at
    Nyquist pulling the gain there to nought, which is exactly the noise
    amplification the plain difference suffers from.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    direct, cascade = [], []
    for zv in zs:
        zc = _cnum(zv)
        if zc == 0:
            raise ValueError("z = 0 is a pole of a causal transfer "
                             "function")
        direct.append((1.0 - zc ** -2) / (2.0 * Tv))
        cascade.append(((1.0 - zc ** -1) / Tv) * (0.5 * (1.0 + zc ** -1)))
    gap = max(abs(a - b) for a, b in zip(direct, cascade))
    return RichResult(payload={
        "H": direct[0] if scalar else direct, "z": z, "T": Tv,
        "cascade": cascade[0] if scalar else cascade,
        "max_difference": gap, "cascade_agrees": gap <= 1e-12,
        "zeros": [1.0, -1.0], "bandpass": True,
        "is_first_difference_times_two_point_ma": True,
        "method": "Rangayyan (2024) eq. (3.129)"})


rangayyan_ch3_three_point_central_diff_transfer_function = cdiff3tf  # pre-policy spelling


# -- rng118: Magnitude response of the three-point central-difference operator..
def cdiff3mag(omega, T=1.0):
    """Magnitude response of the three-point central difference,
    eq. (3.130).

        |H(w)| = (1/T) |sin(w)|

    Nought at BOTH ends -- at DC from the highpass factor and at Nyquist
    from the moving-average factor -- with the peak at w = pi/2.  That
    second zero is what keeps high-frequency noise from being amplified,
    and it is also why the operator stops approximating a derivative well
    above about f_s/10.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    mag = [abs(sin(w)) / Tv for w in ws]
    return RichResult(payload={
        "magnitude": mag[0] if scalar else mag, "omega": omega, "T": Tv,
        "dc_gain": 0.0, "nyquist_gain": 0.0, "peak_at": pi / 2.0,
        "bandpass": True,
        "method": "Rangayyan (2024) eq. (3.130)"})


rangayyan_ch3_three_point_central_diff_magnitude = cdiff3mag  # pre-policy spelling


# -- rng119: Phase response of the three-point central-difference operator..
def cdiff3ph(omega):
    """Phase response of the three-point central difference, eq. (3.131).

        angle H(w) = pi/2 - w

    Slope -1, so the group delay is a WHOLE sample -- against the half
    sample of the plain first difference.  An integer delay can be undone
    by shifting the output back, which is why this operator is the easier
    one to align with the original recording.
    """
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    ph = [pi / 2.0 - w for w in ws]
    return RichResult(payload={
        "phase": ph[0] if scalar else ph, "omega": omega,
        "group_delay": 1.0, "slope": -1.0,
        "quarter_turn_offset": pi / 2.0,
        "integer_delay_can_be_undone_by_shifting": True,
        "method": "Rangayyan (2024) eq. (3.131)"})


rangayyan_ch3_three_point_central_diff_phase = cdiff3ph  # pre-policy spelling


# -- rng120: Modified first-difference filter with pole at 0.995 to remove baseline wander..
def bwander(z, T=1.0, pole=0.995):
    """Baseline-wander filter, eq. (3.132).

        H(z) = (1/T) (1 - z^-1) / (1 - 0.995 z^-1)

    The first-order difference with a pole placed just inside the unit
    circle on the real axis, at DC.  The pole nearly cancels the zero
    everywhere except in a narrow band about DC, so the filter removes
    the wandering baseline WITHOUT the wholesale high-frequency boost the
    plain difference of eq. (3.123) applies -- its gain is essentially
    flat above a few hertz instead of rising with frequency.

    Moving the pole closer to 1 narrows the notch and lengthens the
    transient; a pole AT 1 would cancel the zero exactly and leave
    nothing.  The book uses 0.995 and that is the default here.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    p = float(pole)
    if not 0.0 <= p < 1.0:
        raise ValueError("the pole must lie inside the unit circle, "
                         "0 <= pole < 1; at 1 it cancels the zero exactly")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H = []
    for zv in zs:
        zc = _cnum(zv)
        if zc == 0:
            raise ValueError("z = 0 is a pole of a causal transfer "
                             "function")
        den = 1.0 - p * zc ** -1
        if abs(den) <= 1e-300:
            raise ValueError("z is the pole of H(z)")
        H.append((1.0 - zc ** -1) / (Tv * den))
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z, "T": Tv, "pole": p,
        "zeros": [1.0], "poles": [p], "dc_gain": 0.0,
        "pole_nearly_cancels_the_zero_away_from_dc": True,
        "no_longer_fir": True,
        "method": "Rangayyan (2024) eq. (3.132)"})


rangayyan_ch3_baseline_wander_filter_z_form_a = bwander  # pre-policy spelling


# -- rng121: Equivalent (z, not z^-1) form of the baseline-wander filter..
def bwanderz(z, T=1.0, pole=0.995):
    """Baseline-wander filter in positive powers of z, eq. (3.133).

        H(z) = (1/T) (z - 1) / (z - 0.995)

    The same filter as eq. (3.132), rearranged.  The book keeps this form
    because it is the one the graphical method reads directly: the
    numerator IS the vector from the evaluation point to the zero at
    z = 1 and the denominator the vector to the pole at 0.995, so the
    magnitude response is the ratio of two lengths measured off the
    z-plane diagram.  Both forms are computed here and compared.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    p = float(pole)
    if not 0.0 <= p < 1.0:
        raise ValueError("the pole must lie inside the unit circle")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    H, other = [], []
    for zv in zs:
        zc = _cnum(zv)
        if abs(zc - p) <= 1e-300:
            raise ValueError("z is the pole of H(z)")
        H.append((zc - 1.0) / (Tv * (zc - p)))
        other.append(bwander(zc, T=Tv, pole=p)["H"])
    gap = max(abs(a - b) for a, b in zip(H, other))
    return RichResult(payload={
        "H": H[0] if scalar else H, "z": z, "T": Tv, "pole": p,
        "max_difference_from_eq_3_132": gap,
        "forms_agree": gap <= 1e-9,
        "numerator_is_the_distance_to_the_zero": True,
        "denominator_is_the_distance_to_the_pole": True,
        "method": "Rangayyan (2024) eq. (3.133)"})


rangayyan_ch3_baseline_wander_filter_z_form_b = bwanderz  # pre-policy spelling


# -- rng122: Time-domain difference equation of the baseline-wander filter..
def bwandereq(x, T=1.0, pole=0.995, n=None):
    """Difference equation of the baseline-wander filter, eq. (3.134).

        y(n) = (1/T) [ x(n) - x(n-1) ] + 0.995 y(n-1)

    Note the PLUS on the feedback term: eq. (3.134) is written with the
    pole's coefficient already moved to the right-hand side, so it does
    not carry the minus that the general form of eq. (3.68) does.
    Reading a sign off eq. (3.68) and applying it here gives a filter
    with a pole at -0.995 instead of +0.995 -- a highpass at Nyquist
    rather than at DC.

    The filter is IIR: the book notes it "is no longer an FIR filter".
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    p = float(pole)
    if not 0.0 <= p < 1.0:
        raise ValueError("the pole must lie inside the unit circle")
    out = []
    for i in range(len(xs)):
        prev_x = xs[i - 1] if i >= 1 else 0.0
        prev_y = out[i - 1] if i >= 1 else 0.0
        out.append((xs[i] - prev_x) / Tv + p * prev_y)
    val = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < len(out):
            raise ValueError("n is outside the record")
        val = out[idx]
    return RichResult(payload={
        "y": out, "value": val, "index": n, "T": Tv, "pole": p,
        "feedback_sign": "+", "iir": True,
        "sign_already_moved_to_the_right_hand_side": True,
        "method": "Rangayyan (2024) eq. (3.134)"})


rangayyan_ch3_baseline_wander_filter_difference_eq = bwandereq  # pre-policy spelling


# -- rng123: Squared-magnitude response of the analog Butterworth lowpass filter..
def bwsqmag(Omega, Omega_c, N):
    """Squared magnitude of the analog Butterworth lowpass, eq. (3.135).

        |H_a(jW)|^2 = 1 / ( 1 + (W/W_c)^{2N} )

    Monotonic in both bands -- no ripple anywhere, which is the defining
    property of the Butterworth and the reason the book chooses it.  At
    the cutoff the squared magnitude is exactly 1/2 FOR EVERY ORDER, so
    W_c is the half-power point regardless of N; raising N steepens the
    transition without moving it.
    """
    Wc = float(Omega_c)
    if Wc <= 0:
        raise ValueError("the cutoff Omega_c must be positive")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    scalar = not isinstance(Omega, (list, tuple))
    ws = [float(Omega)] if scalar else [float(v) for v in Omega]
    sq = [1.0 / (1.0 + (abs(w) / Wc) ** (2 * n)) for w in ws]
    return RichResult(payload={
        "squared_magnitude": sq[0] if scalar else sq,
        "magnitude": (sq[0] ** 0.5) if scalar else [v ** 0.5 for v in sq],
        "Omega": Omega, "Omega_c": Wc, "N": n,
        "half_power_at_cutoff": 0.5, "monotonic": True, "no_ripple": True,
        "cutoff_is_half_power_for_every_order": True,
        "method": "Rangayyan (2024) eq. (3.135)"})


rangayyan_ch3_butterworth_lowpass_squared_magnitude = bwsqmag  # pre-policy spelling


# -- rng124: Squared transfer function of the Butterworth lowpass filter in s-domain..
def bwsqlap(s, Omega_c, N):
    """Squared Butterworth transfer function in the s-domain, eq. (3.136).

        H_a(s) H_a(-s) = 1 / ( 1 + (s / (j W_c))^{2N} )

    Obtained from eq. (3.135) by the substitution jW -> s.  It has 2N
    poles, half of them in the right half-plane, so it is NOT a filter:
    only the N left-half-plane poles are kept, and that selection --
    eq. (3.138) -- is what makes the result stable and causal.
    """
    Wc = float(Omega_c)
    if Wc <= 0:
        raise ValueError("the cutoff Omega_c must be positive")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    scalar = not isinstance(s, (list, tuple))
    ss = [s] if scalar else list(s)
    out = []
    for sv in ss:
        sc = _cnum(sv)
        den = 1.0 + (sc / complex(0.0, Wc)) ** (2 * n)
        if abs(den) <= 1e-300:
            raise ValueError("s is a pole of H_a(s) H_a(-s)")
        out.append(1.0 / den)
    return RichResult(payload={
        "H": out[0] if scalar else out, "s": s, "Omega_c": Wc, "N": n,
        "n_poles": 2 * n, "half_are_right_half_plane": True,
        "not_a_filter_until_the_poles_are_selected": True,
        "method": "Rangayyan (2024) eq. (3.136)"})


rangayyan_ch3_butterworth_squared_laplace = bwsqlap  # pre-policy spelling


# -- rng125: Pole positions on the Butterworth circle in the s-plane..
def bwpoles(Omega_c, N, k=None):
    """Butterworth pole positions, eq. (3.137).

        s_k = W_c exp( j pi [ 1/2 + (2k - 1)/(2N) ] ),   k = 1, ..., 2N

    All 2N poles sit on a circle of radius W_c, spaced pi/N apart, placed
    symmetrically about the imaginary axis and never ON it.  For odd N a
    pole falls on the real axis.  Complex poles come in conjugate pairs,
    which is what keeps the filter coefficients real.

    Both the full set and the N left-half-plane poles -- the ones that
    make a stable causal filter -- are returned; ``k`` selects one pole.
    """
    Wc = float(Omega_c)
    if Wc <= 0:
        raise ValueError("the cutoff Omega_c must be positive")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    allp = []
    for i in range(1, 2 * n + 1):
        ang = pi * (0.5 + (2 * i - 1) / (2.0 * n))
        allp.append(complex(Wc * cos(ang), Wc * sin(ang)))
    lhp = [p for p in allp if p.real < 0]
    val = None
    if k is not None:
        kk = int(k)
        if not 1 <= kk <= 2 * n:
            raise ValueError("k must lie in 1..2N")
        val = allp[kk - 1]
    return RichResult(payload={
        "poles": allp, "left_half_plane": lhp, "value": val, "k": k,
        "Omega_c": Wc, "N": n, "radius": Wc,
        "angular_spacing": pi / n,
        "n_left_half_plane": len(lhp),
        "none_on_the_imaginary_axis": all(abs(p.real) > 1e-12
                                          for p in allp),
        "real_pole_for_odd_order": n % 2 == 1,
        "method": "Rangayyan (2024) eq. (3.137)"})


rangayyan_ch3_butterworth_pole_positions = bwpoles  # pre-policy spelling


# -- rng126: Analog Butterworth transfer function from N left-half-plane poles..
def bwanalog(Omega_c, N, G=None, s=None):
    """Analog Butterworth transfer function, eq. (3.138).

        H_a(s) = G / [ (s - p_1)(s - p_2) ... (s - p_N) ]

    built from the N LEFT-half-plane poles of eq. (3.137) only.  With no
    gain supplied, G is chosen to normalize the DC gain to unity, which
    makes it W_c^N -- the product of the pole magnitudes.

    The denominator coefficients come back expanded and are real to
    within rounding, because the poles arrive in conjugate pairs; the
    largest imaginary residue is returned as a check on that.
    """
    Wc = float(Omega_c)
    if Wc <= 0:
        raise ValueError("the cutoff Omega_c must be positive")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    poles = bwpoles(Wc, n)["left_half_plane"]
    if len(poles) != n:
        raise ValueError("expected %d left-half-plane poles, found %d"
                         % (n, len(poles)))
    coefs = _poly_from_roots(poles)
    resid = max(abs(c.imag) for c in coefs)
    den = [c.real for c in coefs]
    gain = float(G) if G is not None else den[0]
    Hs = None
    if s is not None:
        scalar = not isinstance(s, (list, tuple))
        ss = [s] if scalar else list(s)
        vals = []
        for sv in ss:
            sc = _cnum(sv)
            d = sum(den[i] * sc ** i for i in range(len(den)))
            if abs(d) <= 1e-300:
                raise ValueError("s is a pole of H_a(s)")
            vals.append(gain / d)
        Hs = vals[0] if scalar else vals
    return RichResult(payload={
        "poles": poles, "denominator": den, "gain": gain, "H": Hs,
        "Omega_c": Wc, "N": n,
        "max_imaginary_residue": resid,
        "coefficients_are_real": resid <= 1e-9 * max(1.0, max(
            abs(c) for c in den)),
        "gain_normalizes_dc_to_unity": G is None,
        "left_half_plane_only": True,
        "method": "Rangayyan (2024) eq. (3.138)"})


rangayyan_ch3_butterworth_analog_transfer_function = bwanalog  # pre-policy spelling


# -- rng127: Bilinear transformation mapping s-domain to z-domain..
def bilinear(z, T=1.0):
    """Bilinear transformation, eq. (3.139).

        s = (2/T) (1 - z^-1) / (1 + z^-1)

    Maps the whole left half of the s-plane into the unit disc, so a
    stable analog filter always yields a stable digital one -- unlike
    impulse invariance, which aliases.  The price is that the infinite
    frequency axis is squeezed onto a finite circle, and that squeezing
    is the warping of eqs. (3.141)-(3.142).

    z = -1 maps to infinity, which is refused rather than returned as an
    overflow.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(z, (list, tuple))
    zs = [z] if scalar else list(z)
    out = []
    for zv in zs:
        zc = _cnum(zv)
        if zc == 0:
            raise ValueError("z = 0 is not in the domain of the bilinear "
                             "transformation")
        den = 1.0 + zc ** -1
        if abs(den) <= 1e-300:
            raise ValueError("z = -1 maps to s = infinity")
        out.append((2.0 / Tv) * (1.0 - zc ** -1) / den)
    return RichResult(payload={
        "s": out[0] if scalar else out, "z": z, "T": Tv,
        "maps_lhp_into_the_unit_disc": True,
        "stability_is_preserved": True, "no_aliasing": True,
        "warps_the_frequency_axis": True,
        "method": "Rangayyan (2024) eq. (3.139)"})


rangayyan_ch3_bilinear_transformation = bilinear  # pre-policy spelling


# -- rng128: Bilinear transform restricted to the unit circle (sigma=0)..
def bilinunit(omega, T=1.0):
    """Bilinear transformation on the unit circle, eq. (3.140).

        s = sigma + jW = (2/T) (1 - e^-jw) / (1 + e^-jw)
                       = (2j/T) tan(w/2)

    On the unit circle sigma vanishes exactly, so the imaginary axis of
    the s-plane maps onto the unit circle and nowhere else -- that is the
    property that makes the frequency axes correspond at all.  The
    residual real part is returned as a check; it should be zero to
    rounding.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    direct, closed = [], []
    for w in ws:
        zc = complex(cos(w), sin(w))
        den = 1.0 + zc ** -1
        if abs(den) <= 1e-300:
            raise ValueError("w = pi maps to s = infinity")
        direct.append((2.0 / Tv) * (1.0 - zc ** -1) / den)
        closed.append(complex(0.0, 2.0 * tan(w / 2.0) / Tv))
    gap = max(abs(a - b) for a, b in zip(direct, closed))
    sigma = max(abs(v.real) for v in direct)
    return RichResult(payload={
        "s": direct[0] if scalar else direct, "omega": omega, "T": Tv,
        "closed_form": closed[0] if scalar else closed,
        "max_difference": gap, "forms_agree": gap <= 1e-9,
        "max_real_part": sigma, "sigma_vanishes": sigma <= 1e-9,
        "method": "Rangayyan (2024) eq. (3.140)"})


rangayyan_ch3_bilinear_unit_circle_relation = bilinunit  # pre-policy spelling


# -- rng129: Bilinear frequency warping: analog Omega from discrete omega..
def bilinwarp(omega, T=1.0):
    """Bilinear frequency warping, discrete to analog, eq. (3.141).

        W = (2/T) tan(w/2)

    The prewarping step.  It is nonlinear, so a digital cutoff cannot be
    handed to an analog design unchanged: the whole infinite analog axis
    is compressed into w in (-pi, pi), and the compression is severe near
    Nyquist, where tan blows up.  Skipping this step puts the realized
    cutoff below the one asked for, increasingly so as the cutoff nears
    Nyquist.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    if any(abs(w) >= pi for w in ws):
        raise ValueError("eq. (3.141) needs |w| < pi; w = pi maps to an "
                         "infinite analog frequency")
    out = [(2.0 / Tv) * tan(w / 2.0) for w in ws]
    return RichResult(payload={
        "Omega": out[0] if scalar else out, "omega": omega, "T": Tv,
        "nonlinear": True, "prewarping_is_required": True,
        "compression_is_severe_near_nyquist": True,
        "method": "Rangayyan (2024) eq. (3.141)"})


rangayyan_ch3_bilinear_warping_omega_to_Omega = bilinwarp  # pre-policy spelling


# -- rng130: Bilinear frequency warping: discrete omega from analog Omega..
def bilinunwarp(Omega, T=1.0):
    """Bilinear frequency warping, analog to discrete, eq. (3.142).

        w = 2 arctan( W T / 2 )

    The inverse of eq. (3.141), and the two compose to the identity,
    which is checked here.  Every finite analog frequency lands strictly
    inside (-pi, pi): the mapping is onto the open interval, so no analog
    frequency ever reaches Nyquist.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    scalar = not isinstance(Omega, (list, tuple))
    Ws = [float(Omega)] if scalar else [float(v) for v in Omega]
    out = [2.0 * atan(W * Tv / 2.0) for W in Ws]
    back = [(2.0 / Tv) * tan(w / 2.0) for w in out]
    gap = max(abs(a - b) for a, b in zip(Ws, back)) if Ws else 0.0
    return RichResult(payload={
        "omega": out[0] if scalar else out, "Omega": Omega, "T": Tv,
        "round_trip_error": gap, "inverts_eq_3_141": gap <= 1e-9,
        "always_inside_the_open_interval": all(abs(w) < pi for w in out),
        "method": "Rangayyan (2024) eq. (3.142)"})


rangayyan_ch3_bilinear_warping_Omega_to_omega = bilinunwarp  # pre-policy spelling


# -- rng131: Digital Butterworth transfer function after bilinear transform (IIR form)..
def bwdigital(Omega_c=None, N=None, T=1.0, fc=None, fs=None, z=None):
    """Digital Butterworth after the bilinear transformation, eq. (3.143).

        H(z) = G' (1 + z^-1)^N / sum_{k=0}^{N} a_k z^-k,   a_0 = 1

    The N zeros at z = -1 are not a design choice: the bilinear transform
    puts them there, because s = infinity maps to z = -1 and the analog
    prototype has all its zeros at infinity.  G' is set to make |H(1)| = 1
    at DC.

    Give either the prewarped analog cutoff ``Omega_c``, or a digital
    cutoff ``fc`` with the sampling rate ``fs``, in which case the
    prewarping of eq. (3.141) is applied here rather than being left to
    the caller to forget.
    """
    Tv = float(T)
    if Tv <= 0:
        raise ValueError("the sampling interval T must be positive")
    if N is None:
        raise ValueError("the order N is required")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    if (Omega_c is None) == (fc is None):
        raise ValueError("give either the prewarped Omega_c or a digital "
                         "cutoff fc with fs, not both and not neither")
    if fc is not None:
        if fs is None:
            raise ValueError("fc needs the sampling rate fs")
        fsv, fcv = float(fs), float(fc)
        if not 0 < fcv < fsv / 2.0:
            raise ValueError("the cutoff must lie strictly between 0 and "
                             "the Nyquist frequency")
        Tv = 1.0 / fsv
        Wc = (2.0 / Tv) * tan(pi * fcv / fsv)
        prewarped = True
    else:
        Wc = float(Omega_c)
        if Wc <= 0:
            raise ValueError("the cutoff Omega_c must be positive")
        prewarped = False

    poles_s = bwpoles(Wc, n)["left_half_plane"]
    # map each analog pole through the bilinear transform
    poles_z = []
    for p in poles_s:
        poles_z.append((2.0 / Tv + p) / (2.0 / Tv - p))
    den = [c.real for c in _poly_from_roots(poles_z)]
    den = [c / den[-1] for c in den]              # a_0 = 1 in z^-1 form
    a = list(reversed(den))
    num = [c.real for c in _poly_from_roots([-1.0] * n)]
    b = list(reversed(num))
    dc_num = fsum(b)
    dc_den = fsum(a)
    if abs(dc_num) <= 1e-300:
        raise ValueError("the numerator vanishes at DC")
    Gp = dc_den / dc_num
    b = [Gp * v for v in b]
    Hz = None
    if z is not None:
        scalar = not isinstance(z, (list, tuple))
        zs = [z] if scalar else list(z)
        vals = []
        for zv in zs:
            dd = _polyz(a, zv)
            if abs(dd) <= 1e-300:
                raise ValueError("z is a pole of H(z)")
            vals.append(_polyz(b, zv) / dd)
        Hz = vals[0] if scalar else vals
    return RichResult(payload={
        "b": b, "a": a, "gain": Gp, "poles_z": poles_z, "H": Hz,
        "N": n, "Omega_c": Wc, "T": Tv, "prewarped_here": prewarped,
        "zeros_at_minus_one": n,
        "zeros_are_forced_by_the_bilinear_transform": True,
        "dc_gain": 1.0, "leading_a_is_one": abs(a[0] - 1.0) < 1e-12,
        "method": "Rangayyan (2024) eq. (3.143)"})


rangayyan_ch3_butterworth_digital_transfer_function = bwdigital  # pre-policy spelling


# -- rng132: General time-domain difference equation of an IIR filter..
def iirdiffgen(x, b_k, a_k=None, n=None):
    """General IIR difference equation, eq. (3.144).

        y(n) = sum_{k=0}^{N} b_k x(n-k) - sum_{k=1}^{N} a_k y(n-k)

    The time-domain form of eq. (3.143), and how a designed filter is
    actually run over data.  As in eq. (3.68) the feedback is SUBTRACTED,
    and ``a_k`` is a_1..a_N without the leading a_0 = 1.

    This is the same recursion as ``iirdiff``; it is kept under its own
    name because the book states it separately as the realization step
    that follows the bilinear design.
    """
    return iirdiff(x, b_k, a_k=a_k, n=n)


rangayyan_ch3_iir_difference_eq_general = iirdiffgen  # pre-policy spelling


# -- rng133: Direct discrete-domain specification of the Butterworth lowpass response..
def bwdirect(omega, omega_c, N):
    """Butterworth lowpass specified directly in discrete frequency,
    eq. (3.145).

        |H(w)|^2 = 1 / ( 1 + (w/w_c)^{2N} )

    Specifying the response on the discrete-frequency axis outright, with
    no analog prototype and no bilinear transform, so there is no warping
    to prewarp for: w_c is the cutoff that will actually be realized.

    The filter so defined has zero phase, which is only usable when the
    whole record is in hand and the filtering is done in the frequency
    domain -- it is not causal and cannot be run sample by sample.
    """
    wc = float(omega_c)
    if wc <= 0:
        raise ValueError("the cutoff omega_c must be positive")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    scalar = not isinstance(omega, (list, tuple))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    sq = [1.0 / (1.0 + (abs(w) / wc) ** (2 * n)) for w in ws]
    return RichResult(payload={
        "squared_magnitude": sq[0] if scalar else sq,
        "magnitude": (sq[0] ** 0.5) if scalar else [v ** 0.5 for v in sq],
        "omega": omega, "omega_c": wc, "N": n,
        "half_power_at_cutoff": 0.5, "no_warping": True,
        "zero_phase": True, "not_causal": True,
        "method": "Rangayyan (2024) eq. (3.145)"})


rangayyan_ch3_butterworth_lowpass_direct_specification = bwdirect  # pre-policy spelling


# -- rng134: Butterworth lowpass response indexed by DFT bin k..
def bwlpdft(K, kc=None, N=2, fc=None, fs=None):
    """Butterworth lowpass indexed by DFT bin, eq. (3.146).

        |H(k)|^2 = 1 / ( 1 + (k/k_c)^{2N} )

    valid for k = 0, 1, ..., K/2, with the upper half a reflection,
    H(k) = H(K - k).  The book defines the cutoff index as
    k_c = ceil( K w_c / w_s ), and that CEILING matters: rounding down
    puts the realized cutoff below the one requested.

    Give k_c directly, or a cutoff ``fc`` with the sampling rate ``fs``
    and it is computed the book's way.  The full length-K response is
    returned with the reflection already applied, so it can be multiplied
    straight onto a DFT array.
    """
    Kv = int(K)
    if Kv < 2:
        raise ValueError("the DFT length K must be at least 2")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    if (kc is None) == (fc is None):
        raise ValueError("give either the cutoff index kc or a cutoff fc "
                         "with fs, not both and not neither")
    if fc is not None:
        if fs is None:
            raise ValueError("fc needs the sampling rate fs")
        fsv, fcv = float(fs), float(fc)
        if not 0 < fcv < fsv / 2.0:
            raise ValueError("the cutoff must lie strictly between 0 and "
                             "the Nyquist frequency")
        kcv = int(ceil(Kv * fcv / fsv))
    else:
        kcv = int(kc)
    if kcv < 1:
        raise ValueError("the cutoff index must be at least 1")
    half = Kv // 2
    sq = [1.0 / (1.0 + (k / kcv) ** (2 * n)) for k in range(half + 1)]
    full = list(sq)
    for k in range(half + 1, Kv):
        full.append(sq[Kv - k])
    return RichResult(payload={
        "squared_magnitude": full,
        "magnitude": [v ** 0.5 for v in full],
        "half_spectrum": sq, "K": Kv, "kc": kcv, "N": n,
        "dc_gain": 1.0, "reflected": True,
        "cutoff_index_uses_a_ceiling": True,
        "method": "Rangayyan (2024) eq. (3.146)"})


rangayyan_ch3_butterworth_lowpass_dft_indexed = bwlpdft  # pre-policy spelling


# -- rng135: Butterworth highpass response indexed by DFT bin k..
def bwhpdft(K, kc=None, N=2, fc=None, fs=None):
    """Butterworth highpass indexed by DFT bin, eq. (3.149).

        |H(k)|^2 = 1 / ( 1 + (k_c/k)^{2N} )

    The lowpass of eq. (3.146) with the ratio inverted.  At k = 0 the
    ratio is unbounded and the response is exactly nought, which is the
    whole point -- this is the filter the book uses to strip baseline
    drift from an ECG, eighth order with a 2 Hz cutoff.

    The book is careful that removing the low-frequency artifact leaves
    high-frequency noise untouched; a highpass is not a denoiser.
    """
    Kv = int(K)
    if Kv < 2:
        raise ValueError("the DFT length K must be at least 2")
    n = int(N)
    if n < 1:
        raise ValueError("the order N must be at least 1")
    if (kc is None) == (fc is None):
        raise ValueError("give either the cutoff index kc or a cutoff fc "
                         "with fs, not both and not neither")
    if fc is not None:
        if fs is None:
            raise ValueError("fc needs the sampling rate fs")
        fsv, fcv = float(fs), float(fc)
        if not 0 < fcv < fsv / 2.0:
            raise ValueError("the cutoff must lie strictly between 0 and "
                             "the Nyquist frequency")
        kcv = int(ceil(Kv * fcv / fsv))
    else:
        kcv = int(kc)
    if kcv < 1:
        raise ValueError("the cutoff index must be at least 1")
    half = Kv // 2
    sq = [0.0]
    for k in range(1, half + 1):
        sq.append(1.0 / (1.0 + (kcv / k) ** (2 * n)))
    full = list(sq)
    for k in range(half + 1, Kv):
        full.append(sq[Kv - k])
    return RichResult(payload={
        "squared_magnitude": full,
        "magnitude": [v ** 0.5 for v in full],
        "half_spectrum": sq, "K": Kv, "kc": kcv, "N": n,
        "dc_gain": 0.0, "reflected": True,
        "leaves_high_frequency_noise_untouched": True,
        "method": "Rangayyan (2024) eq. (3.149)"})


rangayyan_ch3_butterworth_highpass_dft_indexed = bwhpdft  # pre-policy spelling


# -- rng136: Notch filter with two zeros at 60 Hz on the unit circle..
def notch60(fs, f0=60.0, z=None):
    """Notch filter with two zeros on the unit circle, Section 3.7.

        H(z) = G ( 1 - 2 cos(w_0) z^-1 + z^-2 ),   w_0 = 2 pi f_0 / f_s

    A conjugate pair of zeros AT the interference frequency, so the gain
    there is exactly nought.  With zeros alone the notch is wide: the
    response is pulled down over a broad band either side, which is why
    the book goes on to add poles just inside the zeros to narrow it.

    G normalizes the DC gain to unity.  The zeros lie on the unit circle,
    so this filter is FIR and has exactly linear phase.
    """
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    f0v = float(f0)
    if not 0 < f0v < fsv / 2.0:
        raise ValueError("the notch frequency must lie strictly between 0 "
                         "and the Nyquist frequency")
    w0 = 2.0 * pi * f0v / fsv
    b = [1.0, -2.0 * cos(w0), 1.0]
    dc = fsum(b)
    if abs(dc) <= 1e-300:
        raise ValueError("the notch sits at DC; the gain cannot be "
                         "normalized there")
    G = 1.0 / dc
    b = [G * v for v in b]
    Hz = None
    if z is not None:
        scalar = not isinstance(z, (list, tuple))
        zs = [z] if scalar else list(z)
        vals = [_polyz(b, zv) for zv in zs]
        Hz = vals[0] if scalar else vals
    zeros = [complex(cos(w0), sin(w0)), complex(cos(w0), -sin(w0))]
    return RichResult(payload={
        "b": b, "a": [1.0], "gain": G, "zeros": zeros, "H": Hz,
        "f0": f0v, "fs": fsv, "omega_0": w0,
        "gain_at_the_notch": abs(_polyz(b, complex(cos(w0), sin(w0)))),
        "dc_gain": 1.0, "fir": True, "linear_phase": True,
        "notch_is_wide_without_poles": True,
        "method": "Rangayyan (2024) Section 3.7 (notch filter with two "
                  "zeros)"})


rangayyan_ch3_notch_filter_60Hz = notch60  # pre-policy spelling


# -- rng226: Matched-filter impulse response for the basic pattern g(n)..
def mfilth(g, normalize=False):
    """Matched-filter impulse response for a pattern g(n), Chapter 4.

        h(n) = g(N - 1 - n)

    The template reversed in time, which is what makes the filter's
    output at each instant the cross-correlation of the signal with the
    template.  Reversal is the whole content: convolving with the
    unreversed template correlates with a mirrored pattern and peaks in
    the wrong place.

    The peak lands at index N - 1 when the template is aligned with the
    start of the record, not at 0; that offset is returned so it need not
    be rediscovered.  With ``normalize`` the response is scaled to unit
    energy, which makes outputs comparable across templates.
    """
    gs = aslist(g)
    if not gs:
        raise ValueError("the template needs at least one sample")
    h = list(reversed(gs))
    energy = fsum(v * v for v in gs)
    if normalize:
        if energy <= 0:
            raise ValueError("a template with no energy cannot be "
                             "normalized")
        h = [v / (energy ** 0.5) for v in h]
    return RichResult(payload={
        "h": h, "template": list(gs), "n": len(h),
        "energy": energy, "normalized": bool(normalize),
        "peak_index": len(gs) - 1, "time_reversed": True,
        "output_is_the_cross_correlation": True,
        "method": "Rangayyan (2024) Ch. 4 (matched filter)"})


rangayyan_ch4_matched_filter_h_example = mfilth  # pre-policy spelling


_CHEATSHEET = [
    'Butterworth highpass design',
    'Butterworth lowpass design, eqs. (3.135)-(3.143)',
    'comb filter, notches at every multiple of fs/N',
    'first difference applied to a record',
    'second-order difference operator',
    'rgfir: FIR filter design (windowed sinc) -- see rangayyan_fir_filter for sources.',
    'frequency response from filter coefficients',
    'group delay from the unwrapped phase',
    'rgiir: IIR Butterworth filter -- Rangayyan & Krishnan Sec 3.7.1 / 3.7.2.',
    'rgmavg: Moving-average filter.',
    'notch filter with two zeros and two poles',
    'rgosflt: Order-statistic (median) filter.',
    'phase response, unwrapped by default',
    'ideal sinc lowpass kernel, optionally windowed',
    'rgtfe: Transfer function estimate.',
    'Blackman window',
    'Hamming window',
    'Hann window',
    'window functions: rectangular, Hann, Hamming, Blackman',
    'rng011: Shannon entropy of a discrete process (Rangayyan eq. 3.11).',
    'rng039: 11-point moving average.',
    'rng040: Linear-ramp smoothing filter (Rangayyan eq. 3.42).',
    'rng043: Combined impulse response of two LSI systems in series is their convolution..',
    'rng047: Combined impulse response of two LSI systems in parallel is their sum..',
    'rng048: Bilateral Laplace transform of an impulse response h(t)..',
    'rng050: Frequency response obtained by evaluating the Laplace transform on the imaginary axis..',
    'rng053: Z-transform of a causal FIR system of length N (transfer function).',
    'rng056: Generic rational transfer function of an IIR filter..',
    'rng057: Time-domain difference equation form of an IIR filter..',
    'rng061: Magnitude response from products of distances to zeros and poles..',
    'rng062: Phase response from sums of angles to zeros and poles..',
    'rng087: General FIR filter.',
    'rng088: Transfer function of a generic MA (FIR) filter of order N..',
    'rng089: Time-domain difference equation of the von Hann (Hanning) smoothing filter..',
    'rng090: Impulse response of the Hann smoothing filter..',
    'rng091: Z-domain expression for the Hann filter output..',
    'rng092: Transfer function of the Hann filter (double zero at z=-1)..',
    'rng093: Frequency response of the Hann filter on the unit circle..',
    'rng094: Simplified closed-form frequency response of the Hann filter..',
    'rng095: Magnitude response of the Hann filter..',
    'rng096: Linear phase response of the Hann filter..',
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
