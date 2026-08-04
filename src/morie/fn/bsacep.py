# morie.fn -- bsacep (rootcoder007/morie)
"""Cepstral and homomorphic analysis: real and complex cepstra, liftering, deconvolution, echo removal.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 23
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from math import atan2 as _atan2, cos, exp, fsum, log, log10, pi, sin, sqrt
import math as _math
from . import _array_core as np
from ._rgcore import aslist
from ._richresult import RichResult

__all__ = [
    'rangayyan_ar_to_cepstrum',
    'ccepx',
    'rangayyan_complex_cepstrum',
    'rangayyan_cepstrum_pitch',
    'cepstrum',
    'rangayyan_cepstrum',
    'homdeconv',
    'rangayyan_homomorphic_deconv',
    'homofilt',
    'rangayyan_homomorphic',
    'hompred',
    'rangayyan_homomorphic_pred',
    'lifter',
    'rangayyan_liftering',
    'mfcc',
    'rangayyan_mfcc',
    'minphase',
    'rangayyan_min_phase',
    'vocaltract',
    'rangayyan_vocal_tract',
    'multmodel',
    'rangayyan_ch4_homomorphic_multiplicative',
    'logsep',
    'rangayyan_ch4_homomorphic_log_separation',
    'convmodel',
    'rangayyan_ch4_convolution_model',
    'ccepstrum',
    'rangayyan_ch4_complex_cepstrum_definition',
    'ccepsum',
    'rangayyan_ch4_complex_cepstra_sum',
    'ratz',
    'rangayyan_ch4_rational_z_transform_form',
    'ccepclosed',
    'rangayyan_ch4_complex_cepstrum_closed_form',
    'ccepdecay',
    'rangayyan_ch4_complex_cepstrum_decay_bound',
    'echoseries',
    'rangayyan_ch4_log_echo_power_series',
    'pcepstrum',
    'rangayyan_ch4_power_cepstrum_definition',
    'pcepsum',
    'rangayyan_ch4_power_cepstrum_sum',
    'pceprel',
    'rangayyan_ch4_power_cepstrum_relation',
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
def ccepx(x):
    """Complex cepstrum with the phase-unwrapping detail exposed.

    Rangayyan (2024) eqs. (4.63)-(4.64).  Same computation as
    :func:`ccepstrum`, delegating to it so the two cannot drift, but
    also returning the wrapped phase beside the unwrapped one and the
    count of 2-pi jumps that had to be removed.  The book calls phase
    unwrapping "an important consideration in the evaluation of the
    complex logarithm", and the jump count is how a caller sees whether
    the unwrapping was well conditioned: a jump at nearly every bin
    means the spectrum is too coarsely sampled for the phase to be
    tracked, and the cepstrum should not be trusted.
    """
    r = ccepstrum(x)
    phase = r["phase"]
    wrapped = [_atan2(sin(p), cos(p)) for p in phase]
    jumps = sum(1 for i in range(1, len(wrapped))
                if abs(wrapped[i] - wrapped[i - 1]) > pi)
    out = dict(r)
    out.update({"wrapped_phase": wrapped, "phase_jumps": jumps,
                "well_conditioned": jumps < len(wrapped) // 4,
                "method": "Rangayyan (2024) eqs. (4.63)-(4.64), with the "
                          "phase-unwrapping diagnostics"})
    return RichResult(payload=out)


rangayyan_complex_cepstrum = ccepx  # pre-policy spelling


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
def _fft_pair(x):
    """DFT of a real sequence, returned as (real, imag) lists."""
    n = len(x)
    step = 2.0 * pi / n
    re, im = [], []
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(x)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(x)))
    return re, im


def _ifft_real(re, im):
    """Inverse DFT, real part only."""
    n = len(re)
    step = 2.0 * pi / n
    out = []
    for i in range(n):
        acc = 0.0
        for k in range(n):
            ang = step * i * k
            acc += re[k] * cos(ang) - im[k] * sin(ang)
        out.append(acc / n)
    return out


def _unwrap(phase):
    """Remove 2-pi jumps from a principal-value phase sequence."""
    out = [phase[0]] if phase else []
    off = 0.0
    for i in range(1, len(phase)):
        d = phase[i] - phase[i - 1]
        while d > pi:
            off -= 2.0 * pi
            d -= 2.0 * pi
        while d < -pi:
            off += 2.0 * pi
            d += 2.0 * pi
        out.append(phase[i] + off)
    return out


def cepstrum(x):
    """Real cepstrum: inverse DFT of the log magnitude spectrum.

    c(n) = IDFT( log |DFT(x)| ).

    The real cepstrum keeps only the magnitude, so it discards the phase
    and is NOT invertible -- that is the whole difference from the
    complex cepstrum of eq. (4.64), which keeps log|X| + j angle(X) and
    can be inverted.  The real cepstrum still shows the echo impulses of
    eq. (4.80), which is what it is used for.

    Zero-magnitude bins would send the logarithm to -inf, so they are
    floored at a tiny epsilon and the count is reported: a spectrum with
    many exact zeros means the cepstrum is being taken of something the
    method does not apply to.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    re, im = _fft_pair(xs)
    mags = [sqrt(a * a + b * b) for a, b in zip(re, im)]
    floor = 1e-300
    zeros = sum(1 for v in mags if v <= floor)
    logmag = [log(v if v > floor else floor) for v in mags]
    c = _ifft_real(logmag, [0.0] * n)
    return RichResult(payload={
        "cepstrum": c, "log_magnitude": logmag, "n": n,
        "zero_bins": zeros, "invertible": False,
        "method": "real cepstrum; contrast Rangayyan (2024) eq. (4.64)"})


rangayyan_cepstrum = cepstrum  # pre-policy spelling


# -- rghomdc: Homomorphic deconvolution via complex cepstrum.
def homdeconv(y, cutoff, keep="low"):
    """Homomorphic deconvolution via the complex cepstrum.

    Rangayyan (2024) Section 4.7.2, Figures 4.24-4.25, eqs. (4.61)-(4.66)
    and the speech illustration under eq. (4.85).  The chain is

        DFT -> complex log -> IDFT -> lifter -> DFT -> exp -> IDFT,

    which converts convolution to addition (eqs. 4.62-4.63), separates
    the summands by quefrency (eq. 4.66), and reverses every step.

    Keeping the low-quefrency part estimates the slowly varying
    component -- the vocal-tract response in speech, the basic wavelet
    in the echo problem; keeping the high-quefrency part estimates the
    excitation.

    The output is real to within rounding when the input is real, since
    the book notes the complex cepstrum of a real signal is real; the
    residual imaginary energy is returned as a check that the phase
    unwrapping held.
    """
    ys = aslist(y)
    n = len(ys)
    if n < 8:
        raise ValueError("need at least eight samples")
    cep = ccepstrum(ys)
    c = cep["cepstrum"]
    lf = lifter(c, high=cutoff, low=cutoff, keep=keep)["liftered"]
    re, im = _fft_pair(lf)
    # exponentiate the complex log spectrum, then invert
    out_re, out_im = [], []
    for a, b in zip(re, im):
        m = exp(a)
        out_re.append(m * cos(b))
        out_im.append(m * sin(b))
    rec = _ifft_real(out_re, out_im)
    imag_energy = fsum(v * v for v in out_im)
    return RichResult(payload={
        "y": rec, "cepstrum": c, "liftered": lf, "cutoff": int(cutoff),
        "keep": keep, "n": n,
        "linear_phase_removed": cep["linear_phase_removed"],
        "imaginary_energy": imag_energy,
        "stages": ("DFT", "complex log", "IDFT", "lifter", "DFT", "exp",
                   "IDFT"),
        "method": "Rangayyan (2024) Section 4.7.2, eqs. (4.61)-(4.66)"})


rangayyan_homomorphic_deconv = homdeconv  # pre-policy spelling


# -- rghomo: Homomorphic filtering system for multiplicative signal models.
def homofilt(y, cutoff, keep="low"):
    """Multiplicative homomorphic filter: log, filter, exponentiate.

    Rangayyan (2024) Section 4.7.1, Figure 4.23, eqs. (4.58)-(4.60).
    The three stages are the whole method:

        log  ->  linear filter  ->  exp,

    which turns a product into a sum, separates the summands with an
    ordinary linear filter, and returns to the original space.  The book
    uses it for images as illumination times reflectance, where the
    illumination is the slowly varying (low-frequency) factor.

    The signal must be strictly positive: the real logarithm of stage
    one demands it (eq. 4.59's side condition), and a signal that
    crosses zero needs the complex-log route of :func:`homdeconv`
    instead.  Rejected rather than clipped, because clipping to a floor
    would silently change the factorization being estimated.
    """
    ys = aslist(y)
    n = len(ys)
    if n < 4:
        raise ValueError("need at least four samples")
    if any(v <= 0 for v in ys):
        raise ValueError("the multiplicative homomorphic filter needs a "
                         "strictly positive signal (eq. 4.59); use the "
                         "complex-cepstrum route for signed data")
    if keep not in ("low", "high"):
        raise ValueError("keep must be 'low' or 'high'")
    k = int(cutoff)
    if not 0 <= k <= n // 2:
        raise ValueError("cutoff must lie in 0..N/2")
    ly = [log(v) for v in ys]
    re, im = _fft_pair(ly)
    for i in range(n):
        f = min(i, n - i)
        take = (f <= k) if keep == "low" else (f > k)
        if not take:
            re[i] = 0.0
            im[i] = 0.0
    filtered = _ifft_real(re, im)
    return RichResult(payload={
        "y": [exp(v) for v in filtered], "log_domain": filtered,
        "log_input": ly, "cutoff": k, "keep": keep, "n": n,
        "stages": ("log", "linear filter", "exp"),
        "method": "Rangayyan (2024) Section 4.7.1, eqs. (4.58)-(4.60)"})


rangayyan_homomorphic = homofilt  # pre-policy spelling


# -- rghompr: Homomorphic prediction via complex cepstrum.
def hompred(y, cutoff):
    """Split a signal into its low-time and high-time cepstral parts.

    Rangayyan (2024) Section 4.7.3 and the illustration under eq. (4.85):
    for voiced speech, the vocal-tract response h(n) contributes to the
    complex cepstrum only at low quefrency, within the pitch period,
    while the glottal excitation appears as impulses AT the pitch period
    and its multiples.  Liftering on either side of ``cutoff`` and
    inverting therefore estimates the two components separately.

    Both are returned together, and their convolution is compared
    against the input: eq. (4.66) says the two cepstra add, so the two
    estimated components should convolve back to the signal.  The
    reconstruction error is the honest measure of how well the
    quefrency ranges actually separated -- the book's premise is that
    they do not overlap, and on a real signal they partly do.
    """
    ys = aslist(y)
    n = len(ys)
    if n < 8:
        raise ValueError("need at least eight samples")
    k = int(cutoff)
    if not 1 <= k < n // 2:
        raise ValueError("cutoff must lie in 1..N/2-1")
    # the two lifters must PARTITION the quefrency axis: |q| <= k and
    # |q| > k.  Using the same cutoff for both keeps q = k (and q = 0) in
    # each half, so the components double-count it and their convolution
    # no longer returns the signal.
    low = homdeconv(ys, k, keep="low")["y"]
    high = homdeconv(ys, k + 1, keep="high")["y"]
    # the cepstra add (eq. 4.66), so exponentiating the sum gives the
    # CIRCULAR convolution of the two components -- the DFT works on the
    # circle.  Reconstructing with a linear convolution instead leaves a
    # wrap-around error that looks like a failure of the separation.
    conv = [fsum(low[j] * high[(i - j) % n] for j in range(n))
            for i in range(n)]
    err = max(abs(a - b) for a, b in zip(conv, ys))
    scale = max(abs(v) for v in ys) or 1.0
    return RichResult(payload={
        "low_time": low, "high_time": high, "cutoff": k, "n": n,
        "reconstruction": conv, "reconstruction_error": err,
        "relative_error": err / scale,
        "separation_premise": "eq. (4.66) assumes the two components "
                              "occupy non-overlapping quefrency ranges",
        "method": "Rangayyan (2024) Section 4.7.3"})


rangayyan_homomorphic_pred = hompred  # pre-policy spelling


# -- rglift: Cepstral liftering (low-time / high-time separation).
def lifter(cepstrum_values, low=None, high=None, keep="low"):
    """Cepstral liftering: window the cepstrum by quefrency.

    Rangayyan (2024) Section 4.7.3 and the speech illustration under eq.
    (4.85): the vocal-tract response contributes to the complex cepstrum
    only at LOW quefrency, within the pitch period, while the glottal
    pulse train appears as impulses at the pitch period and its
    multiples.  Selecting a low-quefrency window and inverting therefore
    estimates the vocal-tract response; selecting the high-quefrency
    part estimates the excitation.

    The window is applied SYMMETRICALLY about quefrency zero, because
    the cepstrum of a mixed-phase signal is two-sided (eq. 4.72) and
    keeping only the causal half would discard the maximum-phase
    component and change the signal.

    Parameters
    ----------
    cepstrum_values : array-like
        A cepstrum indexed n = 0, 1, ..., N-1 with negative quefrencies
        wrapped to the upper half, as the DFT returns them.
    low, high : int, optional
        Window edges in quefrency samples.
    keep : {"low", "high", "band"}
        Which region to retain.
    """
    c = aslist(cepstrum_values)
    n = len(c)
    if n < 2:
        raise ValueError("need at least two cepstral samples")
    half = n // 2
    lo = 0 if low is None else int(low)
    hi = half if high is None else int(high)
    if lo < 0 or hi < 0:
        raise ValueError("quefrency limits must be nonnegative")
    if hi < lo:
        raise ValueError("high must not be below low")
    if keep not in ("low", "high", "band"):
        raise ValueError("keep must be 'low', 'high' or 'band'")

    def quef(i):
        return i if i <= half else i - n

    out = []
    for i, v in enumerate(c):
        q = abs(quef(i))
        if keep == "low":
            take = q <= hi
        elif keep == "high":
            take = q >= lo
        else:
            take = lo <= q <= hi
        out.append(v if take else 0.0)
    kept = sum(1 for i in range(n) if out[i] != 0.0 or c[i] == 0.0)
    return RichResult(payload={
        "liftered": out, "n": n, "low": lo, "high": hi, "keep": keep,
        "symmetric": True, "n_kept": kept,
        "energy_kept": (fsum(v * v for v in out)
                        / fsum(v * v for v in c)) if any(c) else 0.0,
        "method": "Rangayyan (2024) Section 4.7.3 (cepstral liftering)"})


rangayyan_liftering = lifter  # pre-policy spelling


# -- rgmfcc: Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis.
def mfcc(x, fs, n_filters=26, n_coeffs=13, fmin=0.0, fmax=None):
    """Mel-frequency cepstral coefficients.

    Davis and Mermelstein (1980), IEEE Trans. ASSP 28(4):357-366.  The
    chain is: power spectrum, triangular filterbank on the mel scale,
    logarithm of each band energy, then a DCT-II:

        mel(f) = 2595 log10(1 + f / 700)
        MFCC   = DCT( log(filterbank energies) ).

    This is a cepstrum in the sense of Rangayyan (2024) Section 4.7 --
    an inverse transform of a log spectrum -- but with two departures
    the book's cepstrum does not make: the frequency axis is warped to
    the mel scale, and the DCT replaces the inverse DFT.  Both are
    perceptual choices from speech recognition, not properties of the
    homomorphic system, so this is cited to Davis and Mermelstein.

    Coefficient 0 is the log total energy; it is returned but is usually
    dropped or replaced, since it tracks recording gain rather than
    spectral shape.
    """
    xs = aslist(x)
    n = len(xs)
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if n < 8:
        raise ValueError("need at least eight samples")
    nf = int(n_filters)
    nc = int(n_coeffs)
    if nf < 2:
        raise ValueError("need at least two mel filters")
    if not 1 <= nc <= nf:
        raise ValueError("n_coeffs must lie in 1..n_filters")
    top = fsv / 2.0 if fmax is None else float(fmax)
    if not 0 <= fmin < top <= fsv / 2.0:
        raise ValueError("need 0 <= fmin < fmax <= fs/2")
    re, im = _fft_pair(xs)
    half = n // 2 + 1
    power = [(re[k] ** 2 + im[k] ** 2) / n for k in range(half)]
    freqs = [k * fsv / n for k in range(half)]

    def to_mel(f):
        return 2595.0 * log10(1.0 + f / 700.0)

    def from_mel(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_lo, m_hi = to_mel(fmin), to_mel(top)
    edges = [from_mel(m_lo + (m_hi - m_lo) * i / (nf + 1))
             for i in range(nf + 2)]
    energies = []
    for i in range(nf):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        acc = 0.0
        for f, p in zip(freqs, power):
            if lo <= f <= mid and mid > lo:
                acc += p * (f - lo) / (mid - lo)
            elif mid < f <= hi and hi > mid:
                acc += p * (hi - f) / (hi - mid)
        energies.append(acc)
    floor = 1e-300
    empty = sum(1 for v in energies if v <= floor)
    logs = [log(v if v > floor else floor) for v in energies]
    coeffs = []
    for k in range(nc):
        coeffs.append(fsum(logs[i] * cos(pi * k * (i + 0.5) / nf)
                           for i in range(nf)))
    return RichResult(payload={
        "mfcc": coeffs, "filterbank_energies": energies,
        "log_energies": logs, "edges": edges, "n_filters": nf,
        "n_coeffs": nc, "fs": fsv, "empty_filters": empty,
        "c0_is_energy": True,
        "method": "Davis and Mermelstein (1980); a mel-warped, DCT-based "
                  "cepstrum, not the homomorphic cepstrum of Rangayyan "
                  "(2024) Section 4.7"})


rangayyan_mfcc = mfcc  # pre-policy spelling


# -- rgminph: Minimum-phase correspondent of a signal.
def minphase(x):
    """Minimum-phase correspondent of a signal.

    Rangayyan (2024) Section 4.7.2, after eq. (4.73): a minimum-phase
    signal has all its poles and zeros inside the unit circle and hence
    a CAUSAL complex cepstrum, while a maximum-phase signal has an
    anticausal one; a mixed-phase signal separates into the two by
    taking the causal (n > 0) and anticausal (n < 0) parts of its
    complex cepstrum and inverting.

    The minimum-phase correspondent is built by that route: fold the
    anticausal half of the cepstrum onto the causal half, which reflects
    every root outside the unit circle to its reciprocal inside without
    changing the magnitude spectrum.  Magnitude preservation is the
    check returned; it is what "correspondent" means.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 8:
        raise ValueError("need at least eight samples")
    c = ccepstrum(xs)["cepstrum"]
    half = n // 2
    folded = [0.0] * n
    folded[0] = c[0]
    for i in range(1, half):
        folded[i] = c[i] + c[n - i]
    if n % 2 == 0:
        folded[half] = c[half]
    re, im = _fft_pair(folded)
    out_re, out_im = [], []
    for a, b in zip(re, im):
        m = exp(a)
        out_re.append(m * cos(b))
        out_im.append(m * sin(b))
    y = _ifft_real(out_re, out_im)
    mre, mim = _fft_pair(xs)
    src = [sqrt(a * a + b * b) for a, b in zip(mre, mim)]
    dre, dim = _fft_pair(y)
    dst = [sqrt(a * a + b * b) for a, b in zip(dre, dim)]
    gap = max(abs(a - b) for a, b in zip(src, dst))
    scale = max(src) or 1.0
    return RichResult(payload={
        "y": y, "cepstrum": c, "n": n,
        "magnitude_error": gap, "magnitude_preserved": gap <= 1e-6 * scale,
        "energy_front_loaded": fsum(v * v for v in y[:half])
        >= fsum(v * v for v in y[half:]),
        "method": "Rangayyan (2024) Section 4.7.2 (minimum-phase "
                  "correspondent from the causal cepstrum)"})


rangayyan_min_phase = minphase  # pre-policy spelling


# -- rgvocal: Vocal tract transfer function extraction via homomorphic deconvolution.
def vocaltract(y, fs, pitch_period=None, cutoff=None,
               pitch_range=(0.002, 0.020)):
    """Vocal-tract response from the low-quefrency cepstrum.

    Rangayyan (2024) Section 4.7.3, under eq. (4.85): a voiced speech
    signal is the convolution of a slowly varying vocal-tract response
    with a faster glottal pulse train, so the tract contributes to the
    complex cepstrum only for quefrencies below the pitch period, where
    the excitation's impulses begin.

    The lifter cutoff is therefore set from the pitch period, not chosen
    arbitrarily: the default is 90% of the pitch period in samples, just
    short of the first excitation impulse.  With no pitch supplied, the
    period is estimated as the quefrency of the largest cepstral peak
    inside ``pitch_range`` -- the standard cepstral pitch detector, and
    the reason the cepstrum was invented.

    The range matters: eq. (4.80) puts impulses at the delay AND ITS
    MULTIPLES, and for a strongly periodic signal the rahmonic at 2 n0
    or 3 n0 can be the larger peak.  Searching the whole cepstrum
    therefore reports a pitch two or three times too low.  The default
    2-20 ms covers 50-500 Hz, which spans human voiced speech.
    """
    ys = aslist(y)
    n = len(ys)
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if n < 16:
        raise ValueError("need at least sixteen samples")
    cep = ccepstrum(ys)["cepstrum"]
    half = n // 2
    lo_q = max(1, int(pitch_range[0] * fsv))
    hi_q = min(half, int(pitch_range[1] * fsv) + 1)
    if hi_q <= lo_q:
        raise ValueError("the pitch range holds no quefrency bins at this "
                         "sampling rate and record length")
    if pitch_period is None:
        cand = [(abs(cep[i]), i) for i in range(lo_q, hi_q)]
        if not cand:
            raise ValueError("the record is too short to hold a pitch peak "
                             "in the requested range")
        peak = max(cand)[1]
        period = peak / fsv
    else:
        period = float(pitch_period)
        peak = int(round(period * fsv))
    k = int(cutoff) if cutoff is not None else max(1, int(0.9 * peak))
    if k >= half:
        raise ValueError("the lifter cutoff exceeds the usable quefrency "
                         "range")
    est = homdeconv(ys, k, keep="low")
    return RichResult(payload={
        "response": est["y"], "cepstrum": cep, "cutoff": k,
        "pitch_period": period, "pitch_hz": 1.0 / period if period > 0
        else None, "peak_quefrency": peak, "fs": fsv, "n": n,
        "method": "Rangayyan (2024) Section 4.7.3 (vocal-tract response "
                  "by low-time liftering)"})


rangayyan_vocal_tract = vocaltract  # pre-policy spelling


# -- rng230: Multiplicative model addressed by homomorphic filtering..
def multmodel(x, p):
    """The multiplicative signal model of homomorphic filtering.

    Rangayyan (2024) eq. (4.58):  y(t) = x(t) p(t).

    The model a multiplicative homomorphic system addresses: two signals
    combined by multiplication rather than addition, so a linear filter
    cannot separate them until the logarithm of eq. (4.59) turns the
    product into a sum.  The book's example is an image as illumination
    times reflectance.
    """
    xs, ps = aslist(x), aslist(p)
    if len(xs) != len(ps):
        raise ValueError("x and p must have the same length")
    if not xs:
        raise ValueError("need at least one sample")
    y = [a * b for a, b in zip(xs, ps)]
    return RichResult(payload={
        "y": y, "x": xs, "p": ps, "n": len(y),
        "separable_by_log": all(v != 0 for v in xs)
        and all(v != 0 for v in ps),
        "method": "Rangayyan (2024) eq. (4.58)"})


rangayyan_ch4_homomorphic_multiplicative = multmodel  # pre-policy spelling


# -- rng231: Logarithm converts product into a sum in homomorphic filtering..
def logsep(x, p):
    """The logarithm turns a product into a sum.

    Rangayyan (2024) eq. (4.59):
        log[y(t)] = log[x(t) p(t)] = log[x(t)] + log[p(t)],
        for x(t) != 0 and p(t) != 0 for all t.

    The book states that side condition explicitly, so a zero in either
    factor is rejected rather than producing -inf; a negative factor is
    rejected too, since the real logarithm is undefined there and the
    complex-log route of eq. (4.63) is the method for signed data.
    Both sides are formed and compared.
    """
    xs, ps = aslist(x), aslist(p)
    if len(xs) != len(ps):
        raise ValueError("x and p must have the same length")
    if not xs:
        raise ValueError("need at least one sample")
    if any(v <= 0 for v in xs) or any(v <= 0 for v in ps):
        raise ValueError("eq. (4.59) needs x(t) != 0 and p(t) != 0; the real "
                         "logarithm also needs them positive")
    y = [a * b for a, b in zip(xs, ps)]
    lhs = [log(v) for v in y]
    rhs = [log(a) + log(b) for a, b in zip(xs, ps)]
    gap = max(abs(u - v) for u, v in zip(lhs, rhs))
    return RichResult(payload={
        "log_y": lhs, "log_x": [log(v) for v in xs],
        "log_p": [log(v) for v in ps], "sum": rhs,
        "max_difference": gap, "additive": gap <= 1e-12 * (1 + max(
            abs(v) for v in lhs)),
        "method": "Rangayyan (2024) eq. (4.59)"})


rangayyan_ch4_homomorphic_log_separation = logsep  # pre-policy spelling


# -- rng233: Convolutional signal model addressed by homomorphic deconvolution..
def convmodel(x, h):
    """The convolutional signal model of homomorphic deconvolution.

    Rangayyan (2024) eq. (4.61):  y(t) = x(t) * h(t).

    The model homomorphic DEconvolution addresses.  Eq. (4.62) sends it
    to a product in the Fourier domain and eq. (4.63) then to a sum, so
    the whole method is this equation plus two transforms.  The book's
    running example is voiced speech: a glottal pulse train convolved
    with the vocal-tract response.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    y = []
    for k in range(len(xs) + len(hs) - 1):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    return RichResult(payload={
        "y": y, "n": len(y), "n_x": len(xs), "n_h": len(hs),
        "method": "Rangayyan (2024) eq. (4.61)"})


rangayyan_ch4_convolution_model = convmodel  # pre-policy spelling


# -- rng236: Definition of the complex cepstrum via inverse z-transform of complex log of Y(z)..
def ccepstrum(x):
    """Complex cepstrum: inverse transform of the complex log spectrum.

    Rangayyan (2024) eq. (4.64):
        y_hat(n) = (1 / 2 pi j) contour_integral log[Y(z)] z^(n-1) dz,

    the inverse z-transform of the complex logarithm, evaluated in
    practice on the unit circle.  The book's note under eq. (4.63) gives
    the complex log as log|X| + j angle(X).

    The phase must be UNWRAPPED before the inverse transform.  A
    principal-value phase jumps by 2 pi wherever it crosses the branch
    cut, and those jumps become spurious high-quefrency energy that
    swamps the cepstrum -- the single most common way a hand-written
    complex cepstrum comes out wrong.  ``linear_phase_removed`` records
    the linear term taken out first, which is the delay r of the z^r
    factor in eq. (4.68).

    The book notes the complex cepstrum is real when the signal is; the
    residual imaginary part is returned as a check on that.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 4:
        raise ValueError("need at least four samples")
    re, im = _fft_pair(xs)
    mags = [sqrt(a * a + b * b) for a, b in zip(re, im)]
    floor = 1e-300
    if any(v <= floor for v in mags):
        raise ValueError("the complex log needs a nonzero spectrum at every "
                         "bin; %d bins vanish" % sum(1 for v in mags
                                                     if v <= floor))
    # Unwrap over the HALF circle only, then impose odd symmetry.  For a
    # real signal the DFT phase satisfies angle(X(N-k)) = -angle(X(k)),
    # so unwrapping straight through k = 0..N-1 forces a monotone ramp
    # that is not the analytic continuation and buries the cepstrum in a
    # spurious 1/n tail.
    half = n // 2
    raw = [_atan2(b, a) for a, b in zip(re, im)]
    up = _unwrap(raw[:half + 1])
    # remove the linear phase (the z^r delay of eq. 4.68): for a real
    # signal the unwrapped phase at Nyquist is an integer multiple of pi,
    # and that integer is the delay r
    r_int = int(round(up[half] / pi)) if half else 0
    slope = r_int * pi / half if half else 0.0
    up = [v - slope * k for k, v in enumerate(up)]
    detr = list(up) + [-up[n - k] for k in range(half + 1, n)]
    logmag = [log(v) for v in mags]
    c = _ifft_real(logmag, detr)
    return RichResult(payload={
        "cepstrum": c, "log_magnitude": logmag, "phase": detr,
        "detrended_phase": detr, "linear_phase_removed": slope,
        "delay_removed": r_int, "n": n,
        "method": "Rangayyan (2024) eqs. (4.63)-(4.64)"})


rangayyan_ch4_complex_cepstrum_definition = ccepstrum  # pre-policy spelling


# -- rng238: Complex cepstra of a convolution decompose as a sum..
def ccepsum(x, h):
    """Complex cepstra of a convolution add.

    Rangayyan (2024) eq. (4.66):  y_hat(n) = x_hat(n) + h_hat(n),
    following from eq. (4.65) for the complex logs.

    This is why homomorphic deconvolution works at all: convolution in
    time becomes addition in the cepstral domain, so a linear (indeed a
    windowing) operation there separates the components.  All three
    cepstra are computed independently and the residual is reported --
    it is not exactly zero in practice because the DFT truncates a
    cepstrum the book states is of infinite duration (the note under eq.
    4.73), so a nonzero residual is expected and its size is the useful
    number.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    n = len(xs) + len(hs) - 1
    y = []
    for k in range(n):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    xp = xs + [0.0] * (n - len(xs))
    hp = hs + [0.0] * (n - len(hs))
    cy = ccepstrum(y)["cepstrum"]
    cx = ccepstrum(xp)["cepstrum"]
    ch = ccepstrum(hp)["cepstrum"]
    resid = [a - b - c for a, b, c in zip(cy, cx, ch)]
    scale = max(abs(v) for v in cy) or 1.0
    return RichResult(payload={
        "y": y, "cepstrum_y": cy, "cepstrum_x": cx, "cepstrum_h": ch,
        "residual": resid, "max_residual": max(abs(v) for v in resid),
        "relative_residual": max(abs(v) for v in resid) / scale,
        "truncation_note": "the complex cepstrum is of infinite duration "
                           "(eq. 4.73), so a finite DFT leaves a residual",
        "method": "Rangayyan (2024) eqs. (4.65)-(4.66)"})


rangayyan_ch4_complex_cepstra_sum = ccepsum  # pre-policy spelling


# -- rng239: Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum)..
def ratz(gain, r, zeros_in, zeros_out, poles_in, poles_out, z=None):
    """Rational z-transform in the pole-zero product form.

    The form whose complex logarithm the book expands at eq. (4.68):

        X(z) = A z^r prod_{k=1}^{MI} (1 - a_k z^-1)
                     prod_{k=1}^{MO} (1 - b_k z)
             / [ prod_{k=1}^{NI} (1 - c_k z^-1)
                 prod_{k=1}^{NO} (1 - d_k z) ],

    with a_k, c_k INSIDE the unit circle and b_k, d_k the reciprocals of
    factors outside it.  The four-way split is not decoration: eq.
    (4.72) gives a different closed form for each group, and it is what
    decides whether the cepstrum is causal, anticausal, or two-sided.

    Membership is checked rather than assumed -- an "inside" root passed
    with modulus above 1 would make eq. (4.72)'s series diverge, so it
    is rejected here instead of producing a silently wrong cepstrum.
    """
    ai = [complex(v) for v in zeros_in]
    bo = [complex(v) for v in zeros_out]
    ci = [complex(v) for v in poles_in]
    do = [complex(v) for v in poles_out]
    for name, group in (("zeros_in", ai), ("poles_in", ci)):
        bad = [v for v in group if abs(v) >= 1.0]
        if bad:
            raise ValueError("%s must lie inside the unit circle; |%r| = %g"
                             % (name, bad[0], abs(bad[0])))
    for name, group in (("zeros_out", bo), ("poles_out", do)):
        bad = [v for v in group if abs(v) >= 1.0]
        if bad:
            raise ValueError("%s holds the RECIPROCAL of a root outside the "
                             "unit circle, so it must itself be inside; "
                             "|%r| = %g" % (name, bad[0], abs(bad[0])))
    out = {"gain": complex(gain), "r": int(r),
           "zeros_in": ai, "zeros_out": bo,
           "poles_in": ci, "poles_out": do,
           "minimum_phase": not bo and not do,
           "maximum_phase": not ai and not ci,
           "method": "Rangayyan (2024) the rational form expanded at "
                     "eq. (4.68)"}
    if z is None:
        out["X"] = None
        return RichResult(payload=out)
    scalar = isinstance(z, (int, float, complex))
    pts = [complex(v) for v in ([z] if scalar else z)]
    vals = []
    for zv in pts:
        if zv == 0:
            raise ValueError("z = 0 is a pole of the z^-1 factors")
        num = complex(gain) * zv ** int(r)
        for ak in ai:
            num *= (1.0 - ak / zv)
        for bk in bo:
            num *= (1.0 - bk * zv)
        den = 1.0 + 0j
        for ck in ci:
            den *= (1.0 - ck / zv)
        for dk in do:
            den *= (1.0 - dk * zv)
        if den == 0:
            raise ValueError("z coincides with a pole of X")
        vals.append(num / den)
    out["X"] = vals[0] if scalar else vals
    out["z"] = pts[0] if scalar else pts
    return RichResult(payload=out)


rangayyan_ch4_rational_z_transform_form = ratz  # pre-policy spelling


# -- rng244: Closed-form complex cepstrum from poles/zeros (inside/outside unit circle)..
def ccepclosed(gain, zeros_in, zeros_out, poles_in, poles_out, nmax=32):
    """Closed-form complex cepstrum from the poles and zeros.

    Rangayyan (2024) eq. (4.72), obtained by expanding every log term of
    eq. (4.68) in its power series and inverting the z-transform:

        x_hat(n) = log|A|                                  for n = 0
                 = -sum_k a_k^n / n + sum_k c_k^n / n      for n > 0
                 = sum_k b_k^-n / n - sum_k d_k^-n / n     for n < 0

    with a_k, c_k the zeros and poles inside the unit circle and b_k,
    d_k the reciprocals of those outside it.

    The book's three properties follow directly and are returned rather
    than left implicit: a minimum-phase signal (nothing outside) has a
    CAUSAL cepstrum, x_hat(n) = 0 for n < 0; a maximum-phase signal
    (nothing inside) has an ANTICAUSAL one; and the cepstrum is of
    infinite duration even for a finite-duration signal, so ``nmax`` is
    a truncation, not the whole thing.
    """
    ai = [complex(v) for v in zeros_in]
    bo = [complex(v) for v in zeros_out]
    ci = [complex(v) for v in poles_in]
    do = [complex(v) for v in poles_out]
    k = int(nmax)
    if k < 1:
        raise ValueError("nmax must be positive")
    g = abs(complex(gain))
    if g <= 0:
        raise ValueError("the gain must be nonzero")
    pos, neg = [], []
    for n in range(1, k + 1):
        pos.append(sum(-(a ** n) / n for a in ai)
                   + sum((c ** n) / n for c in ci))
        neg.append(sum((b ** n) / n for b in bo)
                   - sum((d ** n) / n for d in do))
    quef = list(range(-k, k + 1))
    vals = list(reversed(neg)) + [complex(log(g))] + pos
    return RichResult(payload={
        "cepstrum": vals, "quefrency": quef, "c0": log(g),
        "positive": pos, "negative": list(reversed(neg)),
        "causal": not bo and not do,
        "anticausal": not ai and not ci,
        "infinite_duration": True, "nmax": k,
        "method": "Rangayyan (2024) eq. (4.72)"})


rangayyan_ch4_complex_cepstrum_closed_form = ccepclosed  # pre-policy spelling


# -- rng245: Decay bound for the complex cepstrum: at least as fast as 1/n..
def ccepdecay(zeros_in, zeros_out, poles_in, poles_out, nmax=32,
              constant=None):
    """Decay bound on the complex cepstrum.

    Rangayyan (2024) eq. (4.73):
        |x_hat(n)| < K |alpha^n / n|,   -inf < n < inf,
    where alpha = max(|a_k|, |b_k|, |c_k|, |d_k|) and K is a constant.

    The book's reading: the complex cepstrum decays AT LEAST as fast as
    1/n.  The geometric factor alpha^n is what makes liftering work --
    a signal whose roots all sit well inside the unit circle has a
    cepstrum concentrated at low quefrency -- and a root close to the
    circle sends alpha towards 1, leaving only the 1/n decay and a
    cepstrum that a short lifter will truncate badly.

    ``K`` defaults to the total number of roots, which is the constant
    the term-by-term bound on eq. (4.72) gives: each of the K sums has
    at most one term of size alpha^n / n.
    """
    roots = ([complex(v) for v in zeros_in] + [complex(v) for v in zeros_out]
             + [complex(v) for v in poles_in]
             + [complex(v) for v in poles_out])
    if not roots:
        raise ValueError("need at least one root")
    alpha = max(abs(v) for v in roots)
    if alpha <= 0:
        raise ValueError("all roots are at the origin; the bound is vacuous")
    k = int(nmax)
    if k < 1:
        raise ValueError("nmax must be positive")
    kk = float(len(roots)) if constant is None else float(constant)
    bound = [kk * (alpha ** n) / n for n in range(1, k + 1)]
    return RichResult(payload={
        "alpha": alpha, "K": kk, "bound": bound,
        "quefrency": list(range(1, k + 1)),
        "decays_at_least_as_one_over_n": True,
        "near_unit_circle": alpha > 0.95,
        "method": "Rangayyan (2024) eq. (4.73)"})


rangayyan_ch4_complex_cepstrum_decay_bound = ccepdecay  # pre-policy spelling


# -- rng251: Power-series expansion of the log echo term (a < 1)..
def echoseries(a, n0, terms=10, omega=None):
    """The echo term of the log spectrum, and the impulse train it gives.

    Rangayyan (2024) eqs. (4.74)-(4.80).  A wavelet plus one echo,
    y(n) = h(n) + a h(n - n0), has

        Y_hat(w) = H_hat(w) + a exp(-j w n0)
                   - (a^2/2) exp(-2 j w n0)
                   + (a^3/3) exp(-3 j w n0) - ...            (4.79)
        y_hat(n) = h_hat(n) + a d(n - n0) - (a^2/2) d(n - 2 n0)
                   + (a^3/3) d(n - 3 n0) - ...               (4.80)

    so the complex cepstrum of an echoed signal is that of the basic
    wavelet plus an impulse train at the echo delay and its multiples,
    with alternating signs and amplitudes a^k / k.

    The expansion needs a < 1 -- the book says so at eq. (4.79) -- and
    that is enforced: at a >= 1 the series diverges and the impulse
    train is not a valid reading of the cepstrum.
    """
    av = float(a)
    if not abs(av) < 1.0:
        raise ValueError("eq. (4.79) needs |a| < 1; got %g" % av)
    d = int(n0)
    if d < 1:
        raise ValueError("the echo delay must be at least one sample")
    k = int(terms)
    if k < 1:
        raise ValueError("terms must be positive")
    amps = [((-1) ** (i + 1)) * (av ** i) / i for i in range(1, k + 1)]
    lags = [i * d for i in range(1, k + 1)]
    out = {"amplitudes": amps, "quefrencies": lags, "a": av, "n0": d,
           "terms": k, "first_peak": d,
           "method": "Rangayyan (2024) eqs. (4.79)-(4.80)"}
    if omega is not None:
        scalar = isinstance(omega, (int, float))
        ws = [float(omega)] if scalar else [float(v) for v in omega]
        vals = []
        for w in ws:
            acc = 0j
            for amp, lag in zip(amps, lags):
                acc += amp * complex(cos(-w * lag), sin(-w * lag))
            vals.append(acc)
        exact = []
        for w in ws:
            u = 1.0 + av * complex(cos(-w * d), sin(-w * d))
            exact.append(complex(log(abs(u)), _atan2(u.imag, u.real)))
        out["series"] = vals[0] if scalar else vals
        out["exact"] = exact[0] if scalar else exact
        out["max_error"] = max(abs(u - v) for u, v in zip(vals, exact))
    return RichResult(payload=out)


rangayyan_ch4_log_echo_power_series = echoseries  # pre-policy spelling


# -- rng253: Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2..
def pcepstrum(x, square=True):
    """Power cepstrum.

    Rangayyan (2024) eq. (4.81):
        y_hat_p(n) = { (1 / 2 pi j) contour_integral
                       log|Y(z)|^2 z^(n-1) dz }^2.

    The book notes the final squaring is omitted in some definitions,
    and that this matters: with the square, eq. (4.82) holds only when
    the cross-product term is neglected, which is exact just when the
    two components occupy non-overlapping quefrency ranges; WITHOUT the
    square, no cross-term arises and eq. (4.82) is exact.  ``square``
    selects which definition, defaulting to the book's eq. (4.81).

    The power cepstrum discards the phase, so it cannot reconstruct the
    signal.  It is used to detect echoes and estimate their arrival
    times, which is exactly what survives the loss of phase.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    re, im = _fft_pair(xs)
    p2 = [a * a + b * b for a, b in zip(re, im)]
    floor = 1e-300
    zeros = sum(1 for v in p2 if v <= floor)
    logp = [log(v if v > floor else floor) for v in p2]
    base = _ifft_real(logp, [0.0] * n)
    vals = [v * v for v in base] if square else list(base)
    return RichResult(payload={
        "cepstrum": vals, "unsquared": base, "log_power": logp, "n": n,
        "squared": bool(square), "zero_bins": zeros,
        "retains_phase": False,
        "additivity_exact": not square,
        "method": "Rangayyan (2024) eq. (4.81)"})


rangayyan_ch4_power_cepstrum_definition = pcepstrum  # pre-policy spelling


# -- rng254: Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected)..
def pcepsum(x, h, square=False):
    """Power cepstra of a convolution add.

    Rangayyan (2024) eq. (4.82):  y_hat_p(n) = x_hat_p(n) + h_hat_p(n).

    It follows from |Y(z)|^2 = |X(z)|^2 |H(z)|^2, so log|Y|^2 splits and
    the inverse transform is additive.  The book is explicit that the
    cross-product term was neglected in getting there, and that the
    identity is exact only when the components occupy non-overlapping
    quefrency ranges -- or when the final squaring of eq. (4.81) is
    omitted, in which case no cross-term arises at all.

    ``square`` therefore defaults to False here, the case in which the
    equation actually holds; passing True reproduces eq. (4.81)'s
    definition and reports how large the neglected cross-term is on the
    caller's own data.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    n = len(xs) + len(hs) - 1
    y = []
    for k in range(n):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    xp = xs + [0.0] * (n - len(xs))
    hp = hs + [0.0] * (n - len(hs))
    cy = pcepstrum(y, square=square)["cepstrum"]
    cx = pcepstrum(xp, square=square)["cepstrum"]
    ch = pcepstrum(hp, square=square)["cepstrum"]
    resid = [a - b - c for a, b, c in zip(cy, cx, ch)]
    scale = max(abs(v) for v in cy) or 1.0
    return RichResult(payload={
        "y": y, "cepstrum_y": cy, "cepstrum_x": cx, "cepstrum_h": ch,
        "residual": resid, "max_residual": max(abs(v) for v in resid),
        "relative_residual": max(abs(v) for v in resid) / scale,
        "squared": bool(square), "exact": not square,
        "method": "Rangayyan (2024) eq. (4.82)"})


rangayyan_ch4_power_cepstrum_sum = pcepsum  # pre-policy spelling


# -- rng255: Relation between power cepstrum and complex cepstrum..
def pceprel(x):
    """Power cepstrum from the complex cepstrum.

    Rangayyan (2024) eq. (4.83):
        y_hat_p(n) = [ y_hat(n) + y_hat(-n) ]^2.

    The power cepstrum is the squared EVEN PART of the complex cepstrum,
    doubled -- which is exactly why it loses the phase: the odd part,
    where the phase information lives, is annihilated by the folding.

    Both routes are computed: eq. (4.83) from the complex cepstrum, and
    eq. (4.81) directly from the log power spectrum.  Their agreement is
    the content of the relation; the residual comes from the finite DFT
    truncating an infinite-duration cepstrum, and is reported rather
    than hidden.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 4:
        raise ValueError("need at least four samples")
    c = ccepstrum(xs)["cepstrum"]
    folded = [(c[i] + c[(-i) % n]) ** 2 for i in range(n)]
    direct = pcepstrum(xs, square=True)["cepstrum"]
    resid = [a - b for a, b in zip(folded, direct)]
    scale = max(abs(v) for v in direct) or 1.0
    return RichResult(payload={
        "from_complex": folded, "direct": direct, "residual": resid,
        "max_residual": max(abs(v) for v in resid),
        "relative_residual": max(abs(v) for v in resid) / scale,
        "phase_lost": True, "n": n,
        "method": "Rangayyan (2024) eq. (4.83)"})


rangayyan_ch4_power_cepstrum_relation = pceprel  # pre-policy spelling


_CHEATSHEET = [
    'rgar2cep: Cepstral coefficients from AR coefficients.',
    'rgccep: complex cepstrum with unwrapping diagnostics',
    'rgcepsp: Cepstral pitch detection.',
    'rgcepst: real cepstrum',
    'rghomdc: homomorphic deconvolution, Section 4.7.2',
    'rghomo: multiplicative homomorphic filter, Section 4.7.1',
    'rghompr: low-time / high-time cepstral prediction, Section 4.7.3',
    'rglift: cepstral liftering, Rangayyan Section 4.7.3',
    'rgmfcc: mel-frequency cepstral coefficients (Davis-Mermelstein 1980)',
    'rgminph: minimum-phase correspondent, Section 4.7.2',
    'rgvocal: vocal-tract response by low-time liftering, Section 4.7.3',
    'rng230: multiplicative model, Rangayyan eq. (4.58)',
    'rng231: log of a product is a sum, Rangayyan eq. (4.59)',
    'rng233: convolutional model, Rangayyan eq. (4.61)',
    'rng236: complex cepstrum, Rangayyan eqs. (4.63)-(4.64)',
    'rng238: complex cepstra of a convolution add, eqs. (4.65)-(4.66)',
    'rng239: rational z-transform in pole-zero form, before eq. (4.68)',
    'rng244: closed-form complex cepstrum, Rangayyan eq. (4.72)',
    'rng245: complex-cepstrum decay bound, Rangayyan eq. (4.73)',
    'rng251: echo term of the log spectrum, eqs. (4.79)-(4.80)',
    'rng253: power cepstrum, Rangayyan eq. (4.81)',
    'rng254: power cepstra add, Rangayyan eq. (4.82)',
    'rng255: power cepstrum from the complex cepstrum, eq. (4.83)',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
