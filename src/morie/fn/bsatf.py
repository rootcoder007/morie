# morie.fn -- bsatf (rootcoder007/morie)
"""Time-frequency and multiresolution analysis: STFT, Wigner-Ville and Cohen's class, wavelets, EMD, Hilbert-Huang, VMD.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 50
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2, cos, exp, fsum, log, pi, sin, sqrt
import cmath
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import CubicSpline
from ._signal_core import hilbert

__all__ = [
    'hilbert_huang_spectrum',
    'cdemod',
    'rangayyan_amplitude_demod',
    'biordwt',
    'rangayyan_biorthogonal_wvlt',
    'expkertfd',
    'rangayyan_choi_williams',
    'cprwt',
    'rangayyan_cpr_analysis',
    'cwt',
    'rangayyan_cwt',
    'gtfd',
    'rangayyan_cohen_class',
    'orthfilt',
    'rangayyan_daubechies',
    'atomtfd',
    'rangayyan_decomp_tfd',
    'dwt',
    'rangayyan_dwt',
    'emdens',
    'rangayyan_eemd',
    'sift',
    'rangayyan_emd',
    'imf',
    'rangayyan_emd_imf',
    'twaemd',
    'rangayyan_emd_twa',
    'vfemd',
    'rangayyan_emd_vf_detect',
    'rangayyan_emg_rms',
    'wtentropy',
    'rangayyan_wavelet_entropy',
    'rangayyan_envelope',
    'rangayyan_envelogram',
    'dwt2tap',
    'rangayyan_haar_wavelet',
    'emdspec',
    'rangayyan_hht_spectrum',
    'hrvtv',
    'rangayyan_hrv_time_varying',
    'istft',
    'rangayyan_istft',
    'mra',
    'rangayyan_mra',
    'pcgenvavg',
    'rangayyan_pcg_envelope_avg',
    'ppgwtden',
    'rangayyan_ppg_wavelet',
    'scalogram',
    'rangayyan_scalogram',
    'seizwt',
    'rangayyan_seizure_wavelet',
    'stftparam',
    'rangayyan_stft_params',
    'spectrogram',
    'rangayyan_stft_spectrogram',
    'swt',
    'rangayyan_swt',
    'swtden',
    'rangayyan_swt_denoise',
    'vmodes',
    'rangayyan_vmd',
    'cwtridge',
    'rangayyan_wavelet_struct',
    'wtxcor',
    'rangayyan_wavelet_corr',
    'wvdist',
    'rangayyan_wigner_ville',
    'wtenergy',
    'rangayyan_wavelet_energy',
    'wtmoment',
    'rangayyan_wavelet_moments',
    'wpt',
    'rangayyan_wavelet_packet',
    'wtthresh',
    'rangayyan_wavelet_threshold',
    'wtvar',
    'rangayyan_wavelet_variance',
    'echoimp',
    'rangayyan_ch4_signal_with_echo_input',
    'echosig',
    'rangayyan_ch4_signal_with_echo_output',
    'echoz',
    'rangayyan_ch4_z_transform_signal_echo',
    'echospec',
    'rangayyan_ch4_fourier_signal_echo',
    'echologsp',
    'rangayyan_ch4_log_signal_echo',
    'echocep',
    'rangayyan_ch4_complex_cepstrum_signal_with_echo',
    'echopsd',
    'rangayyan_ch4_power_spectrum_signal_echo',
    'echologpsd',
    'rangayyan_ch4_log_power_spectrum_signal_echo',
    'wigner_ville',
    'rangayyancwt',
    'rangayyandwt',
    'rangayyaneemd',
    'rangayyanemd',
    'rangayyanistft',
    'rangayyanmra',
    'rangayyanswt',
    'rangayyanvmd',
]

# ---------------------------------------------------------------------------
# Shared time-frequency kernel used by the bsatf shelf.
#
# Everything below is plain Python.  The DFT is an O(N^2) direct evaluation
# of the defining sum -- NOT an FFT.  That is a deliberate choice: the
# signals in this shelf are short (tens to a few hundred samples), and an
# honest quadratic DFT is easier to verify than a radix-2 routine that
# silently misbehaves on non-power-of-2 lengths.
# ---------------------------------------------------------------------------


def _tfneed(x, name="x", minlen=2):
    """Coerce to a list of floats and reject anything too short to transform."""
    v = aslist(x)
    if len(v) < minlen:
        raise ValueError(f"{name} must have at least {minlen} samples, got {len(v)}")
    for t in v:
        if t != t or t in (float("inf"), float("-inf")):
            raise ValueError(f"{name} contains a non-finite sample")
    return v


def _tfdft(x):
    """Direct DFT.  X[k] = sum_n x[n] exp(-j 2 pi k n / N).  O(N^2)."""
    n = len(x)
    out = []
    for k in range(n):
        w = -2.0 * pi * k / n
        # (a + jb)(cos + j sin) = (a cos - b sin) + j(a sin + b cos)
        re = fsum((v.real if isinstance(v, complex) else v) * cos(w * i) -
                  (v.imag if isinstance(v, complex) else 0.0) * sin(w * i)
                  for i, v in enumerate(x))
        im = fsum((v.real if isinstance(v, complex) else v) * sin(w * i) +
                  (v.imag if isinstance(v, complex) else 0.0) * cos(w * i)
                  for i, v in enumerate(x))
        out.append(complex(re, im))
    return out


def _tfidft(X):
    """Direct inverse DFT.  x[n] = (1/N) sum_k X[k] exp(+j 2 pi k n / N)."""
    n = len(X)
    out = []
    for i in range(n):
        w = 2.0 * pi * i / n
        re = fsum(X[k].real * cos(w * k) - X[k].imag * sin(w * k) for k in range(n))
        im = fsum(X[k].imag * cos(w * k) + X[k].real * sin(w * k) for k in range(n))
        out.append(complex(re / n, im / n))
    return out


def _tfwin(name, m):
    """Analysis windows.  'rect' is Rangayyan eq (8.7); the tapers are the
    usual raised-cosine forms used for the spectrograms of Section 8.7."""
    if m < 1:
        raise ValueError("window length must be >= 1")
    name = str(name).lower()
    if name in ("rect", "rectangular", "boxcar", "none"):
        return [1.0] * m
    if m == 1:
        return [1.0]
    if name in ("hann", "hanning"):
        return [0.5 - 0.5 * cos(2.0 * pi * i / (m - 1)) for i in range(m)]
    if name == "hamming":
        return [0.54 - 0.46 * cos(2.0 * pi * i / (m - 1)) for i in range(m)]
    if name in ("bartlett", "triang"):
        return [1.0 - abs((i - (m - 1) / 2.0) / ((m - 1) / 2.0)) for i in range(m)]
    raise ValueError(f"unknown window {name!r}; use rect, hann, hamming or bartlett")


def _tfanalytic(x):
    """Analytic signal x + j H{x} built by zeroing the negative-frequency half
    of the DFT (Rangayyan Sec 5.5.3).  Returns a list of complex."""
    n = len(x)
    X = _tfdft([float(v) for v in x])
    h = [0.0] * n
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        for k in range(1, n // 2):
            h[k] = 2.0
    else:
        for k in range(1, (n + 1) // 2):
            h[k] = 2.0
    return _tfidft([X[k] * h[k] for k in range(n)])


# --- orthogonal wavelet filter banks -------------------------------------
# Daubechies scaling-filter taps, normalised so that sum(h) = sqrt(2) and
# sum(h[n] h[n+2m]) = delta(m).  Rangayyan cites Daubechies (ref [74],
# IEEE Trans. Inf. Theory 36(5):961-1005, 1990) but does NOT tabulate the
# coefficients; the tabulated values come from
#   Daubechies, I. (1992). Ten Lectures on Wavelets. SIAM, Table 6.1.
# The orthonormality identities above are asserted at import-free runtime by
# _tffilters, so a mistyped tap cannot pass silently.
_DBTAPS = {
    1: [0.7071067811865476, 0.7071067811865476],
    2: [0.48296291314469025, 0.836516303737469, 0.22414386804185735,
        -0.12940952255092145],
    3: [0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
        -0.13501102001039084, -0.08544127388224149, 0.035226291882100656],
    4: [0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
        -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
        0.032883011666982945, -0.010597401784997278],
    5: [0.160102397974125, 0.6038292697974729, 0.7243085284377726,
        0.13842814590110342, -0.24229488706619015, -0.03224486958502952,
        0.07757149384006515, -0.006241490213011705, -0.012580751999015526,
        0.003335725285001549],
    6: [0.11154074335008017, 0.4946238903983854, 0.7511339080215775,
        0.3152503517092432, -0.22626469396516913, -0.12976686756709563,
        0.09750160558707936, 0.02752286553001629, -0.031582039318031156,
        0.0005538422009938016, 0.004777257511010651, -0.00107730108499558],
    7: [0.07785205408506236, 0.39653931948230575, 0.7291320908465551,
        0.4697822874053586, -0.14390600392910627, -0.22403618499416572,
        0.07130921926705004, 0.0806126091510659, -0.03802993693503463,
        -0.01657454163101562, 0.012550998556013784, 0.00042957797300470274,
        -0.0018016407039998328, 0.0003537138000010399],
    8: [0.05441584224308161, 0.3128715909144659, 0.6756307362980128,
        0.5853546836548691, -0.015829105256023893, -0.2840155429624281,
        0.00047248457399797254, 0.128747426620186, -0.017369301002022108,
        -0.04408825393106472, 0.013981027917015516, 0.008746094047015655,
        -0.004870352993451574, -0.000391740373376471, 0.0006754494059985568,
        -0.00011747678400228192],
    9: [0.03807794736316728, 0.24383467463766728, 0.6048231236767786,
        0.6572880780366389, 0.13319738582208895, -0.29327378327258685,
        -0.09684078322087904, 0.14854074933476008, 0.030725681478322865,
        -0.06763282905952399, 0.000250947114834164, 0.022361662123515244,
        -0.004723204757894831, -0.004281503681904723, 0.0018476468829611268,
        0.00023038576399541288, -0.0002519631889981789, 3.934732031627159e-05],
    10: [0.026670057900950818, 0.18817680007762133, 0.5272011889309198,
         0.6884590394525921, 0.2811723436604265, -0.24984642432648865,
         -0.19594627437659665, 0.12736934033574265, 0.09305736460380659,
         -0.07139414716586077, -0.02945753682194567, 0.03321267405893324,
         0.0036065535669883944, -0.010733175482979604, 0.0013953517469940798,
         0.00199240529499085, -0.0006858566950046825, -0.0001164668549943862,
         9.358867000108985e-05, -1.326420300235487e-05],
}


def _tfdbname(wavelet):
    """Map a wavelet spelling onto a Daubechies order.  'haar' == 'db1'."""
    w = str(wavelet).strip().lower().replace("-", "").replace("_", "")
    if w in ("haar", "db1", "d2"):
        return 1
    if w.startswith("db") and w[2:].isdigit():
        k = int(w[2:])
        if k in _DBTAPS:
            return k
    if w.startswith("d") and w[1:].isdigit() and int(w[1:]) % 2 == 0:
        k = int(w[1:]) // 2
        if k in _DBTAPS:
            return k
    raise ValueError(
        f"unknown wavelet {wavelet!r}; use 'haar' or 'db1'..'db10'"
    )


def _tffilters(wavelet):
    """Return (dec_lo, dec_hi, rec_lo, rec_hi) for an orthogonal wavelet.

    The high-pass mirror is g[n] = (-1)^n h[L-1-n].  The orthonormality of
    the tabulated taps is checked here so that a corrupted table raises
    rather than quietly producing a transform that does not invert.
    """
    h = list(_DBTAPS[_tfdbname(wavelet)])
    L = len(h)
    if abs(fsum(v * v for v in h) - 1.0) > 1e-9:
        raise ValueError(f"scaling filter for {wavelet!r} is not unit-norm")
    for m in range(1, L // 2):
        if abs(fsum(h[n] * h[n + 2 * m] for n in range(L - 2 * m))) > 1e-9:
            raise ValueError(f"scaling filter for {wavelet!r} is not orthogonal")
    g = [((-1) ** n) * h[L - 1 - n] for n in range(L)]
    return h, g, h, g


def _tfdwtstep(a, h, g):
    """One periodic analysis level.  a_k = sum_n h[n] a[(2k+n) mod N].

    Periodic (wrap-around) extension is used because it is the only
    extension for which the decimated filter bank is an exact orthonormal
    basis of R^N, i.e. the only one that round-trips to machine precision.
    """
    n = len(a)
    if n % 2:
        a = list(a) + [a[-1]]
        n += 1
    half = n // 2
    lo, hi = [], []
    for k in range(half):
        lo.append(fsum(h[j] * a[(2 * k + j) % n] for j in range(len(h))))
        hi.append(fsum(g[j] * a[(2 * k + j) % n] for j in range(len(g))))
    return lo, hi


def _tfidwtstep(lo, hi, h, g):
    """Adjoint of _tfdwtstep: x[m] = sum_k lo[k] h[(m-2k) mod N] + hi[k] g[...]."""
    half = len(lo)
    n = 2 * half
    L = len(h)
    out = [0.0] * n
    for k in range(half):
        for j in range(L):
            m = (2 * k + j) % n
            out[m] += lo[k] * h[j] + hi[k] * g[j]
    return out


def _tfdwt(x, wavelet, levels):
    """Multilevel periodic DWT.  Returns (cA_J, [cD_J, ..., cD_1], lengths)."""
    h, g, _, _ = _tffilters(wavelet)
    a = list(x)
    if levels < 1:
        raise ValueError("levels must be >= 1")
    maxlev = 0
    m = len(a)
    while m >= len(h) and m >= 2:
        maxlev += 1
        m = (m + 1) // 2
    if levels > maxlev:
        raise ValueError(
            f"levels={levels} exceeds the maximum {maxlev} for a signal of "
            f"length {len(x)} with filter length {len(h)}"
        )
    details, lengths = [], []
    for _ in range(levels):
        lengths.append(len(a))
        a, d = _tfdwtstep(a, h, g)
        details.append(d)
    details.reverse()
    lengths.reverse()
    return a, details, lengths


def _tfidwt(a, details, lengths, wavelet):
    """Invert _tfdwt.  details/lengths are coarse-to-fine as returned above."""
    h, g, _, _ = _tffilters(wavelet)
    cur = list(a)
    for d, n in zip(details, lengths):
        cur = _tfidwtstep(cur, d, h, g)
        cur = cur[:n]
    return cur


def _tfswt(x, wavelet, levels):
    """Undecimated (stationary / a-trous) wavelet transform.

    No decimation; instead the filters are upsampled by 2^level, so every
    output has the same length as the input and the transform is shift
    invariant.  Nason, G. P., & Silverman, B. W. (1995), "The stationary
    wavelet transform and some statistical applications", Lecture Notes in
    Statistics 103:281-299.  Rangayyan does not define the SWT.
    """
    h, g, _, _ = _tffilters(wavelet)
    n = len(x)
    a = list(x)
    details = []
    approxes = []
    for lev in range(levels):
        step = 2 ** lev
        lo = [fsum(h[j] * a[(i + j * step) % n] for j in range(len(h))) for i in range(n)]
        hi = [fsum(g[j] * a[(i + j * step) % n] for j in range(len(g))) for i in range(n)]
        details.append(hi)
        approxes.append(lo)
        a = lo
    return a, details, approxes


# --- spline interpolation for EMD envelopes -------------------------------
def _tfspline(xs, ys, xq):
    """Natural cubic spline through (xs, ys), evaluated at every xq.

    Rangayyan's EMD recipe (Sec 9.4, step 2) says explicitly that "the cubic
    spline function is recommended for interpolation" of the extrema
    envelopes, so the envelope quality here is not an implementation detail.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two knots for a spline")
    if n == 2:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return [ys[0] + s * (t - xs[0]) for t in xq]
    hh = [xs[i + 1] - xs[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3.0 * ((ys[i + 1] - ys[i]) / hh[i] - (ys[i] - ys[i - 1]) / hh[i - 1])
    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - hh[i - 1] * mu[i - 1]
        mu[i] = hh[i] / l[i]
        z[i] = (alpha[i] - hh[i - 1] * z[i - 1]) / l[i]
    l[n - 1] = 1.0
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / hh[j] - hh[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * hh[j])
    out = []
    for t in xq:
        j = 0
        if t <= xs[0]:
            j = 0
        elif t >= xs[-1]:
            j = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if xs[mid] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            j = lo
        dt = t - xs[j]
        out.append(ys[j] + b[j] * dt + c[j] * dt * dt + d[j] * dt * dt * dt)
    return out


def _tfextrema(x):
    """Indices of interior local maxima and minima (plateaux use the centre)."""
    mx, mn = [], []
    n = len(x)
    i = 1
    while i < n - 1:
        if x[i] > x[i - 1]:
            j = i
            while j < n - 1 and x[j + 1] == x[i]:
                j += 1
            # j may have run to the last sample: a plateau that reaches the
            # end of the record is not an interior extremum.
            if j < n - 1 and x[j] > x[j + 1]:
                mx.append((i + j) // 2)
            i = j + 1
            continue
        if x[i] < x[i - 1]:
            j = i
            while j < n - 1 and x[j + 1] == x[i]:
                j += 1
            if j < n - 1 and x[j] < x[j + 1]:
                mn.append((i + j) // 2)
            i = j + 1
            continue
        i += 1
    return mx, mn


def _tfzerox(x):
    """Number of zero crossings."""
    c = 0
    for i in range(1, len(x)):
        if (x[i - 1] < 0.0 <= x[i]) or (x[i - 1] > 0.0 >= x[i]):
            c += 1
    return c


def _tfsift(x, maxiter=50, tol=0.05):
    """Sift one IMF out of x -- Rangayyan Sec 9.4, algorithm steps 1-5.

    Returns (imf, iterations, converged).  The stopping rule is the
    normalised sum of squared differences between successive proto-modes
    (Huang's SD criterion); the book states the IMF admissibility
    conditions (equal extrema and zero-crossing counts to within one, zero
    mean envelope) but leaves the numerical stopping rule to the cited
    references, so the SD criterion is taken from
      Huang, N. E., et al. (1998), Proc. R. Soc. Lond. A 454:903-995, eq (5.5).
    """
    h = list(x)
    n = len(h)
    t = list(range(n))
    for it in range(1, maxiter + 1):
        mx, mn = _tfextrema(h)
        if len(mx) < 2 or len(mn) < 2:
            return h, it, True
        upx = [0] + mx + [n - 1]
        upy = [h[0]] + [h[i] for i in mx] + [h[n - 1]]
        lox = [0] + mn + [n - 1]
        loy = [h[0]] + [h[i] for i in mn] + [h[n - 1]]
        up = _tfspline(upx, upy, t)
        lo = _tfspline(lox, loy, t)
        mean = [(up[i] + lo[i]) / 2.0 for i in range(n)]
        newh = [h[i] - mean[i] for i in range(n)]
        den = fsum(v * v for v in h)
        sd = fsum((newh[i] - h[i]) ** 2 for i in range(n)) / den if den > 0 else 0.0
        h = newh
        if sd < tol:
            return h, it, True
    return h, maxiter, False


def _tfemd(x, maxmodes=10, tol=0.05):
    """Full EMD.  Returns (imfs, residual) -- Rangayyan Sec 9.4, step 6."""
    res = list(x)
    imfs = []
    for _ in range(maxmodes):
        mx, mn = _tfextrema(res)
        if len(mx) + len(mn) < 3:
            break
        imf, _it, _ok = _tfsift(res, tol=tol)
        imfs.append(imf)
        res = [res[i] - imf[i] for i in range(len(res))]
    return imfs, res


# --- mother wavelets for the CWT (Rangayyan eqs 8.115, 8.116) -------------
def _tfmother(name, t, w0=5.0):
    """Value of the mother wavelet at t.  Returns complex."""
    name = str(name).strip().lower()
    if name in ("mexh", "mexicanhat", "sombrero", "ricker"):
        # Rangayyan eq (8.115): psi(t) = (1 - t^2) exp(-0.5 t^2)
        return complex((1.0 - t * t) * exp(-0.5 * t * t), 0.0)
    if name == "morlet":
        # Rangayyan eq (8.116):
        #   psi(t) = pi^-1/4 [exp(j w0 t) - exp(-0.5 w0^2)] exp(-0.5 t^2)
        # The subtracted constant is the admissibility correction (eq 8.109).
        env = exp(-0.5 * t * t) / (pi ** 0.25)
        return (cmath.exp(1j * w0 * t) - exp(-0.5 * w0 * w0)) * env
    if name in ("haar", "db1"):
        if 0.0 <= t < 0.5:
            return complex(1.0, 0.0)
        if 0.5 <= t < 1.0:
            return complex(-1.0, 0.0)
        return complex(0.0, 0.0)
    raise ValueError(f"unknown wavelet {name!r}; use 'morlet', 'mexh' or 'haar'")


def _tfsupport(name):
    """Half-support of the mother wavelet in units of t (truncation radius)."""
    return {"morlet": 4.0, "mexh": 5.0, "mexicanhat": 5.0, "sombrero": 5.0,
            "ricker": 5.0, "haar": 1.0, "db1": 1.0}[str(name).strip().lower()]


def _tfcwt(x, scales, wavelet="morlet", w0=5.0):
    """Discrete CWT, Rangayyan eq (8.107) with the sampling interval set to 1.

        X(tau, s) = (1/sqrt(s)) sum_t x(t) conj(psi((t - tau)/s))

    The wavelet is truncated at its numerical support, so the cost is
    O(N * sum(s)) rather than O(N^2 * S).
    """
    n = len(x)
    rad = _tfsupport(wavelet)
    out = []
    for s in scales:
        if s <= 0:
            raise ValueError("scales must be positive")
        half = int(rad * s) + 1
        row = []
        for tau in range(n):
            acc = 0j
            for i in range(max(0, tau - half), min(n, tau + half + 1)):
                acc += x[i] * _tfmother(wavelet, (i - tau) / s, w0).conjugate()
            row.append(acc / sqrt(s))
        out.append(row)
    return out


def _tfwvd(x, fs, nfreq=None):
    """Discrete Wigner-Ville distribution, Rangayyan eq (8.123).

        WVD(t, w) = int x(t + tau/2) x*(t - tau/2) exp(-j w tau) dtau

    Discretised on the analytic signal (which halves the required lag
    sampling rate and removes the negative-frequency cross terms), giving
    the usual Claasen-Mecklenbrauker form

        W[n, k] = 2 sum_tau z[n+tau] z*[n-tau] exp(-j 4 pi tau k / N).

    Returns (matrix[time][freq], freqs).  Cost is O(N^2) by construction --
    the WVD genuinely is a quadratic-cost distribution.
    """
    n = len(x)
    nf = int(nfreq or n)
    z = _tfanalytic(x)
    freqs = [k * fs / (2.0 * nf) for k in range(nf)]
    tfd = []
    for i in range(n):
        m = min(i, n - 1 - i)
        ker = [z[i + tau] * z[i - tau].conjugate() for tau in range(-m, m + 1)]
        row = []
        for k in range(nf):
            w = -2.0 * pi * k / nf
            acc = 0j
            for j, tau in enumerate(range(-m, m + 1)):
                acc += ker[j] * cmath.exp(1j * w * tau)
            row.append(2.0 * acc.real)
        tfd.append(row)
    return tfd, freqs


def _tfsmooth2d(tfd, tlen, flen):
    """Separable Gaussian smoothing of a TFD -- Rangayyan eqs (8.125)-(8.127),
    with the kernel factorised as phi(u, Omega) = g(u) H(Omega) (eq 8.126)."""
    nt = len(tfd)
    nf = len(tfd[0])

    def gauss(L):
        if L <= 1:
            return [1.0]
        sig = L / 6.0
        w = [exp(-0.5 * ((i - (L - 1) / 2.0) / sig) ** 2) for i in range(L)]
        s = fsum(w)
        return [v / s for v in w]

    g = gauss(tlen)
    H = gauss(flen)
    tmp = [[0.0] * nf for _ in range(nt)]
    ht = (len(g) - 1) // 2
    for i in range(nt):
        for k in range(nf):
            tmp[i][k] = fsum(g[j] * tfd[min(nt - 1, max(0, i + j - ht))][k]
                             for j in range(len(g)))
    out = [[0.0] * nf for _ in range(nt)]
    hf = (len(H) - 1) // 2
    for i in range(nt):
        for k in range(nf):
            out[i][k] = fsum(H[j] * tmp[i][min(nf - 1, max(0, k + j - hf))]
                             for j in range(len(H)))
    return out


def _tfenergy(v):
    return fsum((abs(t) ** 2) for t in v)



# -- hhtrf: Hilbert-Huang Transform (EMD + instantaneous frequency/amplitude).
def _sift_imf(x: np.ndarray, max_iter: int = 300, sd_tol: float = 0.05) -> np.ndarray:
    """Extract one IMF via sifting."""
    h = x.copy()
    for _ in range(max_iter):
        t = np.arange(len(h))
        max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
        min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
        lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
        mean_env = (upper + lower) / 2
        prev = h.copy()
        h = h - mean_env
        sd = np.sum((prev - h) ** 2) / (np.sum(prev**2) + 1e-12)
        if sd < sd_tol:
            break
    return h


def hilbert_huang_spectrum(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    max_imfs: int = 10,
    n_freq_bins: int = 256,
) -> DescriptiveResult:
    r"""Hilbert-Huang Transform with full Hilbert spectrum.

    Performs Empirical Mode Decomposition to extract IMFs, then applies
    the Hilbert transform to each IMF to obtain instantaneous
    frequency and amplitude.  Constructs a Hilbert spectrum
    :math:`H(f, t)` and marginal spectrum :math:`h(f)`:

    .. math::

        H(\\omega, t) = \\sum_{i=1}^{N} a_i(t) \\,
        \\delta(\\omega - \\omega_i(t))

    .. math::

        h(\\omega) = \\int_0^T H(\\omega, t) \\, dt

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz (default 1.0).
    max_imfs : int
        Maximum number of IMFs to extract (default 10).
    n_freq_bins : int
        Number of frequency bins for the Hilbert spectrum (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs``, ``residue``,
        ``inst_freqs`` (list), ``inst_amps`` (list),
        ``hilbert_spectrum`` (n_freq_bins x n_samples),
        ``marginal_spectrum``, ``freq_axis``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Huang, N.E. et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. Lond. A*, 454, 903--995.
    doi:10.1098/rspa.1998.0193
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    residue = x.copy()
    imfs = []

    for _ in range(max_imfs):
        imf = _sift_imf(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    inst_freqs = []
    inst_amps = []
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freqs.append(freq)
        inst_amps.append(amp)

    freq_axis = np.linspace(0, fs / 2, n_freq_bins)
    hs = np.zeros((n_freq_bins, n))
    df = freq_axis[1] - freq_axis[0] if n_freq_bins > 1 else 1.0
    for freq_arr, amp_arr in zip(inst_freqs, inst_amps):
        for t_idx in range(n):
            f_val = abs(freq_arr[t_idx])
            bin_idx = int(f_val / (df + 1e-12))
            if 0 <= bin_idx < n_freq_bins:
                hs[bin_idx, t_idx] += amp_arr[t_idx] ** 2

    marginal = np.sum(hs, axis=1)

    return DescriptiveResult(
        name="hilbert_huang_spectrum",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "inst_freqs": inst_freqs,
            "inst_amps": inst_amps,
            "hilbert_spectrum": hs,
            "marginal_spectrum": marginal,
            "freq_axis": freq_axis,
        },
    )


hhtrf = hilbert_huang_spectrum


# -- rgampd: Amplitude demodulation (envelope via Hilbert transform).
def cdemod(x, fs=1.0, f0=None, bandwidth=None):
    """Complex demodulation: time-varying amplitude and phase at one frequency.

    Why this exists: a band-pass filter tells you how much energy sits in a
    band, but not how that energy's amplitude and phase drift over time.
    Complex demodulation shifts the band of interest down to DC and low-pass
    filters what is left, so what comes out is the instantaneous envelope and
    phase of that one component -- which is how respiratory modulation is
    tracked in an HRV series, or how a single spectral peak is followed
    through a nonstationary recording.

    Rangayyan & Krishnan (2024) Sec 5.5.1 "Amplitude demodulation" gives the
    procedure.  Eq (5.16) writes an arbitrary signal as

        x(t) = a(t) cos[w0 t + psi(t)] + xr(t),

    where a(t) and psi(t) are "the time-varying amplitude and phase of the
    component at w0" and xr(t) is the rest of the signal.  Eq (5.18) shifts
    the signal by -w0 via multiplication with 2 exp(-j w0 t),

        y(t) = 2 x(t) exp(-j w0 t),

    which the book decomposes into a term at DC, a term centred at 2 w0 and a
    term centred at w0; "a lowpass filter may be used to extract the first
    term" to obtain eq (5.19),

        y0(t) ~ a(t) exp[j psi(t)],

    after which "the desired entities may then be extracted as
    a(t) ~ |y0(t)| and psi(t) ~ angle y0(t)".  The book notes that "the
    frequency resolution of the method depends on the bandwidth of the lowpass
    filter used", which is the ``bandwidth`` argument here.

    NOTE ON THE PREVIOUS DOCSTRING: this function was documented as
    "envelope via Hilbert transform", citing Sec 5.5.1.  Sec 5.5.1 does not
    define a Hilbert envelope -- that is Sec 5.5.3, and it already has its own
    function in this module.  Sec 5.5.1 is rectification/lowpass and complex
    demodulation, eqs (5.15)-(5.19), which is what is implemented here.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    f0 : float or None
        Demodulation frequency w0/(2 pi) in Hz.  Default is the dominant
        spectral peak of the signal.
    bandwidth : float or None
        Half-width in Hz of the lowpass filter applied after the shift.
        Default fs/16.  Must be smaller than f0 so the image at 2 w0 is
        rejected.

    Returns
    -------
    RichResult with keys ``amplitude`` (a(t), eq 5.19), ``phase``
    (psi(t), unwrapped), ``demodulated`` (complex y0(t)), ``f0``,
    ``bandwidth``, ``mean_amplitude``, ``method``.

    Raises
    ------
    ValueError
        If fs, f0 or bandwidth is not positive, f0 exceeds the Nyquist
        frequency, or the bandwidth is too wide to reject the 2 w0 image.
    """
    v = _tfneed(x, "x", 4)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    n = len(v)
    X = _tfdft(v)
    if f0 is None:
        half = n // 2 + 1
        kbest = max(range(1, half), key=lambda k: abs(X[k])) if half > 1 else 0
        f0 = kbest * fs / n
    f0 = float(f0)
    if f0 <= 0.0:
        raise ValueError("f0 must be positive")
    if f0 >= fs / 2.0:
        raise ValueError(f"f0={f0} must be below the Nyquist frequency {fs / 2.0}")
    bw = fs / 16.0 if bandwidth is None else float(bandwidth)
    if bw <= 0.0:
        raise ValueError("bandwidth must be positive")
    if bw >= f0:
        raise ValueError(
            f"bandwidth={bw} Hz is not smaller than f0={f0} Hz; the image at "
            f"2*f0 (eq 5.18) would leak through the lowpass filter"
        )
    # eq (5.18): shift by -w0
    y = [2.0 * v[i] * cmath.exp(-2j * pi * f0 * i / fs) for i in range(n)]
    # Ideal (brick-wall) lowpass of half-width bw, applied in the DFT domain.
    Y = _tfdft(y)
    kc = max(1, int(bw * n / fs))
    Z = [0j] * n
    for k in range(n):
        kk = k if k <= n // 2 else k - n
        if abs(kk) <= kc:
            Z[k] = Y[k]
    y0 = _tfidft(Z)
    amp = [abs(t) for t in y0]
    ph = [atan2(t.imag, t.real) for t in y0]
    unw = [ph[0]]
    for i in range(1, n):
        d = ph[i] - ph[i - 1]
        while d > pi:
            d -= 2.0 * pi
        while d < -pi:
            d += 2.0 * pi
        unw.append(unw[-1] + d)
    return RichResult(
        payload={
            "amplitude": amp,
            "phase": unw,
            "demodulated": y0,
            "f0": f0,
            "bandwidth": bw,
            "mean_amplitude": fsum(amp) / n,
            "method": "Complex demodulation, Rangayyan & Krishnan (2024) "
                      "Sec 5.5.1 eqs (5.16)-(5.19)",
        }
    )


rangayyan_amplitude_demod = cdemod  # pre-policy spelling


# -- rgbiorth: Biorthogonal wavelet (symmetric, linear phase) DWT.
def biordwt(x, wavelet="bior2.2", levels=3):
    """Biorthogonal (symmetric, linear-phase) wavelet transform.

    Why this exists: no orthogonal wavelet except the two-tap one can be
    symmetric.  Asymmetric filters have nonlinear phase, which shifts the
    features of a signal by a scale-dependent amount -- fatal when the point
    of the analysis is WHERE an event happened, as in ECG fiducial-point
    detection.  Giving up orthogonality (analysis and synthesis filters are
    now different but dual) buys exact symmetry and exact linear phase while
    keeping perfect reconstruction.

    The wavelet implemented is the 5/3 pair -- analysis filters of length 5
    (lowpass) and 3 (highpass), both symmetric.  Rangayyan & Krishnan (2024)
    mentions biorthogonal wavelets only in passing, in Sec 8.14 on PPG
    denoising ("five different types of wavelets: Daubechies, biorthogonal,
    reverse biorthogonal, symlet, and Coiflet were applied for denoising"),
    and does NOT define them or give any filter coefficients.  The primary
    sources are
      Cohen, A., Daubechies, I., & Feauveau, J.-C. (1992). "Biorthogonal
      bases of compactly supported wavelets." Comm. Pure Appl. Math.
      45(5):485-560,
    and, for the lifting factorisation used here,
      Daubechies, I., & Sweldens, W. (1998). "Factoring wavelet transforms
      into lifting steps." J. Fourier Anal. Appl. 4(3):247-269, Sec 6.

    Lifting is used rather than direct convolution because the inverse is
    then the analysis steps run backwards with the signs flipped: perfect
    reconstruction is structural, not something the filter design has to
    achieve numerically.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'bior2.2'`` / ``'5/3'`` / ``'cdf53'`` -- the only pair implemented.
    levels : int
        Number of decomposition levels.

    Returns
    -------
    RichResult with keys ``approx``, ``details``, ``coeffs``, ``lengths``,
    ``reconstructed``, ``max_reconstruction_error``, ``levels``,
    ``symmetric``, ``method``.

    Raises
    ------
    ValueError
        For an unsupported wavelet name, levels < 1, or a signal too short.
    """
    v = _tfneed(x, "x", 4)
    w = str(wavelet).strip().lower().replace("-", "").replace("_", "")
    if w not in ("bior2.2", "bior22", "5/3", "53", "cdf53", "legall"):
        raise ValueError(
            f"unsupported biorthogonal wavelet {wavelet!r}; only the 5/3 "
            f"(bior2.2) pair is implemented"
        )
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")

    def fwd(a):
        n = len(a)
        if n < 2:
            raise ValueError("signal too short for another level")
        if n % 2:
            a = list(a) + [a[-1]]
            n += 1
        s = [a[2 * i] for i in range(n // 2)]
        d = [a[2 * i + 1] for i in range(n // 2)]
        # predict: d_k <- d_k - (s_k + s_{k+1}) / 2   (symmetric extension)
        d = [d[k] - 0.5 * (s[k] + s[min(k + 1, len(s) - 1)]) for k in range(len(d))]
        # update: s_k <- s_k + (d_{k-1} + d_k) / 4
        s = [s[k] + 0.25 * (d[max(k - 1, 0)] + d[k]) for k in range(len(s))]
        return s, d

    def inv(s, d, n):
        s = [s[k] - 0.25 * (d[max(k - 1, 0)] + d[k]) for k in range(len(s))]
        d = [d[k] + 0.5 * (s[k] + s[min(k + 1, len(s) - 1)]) for k in range(len(d))]
        out = []
        for k in range(len(s)):
            out.append(s[k])
            out.append(d[k])
        return out[:n]

    a = list(v)
    details, lengths = [], []
    for _ in range(lv):
        if len(a) < 2:
            raise ValueError(f"levels={lv} is too many for a signal of length {len(v)}")
        lengths.append(len(a))
        a, d = fwd(a)
        details.append(d)
    details.reverse()
    lengths.reverse()
    rec = list(a)
    for d, n in zip(details, lengths):
        rec = inv(rec, d, n)
    err = max(abs(rec[i] - v[i]) for i in range(len(v)))
    coeffs = [list(a)] + [list(c) for c in details]
    return RichResult(
        payload={
            "approx": list(a),
            "details": [list(c) for c in details],
            "coeffs": coeffs,
            "lengths": lengths,
            "levels": lv,
            "reconstructed": rec,
            "max_reconstruction_error": err,
            "symmetric": True,
            "wavelet": "bior2.2 (CDF 5/3)",
            "method": "Biorthogonal 5/3 (CDF) wavelet transform via lifting; "
                      "Cohen, Daubechies & Feauveau (1992) and Daubechies & "
                      "Sweldens (1998) -- not defined in Rangayyan & Krishnan",
        }
    )


rangayyan_biorthogonal_wvlt = biordwt  # pre-policy spelling


# -- rgchoi: Choi-Williams distribution (exponential kernel).
def expkertfd(x, fs=1.0, sigma=1.0, nfreq=None, maxlag=None):
    """Exponential-kernel time-frequency distribution.

    Why this exists: the separable Gaussian smoothing of :func:`gtfd` blurs
    the whole plane uniformly, so it costs auto-term resolution everywhere in
    order to kill cross terms that live only off the diagonal.  The
    exponential kernel is shaped instead: it is unity along both the theta
    and the tau axes -- where the auto terms sit -- and decays away from them,
    where the cross terms sit.  So it suppresses interference while leaving
    the marginals (eqs 8.118 and 8.119) intact.

    This is the distribution usually called the Choi-Williams distribution.
    It is a member of Cohen's class as defined by Rangayyan & Krishnan (2024)
    eq (8.124), with kernel Phi(theta, tau), but the book does NOT give this
    kernel or name this distribution anywhere; do not attribute the kernel to
    it.  The primary source is
      Choi, H.-I., & Williams, W. J. (1989). "Improved time-frequency
      representation of multicomponent signals using exponential kernels."
      IEEE Trans. Acoustics, Speech and Signal Processing 37(6):862-871,
    which sets Phi(theta, tau) = exp(-(theta tau)^2 / sigma).  In the
    time-lag domain that kernel becomes the Gaussian running average

        w_sigma(mu, tau) = sqrt(sigma / (4 pi tau^2))
                           exp(-sigma mu^2 / (4 tau^2)),

    whose width grows with the lag |tau| -- which is exactly why it does not
    blur the zero-lag (auto-term) part of the kernel.  Small sigma smooths
    hard, large sigma tends to the plain WVD of eq (8.123).

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    sigma : float
        Kernel width parameter; must be positive.
    nfreq : int or None
        Frequency bins over [0, fs/2).  Default len(x).
    maxlag : int or None
        Largest lag |tau| retained.  Default len(x) // 4, which bounds the
        O(N * maxlag * (2 maxlag + 1) * nfreq) cost of the running average.

    Returns
    -------
    RichResult with keys ``tfd``, ``times``, ``freqs``, ``sigma``,
    ``maxlag``, ``peak_freq``, ``crossterm_ratio``, ``method``.

    Raises
    ------
    ValueError
        If sigma or fs is not positive, or the signal is too short.
    """
    v = _tfneed(x, "x", 8)
    fs = float(fs)
    sigma = float(sigma)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    n = len(v)
    nf = int(nfreq or n)
    ml = max(1, n // 4) if maxlag is None else int(maxlag)
    if ml < 1:
        raise ValueError("maxlag must be >= 1")
    z = _tfanalytic(v)
    freqs = [k * fs / (2.0 * nf) for k in range(nf)]
    tfd = []
    for i in range(n):
        ker = []
        lags = []
        for tau in range(-ml, ml + 1):
            if i + tau < 0 or i - tau < 0 or i + tau >= n or i - tau >= n:
                continue
            if tau == 0:
                ker.append(z[i] * z[i].conjugate())
                lags.append(0)
                continue
            # Gaussian running average in mu, width proportional to |tau|.
            sd = 2.0 * abs(tau) / sqrt(2.0 * sigma)
            span = max(1, int(3.0 * sd))
            acc = 0j
            wsum = 0.0
            for mu in range(-span, span + 1):
                a, b = i + mu + tau, i + mu - tau
                if a < 0 or b < 0 or a >= n or b >= n:
                    continue
                wgt = exp(-sigma * mu * mu / (4.0 * tau * tau))
                acc += wgt * z[a] * z[b].conjugate()
                wsum += wgt
            if wsum <= 0.0:
                continue
            ker.append(acc / wsum)
            lags.append(tau)
        row = []
        for k in range(nf):
            w = -2.0 * pi * k / nf
            acc = 0j
            for j, tau in enumerate(lags):
                acc += ker[j] * cmath.exp(1j * w * tau)
            row.append(2.0 * acc.real)
        tfd.append(row)
    tot = fsum(abs(t) for r in tfd for t in r)
    neg = fsum(-t for r in tfd for t in r if t < 0.0)
    col = [fsum(row[k] for row in tfd) for k in range(nf)]
    return RichResult(
        payload={
            "tfd": tfd,
            "times": [i / fs for i in range(n)],
            "freqs": freqs,
            "sigma": sigma,
            "maxlag": ml,
            "peak_freq": freqs[max(range(nf), key=lambda k: col[k])],
            "crossterm_ratio": (neg / tot) if tot > 0.0 else 0.0,
            "method": "Exponential-kernel (Choi-Williams) TFD, Choi & Williams "
                      "(1989) IEEE TASSP 37(6):862-871; a member of Cohen's "
                      "class, Rangayyan & Krishnan (2024) eq (8.124)",
        }
    )


rangayyan_choi_williams = expkertfd  # pre-policy spelling


# -- rgcpr: CPR analysis via wavelet for shockable rhythm detection.
def cprwt(ecg, fs=250.0, scales=None, w0=5.0, band=(3.0, 21.0)):
    """Scale-distribution width of a fibrillation waveform, for CPR studies.

    Why this exists: during resuscitation the question is not "is this
    ventricular fibrillation" but "has this fibrillation waveform become
    organised enough to be worth shocking".  Organisation is a morphology
    question, and it shows up as how many wavelet scales are needed to hold
    the signal's energy: an organised waveform concentrates into a few scales,
    a disorganised one spreads over many.  The width of that scale-energy
    distribution is the marker.

    Rangayyan & Krishnan (2024) Sec 8.15 "Application: Wavelet Analysis for
    CPR Studies" describes the method of its reference [92] (Umapathy et al.).
    The section states that "the approach described here employs the CWT; see
    Equation 8.107", that Umapathy et al. "determined that the Morlet wavelet
    provided the best fit to the ventricular fibrillation waveforms", that
    "a bandpass filter (3 - 21 Hz) was implemented to remove low- and
    high-frequency artifacts", and that segments were "5 s segments (1,250
    samples at 250 Hz)".  The energy decomposition is stated as
    "Ex = Es1 + Es2 + Es3 + ... + EsN", and the feature is the scale
    distribution width: "the normalized energy distribution was computed at
    all scales and the width of the distribution was used as the feature,
    denoted as the scale distribution width (SDW)".  The book also states the
    measurement rule: "Umapathy et al. opted to measure the width of the
    distribution at half the height of the apex", i.e. a full width at half
    maximum, which is what is computed here.  Interpretation, per the same
    section: "smaller widths imply more ordered components and larger widths
    indicate disorganized components".

    Parameters
    ----------
    ecg : sequence of float
        ECG / fibrillation waveform samples.
    fs : float
        Sampling rate in Hz.  Default 250 Hz, the rate used in Sec 8.15.
    scales : sequence of float or None
        CWT scales.  Default is the ladder whose pseudo-frequencies cover the
        band, per "a range of wavelet scales whose associated frequencies
        encompassed the ventricular fibrillation bandwidth".
    w0 : float
        Morlet central frequency, eq (8.116).
    band : (float, float)
        The 3-21 Hz analysis band of Sec 8.15.  The signal is band-limited to
        this range before the CWT.

    Returns
    -------
    RichResult with keys ``sdw`` (scale distribution width, in scale-ladder
    units), ``scale_energy`` (normalised, sums to 1), ``scales``, ``freqs``,
    ``peak_scale``, ``peak_freq``, ``organised`` (True when the SDW is below
    half the number of scales -- a readable summary, NOT a clinical
    threshold; the book gives none), ``band``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive, the band is not increasing, or the band's
        upper edge is above the Nyquist frequency.
    """
    v = _tfneed(ecg, "ecg", 16)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"band must satisfy 0 < low < high, got {band!r}")
    if hi > fs / 2.0:
        raise ValueError(f"band upper edge {hi} Hz exceeds Nyquist {fs / 2.0} Hz")
    n = len(v)
    # 3-21 Hz bandpass, applied in the DFT domain (Sec 8.15).
    X = _tfdft(v)
    Y = [0j] * n
    for k in range(n):
        f = (k if k <= n // 2 else k - n) * fs / n
        if lo <= abs(f) <= hi:
            Y[k] = X[k]
    filt = [t.real for t in _tfidft(Y)]
    if scales is None:
        sc, s = [], max(1.0, w0 * fs / (2.0 * pi * hi))
        top = w0 * fs / (2.0 * pi * lo)
        while s <= top and len(sc) < 32:
            sc.append(s)
            s *= 2.0 ** 0.25
        scales = sc or [1.0]
    sc = [float(s) for s in scales]
    co = _tfcwt(filt, sc, "morlet", float(w0))
    ener = [fsum(abs(c) ** 2 for c in row) for row in co]
    tot = fsum(ener)
    if tot <= 0.0:
        raise ValueError("no energy in the 3-21 Hz band; SDW is undefined")
    norm = [e / tot for e in ener]
    pk = max(range(len(norm)), key=lambda i: norm[i])
    halfmax = norm[pk] / 2.0
    # Full width at half maximum, counted in scale-ladder steps (Sec 8.15).
    lo_i = pk
    while lo_i > 0 and norm[lo_i - 1] >= halfmax:
        lo_i -= 1
    hi_i = pk
    while hi_i < len(norm) - 1 and norm[hi_i + 1] >= halfmax:
        hi_i += 1
    # Linear interpolation of the two half-maximum crossings, so the width is
    # continuous rather than quantised to whole ladder steps.
    left = float(lo_i)
    if lo_i > 0 and norm[lo_i] > norm[lo_i - 1]:
        left = lo_i - (norm[lo_i] - halfmax) / (norm[lo_i] - norm[lo_i - 1])
    right = float(hi_i)
    if hi_i < len(norm) - 1 and norm[hi_i] > norm[hi_i + 1]:
        right = hi_i + (norm[hi_i] - halfmax) / (norm[hi_i] - norm[hi_i + 1])
    sdw = right - left
    return RichResult(
        payload={
            "sdw": sdw,
            "scale_energy": norm,
            "scales": sc,
            "freqs": [w0 * fs / (2.0 * pi * s) for s in sc],
            "peak_scale": sc[pk],
            "peak_freq": w0 * fs / (2.0 * pi * sc[pk]),
            "organised": bool(sdw < len(sc) / 2.0),
            "band": (lo, hi),
            "method": "Wavelet scale distribution width (SDW) of a "
                      "fibrillation waveform, Rangayyan & Krishnan (2024) "
                      "Sec 8.15, Morlet CWT of eqs (8.107)/(8.116), 3-21 Hz "
                      "band, FWHM of the normalised scale-energy distribution",
        }
    )


rangayyan_cpr_analysis = cprwt  # pre-policy spelling


# -- rgcwt: Continuous wavelet transform (CWT).
def cwt(x, fs=1.0, wavelet="morlet", scales=None, w0=5.0):
    """Continuous wavelet transform: correlate the signal with scaled wavelets.

    Why this exists: the STFT uses one window width for the whole plane, so
    its resolution cell is the same at 2 Hz and at 200 Hz.  Biomedical events
    are not like that -- an ECG QRS complex is brief and broadband, the T wave
    is long and narrow-band.  The CWT scales the analysing function with
    frequency, so it buys fine time resolution where the signal is fast and
    fine frequency resolution where it is slow.

    Rangayyan & Krishnan (2024) Sec 8.8, eq (8.107):

        X(tau, s) = (1/sqrt(s)) integral x(t) psi*((t - tau)/s) dt.

    The text explains that the coefficient "indicates the commonality between
    the two functions, or the extent or strength of the wavelet that is
    present in the signal".  Eq (8.108) is the finite-energy requirement and
    eq (8.109) the admissibility condition on the mother wavelet.  The two
    mother wavelets offered here are the book's own: eq (8.115), the Mexican
    hat psi(t) = (1 - t^2) exp(-0.5 t^2), described there as "the negative of
    the second derivative of a Gaussian"; and eq (8.116), the Morlet
    psi(t) = pi^(-1/4) [exp(j w0 t) - exp(-0.5 w0^2)] exp(-0.5 t^2), whose
    subtracted constant is the admissibility correction.

    Following the book's own note that "the time axis is discretized to the
    same sampling interval as that of the signal", tau runs over sample
    indices and the scale s is in samples.  The wavelet is truncated at its
    numerical support, so the cost is O(N * sum(scales)).

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz (used only to report pseudo-frequencies).
    wavelet : str
        ``'morlet'`` (eq 8.116), ``'mexh'`` (eq 8.115) or ``'haar'``.
    scales : sequence of float or None
        Scales in samples.  Default is the dyadic ladder 1, 2, 4, ... up to
        an eighth of the signal length, matching the book's Figure 8.28,
        which uses scales 1, 2, 4 and 8.
    w0 : float
        Morlet central frequency of eq (8.116).  Ignored otherwise.

    Returns
    -------
    RichResult with keys ``coeffs`` (scales x time, complex for Morlet),
    ``scales``, ``freqs`` (pseudo-frequency per scale), ``times``,
    ``energy_per_scale``, ``peak_scale``, ``method``.

    Raises
    ------
    ValueError
        For a non-positive scale, unknown wavelet or non-positive fs.
    """
    v = _tfneed(x, "x", 4)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if scales is None:
        sc, s = [], 1.0
        while s <= max(1.0, len(v) / 8.0):
            sc.append(s)
            s *= 2.0
        scales = sc or [1.0]
    sc = [float(s) for s in scales]
    if not sc:
        raise ValueError("scales must not be empty")
    for s in sc:
        if s <= 0.0:
            raise ValueError("all scales must be positive")
    co = _tfcwt(v, sc, wavelet, float(w0))
    # Pseudo-frequency: the centre frequency of the mother wavelet divided by
    # the scale.  Sec 8.8 notes "there exists an inverse relationship between
    # scale and frequency" (below eq 8.115) and illustrates it in Figure 8.30.
    fc = {"morlet": float(w0) / (2.0 * pi), "mexh": 0.25, "mexicanhat": 0.25,
          "sombrero": 0.25, "ricker": 0.25, "haar": 0.5,
          "db1": 0.5}[str(wavelet).strip().lower()]
    epr = [fsum(abs(c) ** 2 for c in row) for row in co]
    return RichResult(
        payload={
            "coeffs": co,
            "scales": sc,
            "freqs": [fc * fs / s for s in sc],
            "times": [i / fs for i in range(len(v))],
            "energy_per_scale": epr,
            "peak_scale": sc[max(range(len(sc)), key=lambda i: epr[i])],
            "wavelet": str(wavelet),
            "method": "Continuous wavelet transform, Rangayyan & Krishnan "
                      "(2024) eq (8.107); mother wavelets eqs (8.115)/(8.116)",
        }
    )


rangayyan_cwt = cwt  # pre-policy spelling


# -- rgcwvd: Cohen's class TFDs via kernel function.
def gtfd(x, fs=1.0, kernel="spwvd", nfreq=None, tsmooth=None, fsmooth=None):
    """Cohen's class generalised TFD: a kernel-smoothed Wigner-Ville surface.

    Why this exists: every shift-invariant quadratic time-frequency
    distribution is the WVD convolved with some 2-D kernel.  Choosing that
    kernel is the whole design problem -- a wide kernel kills the cross terms
    that make a raw WVD unreadable, a narrow one keeps the localisation.
    This function exposes the choice rather than hard-coding one distribution.

    Rangayyan & Krishnan (2024) Sec 8.9, eq (8.124), defines the generalised
    TFD as a triple integral of x(u + tau/2) x*(u - tau/2) against a
    transformation kernel Phi(theta, tau), and states that "the kernel acts as
    a lowpass filter and minimizes cross-terms".  Eq (8.125) gives the
    equivalent and computationally far cheaper time-frequency-domain form,
    the smoothed WVD

        SWVD(t, w) = (1/2 pi) int int phi(u, Om) WVD(t - u, w - Om) du dOm,

    eq (8.126) factorises the kernel as phi(u, Om) = g(u) H(Om), and eq
    (8.127) names the result of a separable smoothing the smoothed
    pseudo-WVD (SPWVD).  The book states that "Gaussian functions are
    commonly used as smoothing windows", which is what is used here, and
    warns that "smoothing windows suppress cross-terms but smear localized
    components".

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    kernel : str
        ``'wvd'``   -- Phi = 1, no smoothing (recovers eq 8.123).
        ``'pwvd'``  -- frequency smoothing only (pseudo-WVD).
        ``'swvd'``  -- time smoothing only (eq 8.125 with H = delta).
        ``'spwvd'`` -- separable smoothing in both, eqs (8.126)-(8.127).
    nfreq : int or None
        Frequency bins over [0, fs/2).  Default len(x).
    tsmooth, fsmooth : int or None
        Lengths of g(u) and H(Omega) in samples/bins.  Default is about an
        eighth of the corresponding axis, at least 3.

    Returns
    -------
    RichResult with keys ``tfd``, ``times``, ``freqs``, ``kernel``,
    ``tsmooth``, ``fsmooth``, ``peak_freq``, ``crossterm_ratio``, ``method``.
    ``crossterm_ratio`` is the fraction of the total distribution mass that
    is negative; a nonnegative TFD scores 0 (see the nonnegativity criterion
    stated in Sec 8.9 above eq 8.118).

    Raises
    ------
    ValueError
        For an unknown kernel or a non-positive sampling rate.
    """
    v = _tfneed(x, "x", 4)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    k = str(kernel).strip().lower()
    if k not in ("wvd", "pwvd", "swvd", "spwvd"):
        raise ValueError(
            f"unknown kernel {kernel!r}; use 'wvd', 'pwvd', 'swvd' or 'spwvd'"
        )
    nf = int(nfreq or len(v))
    tfd, freqs = _tfwvd(v, fs, nf)
    tl = max(3, len(v) // 8) if tsmooth is None else int(tsmooth)
    fl = max(3, nf // 8) if fsmooth is None else int(fsmooth)
    if tl < 1 or fl < 1:
        raise ValueError("tsmooth and fsmooth must be >= 1")
    if k == "wvd":
        tl = fl = 1
    elif k == "pwvd":
        tl = 1
    elif k == "swvd":
        fl = 1
    if tl > 1 or fl > 1:
        tfd = _tfsmooth2d(tfd, tl, fl)
    tot = fsum(abs(t) for r in tfd for t in r)
    neg = fsum(-t for r in tfd for t in r if t < 0.0)
    col = [fsum(row[j] for row in tfd) for j in range(nf)]
    return RichResult(
        payload={
            "tfd": tfd,
            "times": [i / fs for i in range(len(v))],
            "freqs": freqs,
            "kernel": k,
            "tsmooth": tl,
            "fsmooth": fl,
            "peak_freq": freqs[max(range(nf), key=lambda j: col[j])],
            "crossterm_ratio": (neg / tot) if tot > 0.0 else 0.0,
            "method": "Cohen's class generalised TFD, Rangayyan & Krishnan "
                      "(2024) eq (8.124), evaluated as the smoothed WVD of "
                      "eqs (8.125)-(8.127) with separable Gaussian kernels",
        }
    )


rangayyan_cohen_class = gtfd  # pre-policy spelling


# -- rgdaub: Daubechies wavelet filter coefficients (db2-db10).
def orthfilt(order=4):
    """Orthogonal (Daubechies) wavelet filter bank taps and their properties.

    Why this exists: every discrete wavelet routine in this shelf is a
    two-channel filter bank, and the only thing that distinguishes one
    wavelet from another is the four filters.  Exposing them lets a caller
    check the properties that make the transform invertible instead of taking
    them on faith, and lets a caller build their own decomposition.

    Rangayyan & Krishnan (2024) Sec 8.8 names the Daubechies family among
    "the commonly used wavelets" (paragraph before eq 8.115) and cites
      Daubechies, I. (1990). "The wavelet transform, time-frequency
      localization and signal analysis." IEEE Trans. Information Theory
      36(5):961-1005  (the book's reference [74]),
    but the book does NOT tabulate the coefficients.  The taps returned here
    are from
      Daubechies, I. (1992). Ten Lectures on Wavelets. SIAM, Table 6.1,
    normalised so that sum(h) = sqrt(2).  Order 1 is the Haar wavelet.

    The three identities reported -- sum(h) = sqrt(2), sum(h^2) = 1 and
    sum(h[n] h[n+2m]) = 0 for m != 0 -- are the double-shift orthonormality
    conditions.  They are computed, not asserted, so a corrupted table is
    visible in the output.

    Parameters
    ----------
    order : int
        Daubechies order 1..10 (order 1 = Haar, filter length 2 * order).

    Returns
    -------
    RichResult with keys ``dec_lo``, ``dec_hi``, ``rec_lo``, ``rec_hi``,
    ``order``, ``length``, ``vanishing_moments``, ``sum_lo``, ``norm_lo``,
    ``max_shift_inner_product``, ``method``.

    Raises
    ------
    ValueError
        If order is outside 1..10.
    """
    k = int(order)
    if k not in _DBTAPS:
        raise ValueError(f"order must be an integer in 1..10, got {order!r}")
    h, g, rl, rh = _tffilters(f"db{k}")
    L = len(h)
    worst = 0.0
    for m in range(1, L // 2):
        worst = max(worst, abs(fsum(h[n] * h[n + 2 * m] for n in range(L - 2 * m))))
    return RichResult(
        payload={
            "dec_lo": list(h),
            "dec_hi": list(g),
            "rec_lo": list(rl),
            "rec_hi": list(rh),
            "order": k,
            "length": L,
            "vanishing_moments": k,
            "sum_lo": fsum(h),
            "norm_lo": fsum(t * t for t in h),
            "max_shift_inner_product": worst,
            "method": "Daubechies orthogonal scaling/wavelet filters, "
                      "Daubechies (1992) Ten Lectures on Wavelets Table 6.1; "
                      "family cited by Rangayyan & Krishnan (2024) Sec 8.8 "
                      "but not tabulated there",
        }
    )


rangayyan_daubechies = orthfilt  # pre-policy spelling


# -- rgdtfd: Decomposition-based adaptive TFD using MP atoms.
def atomtfd(x, fs=1.0, dictionary="gabor", max_atoms=8, nfreq=None, min_decay=1e-3):
    """Decomposition-based adaptive TFD: the WVD of matching-pursuit atoms.

    Why this exists: cross terms in a bilinear TFD arise from the interaction
    of pairs of signal components.  Smoothing (see :func:`gtfd`) removes them
    by blurring everything.  The adaptive route instead decomposes the signal
    into components FIRST, then sums the WVD of each component separately --
    the cross terms are never formed, so nothing has to be blurred away and
    the localisation of each atom survives intact.

    Rangayyan & Krishnan (2024) Sec 9.3 "Matching Pursuit" gives the
    decomposition: eq (9.1) expands x(t) over TF atoms, eq (9.2) defines the
    atom as a scaled, translated and modulated window

        g_gamma(t) = (1/sqrt(s)) g((t - tau)/s) exp[j(2 pi f t + phi)],

    eq (9.3) gives the Gaussian window g(t) = 2^(1/4) exp(-pi t^2) whose
    atoms are "known as Gabor atoms", eqs (9.4)-(9.5) give the greedy
    residual iteration, eq (9.6) defines the decay parameter

        lambda(m) = sqrt(1 - ||R_m x||^2 / ||R_(m-1) x||^2)

    used as a stopping rule, and eq (9.7) is the coherent reconstruction.
    Sec 9.6 "Decomposition-based Adaptive TFD" then states that the MPTFD "is
    obtained by taking the WVD of the TF atoms used to represent the signal
    as in Equation 9.7", eq (9.15), keeping only the diagonal terms.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    dictionary : str
        ``'gabor'`` -- Gaussian-envelope atoms, eq (9.3).
        ``'fourier'`` -- pure tones (no envelope), the degenerate dictionary.
    max_atoms : int
        Iteration cap M in eq (9.5).
    nfreq : int or None
        Frequency bins over [0, fs/2).  Default len(x).
    min_decay : float
        Stop once the eq (9.6) decay parameter falls below this.

    Returns
    -------
    RichResult with keys ``tfd``, ``times``, ``freqs``, ``atoms``
    (one dict per atom: scale, translation, frequency, coefficient),
    ``residual_energy``, ``explained``, ``n_atoms``, ``decay``, ``method``.

    Raises
    ------
    ValueError
        For an unknown dictionary or non-positive fs / max_atoms.
    """
    v = _tfneed(x, "x", 8)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    dic = str(dictionary).strip().lower()
    if dic not in ("gabor", "fourier"):
        raise ValueError(f"unknown dictionary {dictionary!r}; use 'gabor' or 'fourier'")
    ma = int(max_atoms)
    if ma < 1:
        raise ValueError("max_atoms must be >= 1")
    n = len(v)
    nf = int(nfreq or n)
    # Dyadic scales and translations on a coarse grid -- eq (9.2) parameters
    # (s, tau, f).  A coarse grid is the honest tradeoff here: matching
    # pursuit over a fully dense dictionary is O(N^2 log N) per iteration.
    scales = [s for s in (n / 2.0, n / 4.0, n / 8.0, n / 16.0) if s >= 2.0]
    if dic == "fourier":
        scales = [float(n)]
    shifts = list(range(0, n, max(1, n // 8)))
    kmax = nf
    atoms = []

    def atom(s, tau, k):
        f = k * fs / (2.0 * nf)
        out = []
        for i in range(n):
            if dic == "gabor":
                u = (i - tau) / s
                env = (2.0 ** 0.25) * exp(-pi * u * u) / sqrt(s)
            else:
                env = 1.0 / sqrt(n)
            out.append(env * cmath.exp(2j * pi * f * i / fs))
        nrm = sqrt(fsum(abs(c) ** 2 for c in out))
        if nrm <= 0.0:
            raise ValueError("degenerate atom (zero norm)")
        return [c / nrm for c in out]

    res = [complex(t, 0.0) for t in _tfanalytic(v)]
    e0 = _tfenergy(res)
    decay = []
    prev = e0
    for _ in range(ma):
        best, bestval = None, -1.0
        for s in scales:
            for tau in shifts:
                for k in range(1, kmax):
                    g = atom(s, tau, k)
                    ip = sum(res[i] * g[i].conjugate() for i in range(n))
                    if abs(ip) > bestval:
                        bestval, best = abs(ip), (s, tau, k, ip, g)
        if best is None:
            break
        s, tau, k, ip, g = best
        res = [res[i] - ip * g[i] for i in range(n)]
        cur = _tfenergy(res)
        lam = sqrt(max(0.0, 1.0 - cur / prev)) if prev > 0.0 else 0.0
        decay.append(lam)
        atoms.append({"scale": s, "translation": tau,
                      "freq": k * fs / (2.0 * nf), "coeff": abs(ip)})
        prev = cur
        if lam < float(min_decay):
            break

    freqs = [k * fs / (2.0 * nf) for k in range(nf)]
    tfd = [[0.0] * nf for _ in range(n)]
    for a in atoms:
        # WVD of one Gabor atom, eq (9.15): a Gaussian blob at (tau, f)
        # weighted by |<R_n x, g_gamma>|^2.  Formed analytically rather than
        # by running _tfwvd on each atom, which would be N^2 per atom.
        s, tau, f, c = a["scale"], a["translation"], a["freq"], a["coeff"]
        df = fs / (2.0 * nf)
        for i in range(n):
            te = exp(-2.0 * pi * ((i - tau) / s) ** 2)
            if te < 1e-12:
                continue
            for k in range(nf):
                fe = exp(-2.0 * pi * ((k * df - f) * s / fs) ** 2)
                tfd[i][k] += (c * c) * te * fe
    col = [fsum(row[k] for row in tfd) for k in range(nf)]
    return RichResult(
        payload={
            "tfd": tfd,
            "times": [i / fs for i in range(n)],
            "freqs": freqs,
            "atoms": atoms,
            "n_atoms": len(atoms),
            "decay": decay,
            "residual_energy": prev,
            "explained": (1.0 - prev / e0) if e0 > 0.0 else 0.0,
            "peak_freq": freqs[max(range(nf), key=lambda k: col[k])] if nf else 0.0,
            "method": "Matching-pursuit TFD (MPTFD), Rangayyan & Krishnan (2024) "
                      "eq (9.15), over the Gabor dictionary of eqs (9.2)-(9.3) "
                      "with the eq (9.6) decay stopping rule",
        }
    )


rangayyan_decomp_tfd = atomtfd  # pre-policy spelling


# -- rgdwt: Discrete wavelet transform (DWT) via filterbank.
def dwt(x, wavelet="db4", levels=3):
    """Discrete wavelet transform by the decimated two-channel filter bank.

    Why this exists: the CWT is massively redundant -- it evaluates every
    scale at every sample.  Sampling the (tau, s) plane on the dyadic grid
    removes that redundancy and turns the transform into a critically sampled
    orthonormal basis: N samples in, N coefficients out, exactly invertible.
    That is what makes wavelet compression and denoising possible at all.

    Rangayyan & Krishnan (2024) Sec 8.8, eq (8.111):

        X(m, n) = integral x(t) psi*_{m,n}(t) dt

    with the discretised wavelet of eq (8.112),

        psi_{m,n}(t) = s0^(-m/2) psi((t - n tau0 s0^m) / s0^m),

    where m and n "control dilation and translation of the wavelet on a
    discrete grid".  Eq (8.113) fixes the dyadic grid s0 = 2, tau0 = 1, giving
    psi_{m,n}(t) = 2^(-m/2) psi(2^(-m) t - n), and the book notes that "the
    use of an appropriate grid for the DWT reduces the redundancy of
    representation as compared to the CWT".  Eq (8.114) is the reconstruction.

    The filter-bank realisation of eqs (8.111)-(8.113) is Mallat's, cited by
    the book as reference [72] (Mallat, S. G., "A theory for multiresolution
    signal decomposition: the wavelet representation", IEEE Trans. PAMI
    11(7):674-693, 1989).  Periodic extension is used, which is the only
    boundary handling for which the decimated bank is an exact orthonormal
    basis of R^N -- so the transform round-trips to machine precision.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Number of decomposition levels m.

    Returns
    -------
    RichResult with keys ``approx`` (cA at the coarsest level), ``details``
    (cD coarse-to-fine), ``coeffs`` (approx followed by details),
    ``lengths``, ``levels``, ``wavelet``, ``energy``, ``method``.

    Raises
    ------
    ValueError
        If levels exceeds what the signal length and filter length allow, or
        the wavelet name is unknown.
    """
    v = _tfneed(x, "x", 2)
    lv = int(levels)
    a, d, ln = _tfdwt(v, wavelet, lv)
    coeffs = [list(a)] + [list(c) for c in d]
    return RichResult(
        payload={
            "approx": list(a),
            "details": [list(c) for c in d],
            "coeffs": coeffs,
            "lengths": ln,
            "levels": lv,
            "wavelet": str(wavelet),
            "energy": fsum(fsum(t * t for t in c) for c in coeffs),
            "method": "Dyadic DWT via the decimated filter bank, Rangayyan & "
                      "Krishnan (2024) eqs (8.111)-(8.113); Mallat (1989) "
                      "algorithm, periodic extension",
        }
    )


rangayyan_dwt = dwt  # pre-policy spelling


# -- rgeemd: Ensemble EMD (EEMD) for mode mixing alleviation.
def emdens(x, n_ensembles=20, noise_std=0.2, max_imfs=8, seed=0):
    """Ensemble EMD: noise-assisted decomposition that fixes mode mixing.

    Why this exists: plain EMD decides where to split the signal from its own
    extrema, so an intermittent burst can throw a whole scale into the wrong
    mode -- one IMF ends up carrying two different oscillations, or one
    oscillation gets scattered over several IMFs.  Adding independent white
    noise to each trial forces the extrema onto a uniform scale grid; averaged
    over trials the noise cancels and what survives is the dyadic filter bank
    the algorithm was supposed to be.

    Rangayyan & Krishnan (2024) Sec 9.4.1 describes mode mixing as the case
    where "different frequency components get captured in the same IMF or a
    single frequency component gets decomposed into multiple IMFs, thereby
    making the interpretation of the IMFs vague", and states that "the main
    idea behind EEMD is the addition of white noise at different scales to the
    signal.  Iteratively, through superposition and averaging, the added white
    noise averages out".  The procedure is given as eq (9.13),

        x_k(n) = x(n) + w_k(n),

    followed by standard EMD of each x_k(n) and ensemble averaging of the
    IMFs.  The book's reference for the method is [17], Wu, Z., & Huang, N. E.,
    "Ensemble empirical mode decomposition: a noise-assisted data analysis
    method", Advances in Adaptive Data Analysis 1(1):1-41, 2009.

    The noise is generated by a seeded linear congruential generator so that
    a given seed always yields the same decomposition; there is no hidden
    global random state.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    n_ensembles : int
        Number of noise realisations to average (M in the book's step 3).
    noise_std : float
        Noise amplitude as a fraction of the signal standard deviation.
    max_imfs : int
        Cap on the number of modes per trial.
    seed : int
        Seed for the deterministic noise generator.

    Returns
    -------
    RichResult with keys ``imfs`` (ensemble-averaged), ``residual``,
    ``n_imfs``, ``n_ensembles``, ``noise_std``, ``seed``,
    ``reconstruction_error``, ``energy_per_imf``, ``method``.  Note that
    ensemble averaging is NOT exactly additive: the reconstruction error is
    the residual noise level, and the book itself frames the averaging as
    approximate ("as M increases, white noise averages to lower levels").

    Raises
    ------
    ValueError
        If n_ensembles < 1 or noise_std is negative.
    """
    v = _tfneed(x, "x", 8)
    ne = int(n_ensembles)
    if ne < 1:
        raise ValueError("n_ensembles must be >= 1")
    ns = float(noise_std)
    if ns < 0.0:
        raise ValueError("noise_std must be non-negative")
    n = len(v)
    mu = fsum(v) / n
    sd = sqrt(fsum((t - mu) ** 2 for t in v) / n)
    amp = ns * (sd if sd > 0.0 else 1.0)
    # Deterministic noise: a seeded LCG plus a Box-Muller pair.  Explicit and
    # reproducible -- no dependence on any module-level RNG state.
    state = (int(seed) * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def unif():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) + 1.0) / (2.0 ** 53 + 1.0)

    mi = int(max_imfs)
    acc = [[0.0] * n for _ in range(mi)]
    cnt = [0] * mi
    resacc = [0.0] * n
    for _k in range(ne):
        noise = []
        while len(noise) < n:
            u1, u2 = unif(), unif()
            r = sqrt(-2.0 * log(u1))
            noise.append(amp * r * cos(2.0 * pi * u2))
            noise.append(amp * r * sin(2.0 * pi * u2))
        xk = [v[i] + noise[i] for i in range(n)]
        imfs, res = _tfemd(xk, mi, 0.05)
        for j, c in enumerate(imfs[:mi]):
            for i in range(n):
                acc[j][i] += c[i]
            cnt[j] += 1
        for i in range(n):
            resacc[i] += res[i]
    out = []
    for j in range(mi):
        if cnt[j] == 0:
            break
        out.append([acc[j][i] / ne for i in range(n)])
    resid = [resacc[i] / ne for i in range(n)]
    tot = [fsum([resid[i]] + [c[i] for c in out]) for i in range(n)]
    err = max(abs(tot[i] - v[i]) for i in range(n))
    return RichResult(
        payload={
            "imfs": out,
            "residual": resid,
            "n_imfs": len(out),
            "n_ensembles": ne,
            "noise_std": ns,
            "seed": int(seed),
            "reconstruction_error": err,
            "energy_per_imf": [fsum(c[i] ** 2 for i in range(n)) for c in out],
            "method": "Ensemble EMD, Rangayyan & Krishnan (2024) Sec 9.4.1 "
                      "eq (9.13) and steps 1-4; method of Wu & Huang (2009), "
                      "the book's reference [17]",
        }
    )


rangayyan_eemd = emdens  # pre-policy spelling


# -- rgemd: Empirical mode decomposition (EMD) sifting algorithm.
def sift(x, max_imfs=10, tol=0.05):
    """Empirical mode decomposition by the sifting algorithm.

    Why this exists: every other transform in this shelf projects the signal
    onto a basis chosen in advance -- sinusoids, wavelets, Gabor atoms.  If
    the signal does not look like the basis, the representation is diffuse
    and hard to read.  EMD picks its components out of the data itself, so it
    can follow a component whose frequency and amplitude both drift, which is
    what most physiological oscillations actually do.

    Rangayyan & Krishnan (2024) Sec 9.4 describes EMD as "a data-driven
    algorithm that does not use predetermined basis functions" and gives the
    standard algorithm verbatim:

      1. Find the locations of all of the extrema in the given signal x(n).
      2. Interpolate between the minima to obtain the lower signal envelope,
         xmin(n); do the same with the maxima to obtain xmax(n).  The cubic
         spline function is recommended for interpolation.
      3. Compute the mean xm(n) = [xmin(n) + xmax(n)]/2.
      4. Subtract the mean from the signal to obtain s(n) = x(n) - xm(n).
      5. If s(n) meets the criteria for an IMF, define c(n) = s(n) as an IMF;
         otherwise set x(n) = s(n) and repeat from step 1.
      6. To get the remaining IMFs, define the residual as x(n) - c(n) and
         repeat the procedure.

    The IMF criteria, also stated in Sec 9.4, are that "the number of extrema
    and the number of zero-crossings must be equal or differ at most by one"
    and that "at any location, the mean of the envelopes defined by the local
    maxima and the local minima is zero".  The book leaves the numerical
    stopping tolerance to its references; the normalised sum-of-squared-
    differences criterion used here is from Huang, N. E., et al. (1998),
    Proc. R. Soc. Lond. A 454:903-995, eq (5.5).

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    max_imfs : int
        Cap on the number of modes; sifting also stops on its own when the
        residual has too few extrema to define an envelope (the book's
        "no complete oscillation can be identified in the residual").
    tol : float
        Sifting stop tolerance; smaller means more sifting per IMF.

    Returns
    -------
    RichResult with keys ``imfs`` (list of full-length modes, highest
    frequency first), ``residual``, ``n_imfs``, ``reconstruction_error``
    (max |sum(imfs) + residual - x|; EMD is exactly additive by
    construction, so this is a machine-epsilon check), ``energy_per_imf``,
    ``method``.

    Raises
    ------
    ValueError
        If max_imfs < 1 or tol is not positive.
    """
    v = _tfneed(x, "x", 8)
    mi = int(max_imfs)
    if mi < 1:
        raise ValueError("max_imfs must be >= 1")
    t = float(tol)
    if t <= 0.0:
        raise ValueError("tol must be positive")
    imfs, res = _tfemd(v, mi, t)
    n = len(v)
    tot = [fsum([res[i]] + [c[i] for c in imfs]) for i in range(n)]
    err = max(abs(tot[i] - v[i]) for i in range(n))
    return RichResult(
        payload={
            "imfs": imfs,
            "residual": res,
            "n_imfs": len(imfs),
            "reconstruction_error": err,
            "energy_per_imf": [fsum(c[i] ** 2 for i in range(n)) for c in imfs],
            "tol": t,
            "method": "Empirical mode decomposition by sifting, Rangayyan & "
                      "Krishnan (2024) Sec 9.4 algorithm steps 1-6; SD stopping "
                      "rule from Huang et al. (1998) Proc. R. Soc. A 454",
        }
    )


rangayyan_emd = sift  # pre-policy spelling


# -- rgemdimf: Intrinsic mode function (IMF) extraction and validation.
def imf(x, max_iter=50, tol=0.05):
    """Extract one intrinsic mode function and test whether it really is one.

    Why this exists: :func:`sift` returns modes; this returns the one mode
    plus the evidence.  An IMF is only meaningful as an instantaneous
    amplitude/frequency pair if it actually satisfies the two admissibility
    conditions, and sifting can be stopped early or run on a signal with too
    few extrema.  Checking is cheap and the alternative is a Hilbert transform
    of something that has no single well-defined instantaneous frequency.

    Rangayyan & Krishnan (2024) Sec 9.4 states the two properties an IMF is
    required to have:

      - "Over the duration of an IMF, the number of extrema and the number of
        zero-crossings must be equal or differ at most by one."
      - "At any location, the mean of the envelopes defined by the local
        maxima and the local minima is zero."

    Both are computed and reported.  The section also notes that "an IMF is
    not restricted to be a narrowband signal; it could be modulated in both
    amplitude and frequency and also be nonstationary" -- so a wide bandwidth
    is not a failure.  Once admissible, eqs (9.8)-(9.11) give the analytic
    signal ca(n) = c(n) + j cH(n), its amplitude a(n) = sqrt(c^2 + cH^2) and
    its phase theta(n) = arctan(cH(n)/c(n)), which are returned here too.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    max_iter : int
        Maximum sifting iterations.
    tol : float
        Sifting stop tolerance.

    Returns
    -------
    RichResult with keys ``imf``, ``residual``, ``n_extrema``,
    ``n_zero_crossings``, ``extrema_zerox_ok`` (the first condition),
    ``mean_envelope`` (the pointwise envelope mean), ``max_envelope_mean``
    and ``envelope_mean_ok`` (the second condition), ``is_imf``,
    ``iterations``, ``converged``, ``amplitude`` (eq 9.10), ``phase``
    (eq 9.11), ``method``.

    Raises
    ------
    ValueError
        If max_iter < 1, tol is not positive, or the signal has fewer than
        two maxima and two minima (no envelope can be built at all).
    """
    v = _tfneed(x, "x", 8)
    mi = int(max_iter)
    if mi < 1:
        raise ValueError("max_iter must be >= 1")
    t = float(tol)
    if t <= 0.0:
        raise ValueError("tol must be positive")
    mx0, mn0 = _tfextrema(v)
    if len(mx0) < 2 or len(mn0) < 2:
        raise ValueError(
            f"the signal has {len(mx0)} maxima and {len(mn0)} minima; at least "
            f"two of each are needed to build the spline envelopes of Sec 9.4 "
            f"step 2"
        )
    c, iters, conv = _tfsift(v, mi, t)
    n = len(c)
    mx, mn = _tfextrema(c)
    nex = len(mx) + len(mn)
    nzx = _tfzerox(c)
    cond1 = abs(nex - nzx) <= 1
    if len(mx) >= 2 and len(mn) >= 2:
        tt = list(range(n))
        up = _tfspline([0] + mx + [n - 1], [c[0]] + [c[i] for i in mx] + [c[n - 1]], tt)
        lo = _tfspline([0] + mn + [n - 1], [c[0]] + [c[i] for i in mn] + [c[n - 1]], tt)
        menv = [(up[i] + lo[i]) / 2.0 for i in range(n)]
    else:
        menv = [0.0] * n
    amp = max(abs(t2) for t2 in c) or 1.0
    mem = max(abs(t2) for t2 in menv)
    cond2 = mem <= 0.05 * amp
    za = _tfanalytic(c)
    return RichResult(
        payload={
            "imf": c,
            "residual": [v[i] - c[i] for i in range(n)],
            "n_extrema": nex,
            "n_zero_crossings": nzx,
            "extrema_zerox_ok": cond1,
            "mean_envelope": menv,
            "max_envelope_mean": mem,
            "envelope_mean_ok": cond2,
            "is_imf": bool(cond1 and cond2),
            "iterations": iters,
            "converged": conv,
            "amplitude": [abs(t2) for t2 in za],
            "phase": [atan2(t2.imag, t2.real) for t2 in za],
            "method": "IMF extraction and admissibility test, Rangayyan & "
                      "Krishnan (2024) Sec 9.4 (IMF properties) with the "
                      "analytic-signal quantities of eqs (9.8)-(9.11)",
        }
    )


rangayyan_emd_imf = imf  # pre-policy spelling


# -- rgemdtwa: T-wave alternans detection via EMD-based signal decomposition.
def twaemd(ecg, fs=250.0, r_peaks=None, twa_window=(0.15, 0.40), max_imfs=6):
    """Detect T-wave alternans by comparing odd and even beats' T waves.

    Why this exists: T-wave alternans is an ABAB pattern -- the T wave differs
    between consecutive beats, at microvolt scale.  Because the alternation is
    exactly period-2 in beat number, it appears at half the beat rate, which
    is a frequency no per-beat measurement can see.  Comparing the averaged
    odd-beat T wave with the averaged even-beat T wave puts the alternation on
    the measurement axis directly, and averaging is what pulls a microvolt
    signal out of the noise.

    Rangayyan & Krishnan (2024) Sec 9.2.3 "Detection of microvolt T-wave
    alternans in long-term ECG recordings" states that "TWA is a
    heart-rate-dependent anomaly observed in surface ECG in which the
    amplitude and/or shape of the T wave changes every second heart beat",
    that "the TWA signal amplitude is fairly small, typically in the microvolt
    range", and that the standard spectral and modified-moving-average methods
    "do not handle signal nonstationarity", which is why the chapter reaches
    for adaptive decomposition.  EMD is used here to strip the baseline wander -- the
    monotonic sifting residual -- before the beats are aligned, since a
    drifting baseline biases an odd/even comparison directly.  Only the
    residual is removed; whole IMFs are not discarded, because the T wave is
    itself low frequency and would go with them.
    The EMD itself is Sec 9.4 steps 1-6.

    The book does NOT specify an alternans amplitude threshold or an
    R-peak detector, so neither is invented: R peaks must be supplied (or a
    simple amplitude-threshold detector is used and is reported as such), and
    the alternans voltage is returned as a number, not a verdict.

    Parameters
    ----------
    ecg : sequence of float
        ECG samples.
    fs : float
        Sampling rate in Hz.
    r_peaks : sequence of int or None
        R-peak sample indices.  If None, peaks are found by a simple
        threshold-and-refractory rule and ``rpeaks_supplied`` is False.
    twa_window : (float, float)
        Start and end of the T-wave window, in seconds after the R peak.
    max_imfs : int
        Number of IMFs kept for detrending.

    Returns
    -------
    RichResult with keys ``twa_amplitude`` (peak |odd mean - even mean| in
    the T window, in signal units), ``twa_rms``, ``odd_mean``, ``even_mean``,
    ``difference``, ``n_beats``, ``n_odd``, ``n_even``, ``r_peaks``,
    ``rpeaks_supplied``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive, the T window is not increasing, or fewer than
        four beats are available (two of each parity are the minimum for an
        odd/even comparison).
    """
    v = _tfneed(ecg, "ecg", 32)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    t0, t1 = float(twa_window[0]), float(twa_window[1])
    if not 0.0 <= t0 < t1:
        raise ValueError(f"twa_window must satisfy 0 <= start < end, got {twa_window!r}")
    n = len(v)
    # Detrend by discarding the EMD residual and the slowest mode -- the
    # baseline wander that would otherwise masquerade as alternans.
    imfs, res = _tfemd(v, int(max_imfs), 0.05)
    # Detrending = discard the EMD RESIDUAL only.  The residual is the
    # monotonic trend the sifting loop could no longer decompose, i.e. exactly
    # the baseline wander.  Dropping whole IMFs instead would take T-wave
    # content with it, since the T wave is itself low frequency.
    det = [v[i] - res[i] for i in range(n)] if imfs else list(v)
    supplied = r_peaks is not None
    if supplied:
        rp = [int(t) for t in r_peaks]
    else:
        # Detect on the RAW signal: the QRS is the sharpest feature there,
        # and EMD detrending is tuned to preserve the T wave, not the spike.
        mx = max(abs(t) for t in v) or 1.0
        thr = 0.5 * mx
        refr = max(1, int(0.2 * fs))
        rp, i = [], 0
        while i < n:
            if abs(v[i]) >= thr:
                j = i
                while j < n and abs(v[j]) >= thr:
                    j += 1
                rp.append(max(range(i, j), key=lambda q: abs(v[q])))
                i = j + refr
            else:
                i += 1
    rp = [p for p in rp if 0 <= p < n]
    if len(rp) < 4:
        raise ValueError(
            f"only {len(rp)} R peaks available; at least 4 are needed for an "
            f"odd/even T-wave comparison"
        )
    a, b = int(t0 * fs), int(t1 * fs)
    if b <= a:
        raise ValueError("the T-wave window is empty at this sampling rate")
    odd, even = [], []
    for bi, p in enumerate(rp):
        if p + b > n:
            continue
        seg = det[p + a:p + b]
        (even if bi % 2 == 0 else odd).append(seg)
    if len(odd) < 2 or len(even) < 2:
        raise ValueError(
            f"only {len(even)} even and {len(odd)} odd complete T windows; "
            f"at least two of each are needed"
        )
    m = b - a
    om = [fsum(s[i] for s in odd) / len(odd) for i in range(m)]
    em = [fsum(s[i] for s in even) / len(even) for i in range(m)]
    diff = [om[i] - em[i] for i in range(m)]
    return RichResult(
        payload={
            "twa_amplitude": max(abs(t) for t in diff),
            "twa_rms": sqrt(fsum(t * t for t in diff) / m),
            "odd_mean": om,
            "even_mean": em,
            "difference": diff,
            "n_beats": len(rp),
            "n_odd": len(odd),
            "n_even": len(even),
            "r_peaks": rp,
            "rpeaks_supplied": supplied,
            "method": "Odd/even T-wave alternans amplitude after EMD "
                      "detrending; Rangayyan & Krishnan (2024) Sec 9.2.3 (TWA "
                      "definition) and Sec 9.4 (EMD).  The book gives no "
                      "alternans threshold, so none is applied",
        }
    )


rangayyan_emd_twa = twaemd  # pre-policy spelling


# -- rgemdvf: Ventricular fibrillation detection using EMD features.
def vfemd(ecg, fs=250.0, n_imfs=6, tol=0.05):
    """Characterise a fibrillation waveform through its intrinsic mode functions.

    Why this exists: dominant-frequency analysis of ventricular fibrillation
    assumes a stationary driving frequency, and the sources ("rotors") move.
    Decomposing the electrogram into IMFs first gives a set of components each
    of which can be followed in instantaneous frequency as the source
    migrates, which a single dominant-frequency number cannot do.

    Rangayyan & Krishnan (2024) Sec 8.16 "Application: Detection of
    Ventricular Fibrillation in ECG Signals" states that "ventricular
    fibrillation is a nonstationary phenomenon that alters the temporal
    waveform shape, phase, and frequency dynamics of cardiac-surface
    electrograms", that "the majority of current research restricts frequency
    analysis to segmented, time-averaged spectrum analysis, omitting critical
    information on the temporal evolution of spectral features", and that "the
    IMF is more suited than dominant frequency techniques to cope with
    migratory sources and conduction blockages" (its reference [98], Umapathy
    et al.).  The IMFs themselves come from Sec 9.4, with the instantaneous
    amplitude and frequency of eqs (9.8)-(9.11).

    NOTE ON THE PREVIOUS DOCSTRING: this function was documented as "IMF
    energies in 3-10 Hz band elevated in VF; threshold decision".  No such
    band and no such threshold appears in Sec 8.16 or anywhere else in the
    book -- the only band the book gives in this context is the 3-21 Hz
    bandpass of Sec 8.15.  No detection threshold is invented here; the
    per-IMF descriptors the book actually motivates are returned instead.

    Parameters
    ----------
    ecg : sequence of float
        Electrogram / ECG samples.
    fs : float
        Sampling rate in Hz.
    n_imfs : int
        Cap on the number of modes.
    tol : float
        Sifting tolerance.

    Returns
    -------
    RichResult with keys ``imfs``, ``n_imfs``, ``features`` (one dict per
    IMF: ``energy``, ``relative_energy``, ``mean_freq``, ``freq_std``
    -- the temporal spread of the instantaneous frequency, i.e. how much the
    source migrates -- and ``mean_amplitude``), ``dominant_imf``,
    ``dominant_freq``, ``residual``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive or n_imfs < 1.
    """
    v = _tfneed(ecg, "ecg", 16)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    ni = int(n_imfs)
    if ni < 1:
        raise ValueError("n_imfs must be >= 1")
    n = len(v)
    imfs, res = _tfemd(v, ni, float(tol))
    tot = fsum(fsum(c[i] ** 2 for i in range(n)) for c in imfs) or 1.0
    feats = []
    for c in imfs:
        za = _tfanalytic(c)
        a = [abs(t) for t in za]
        ph = [atan2(t.imag, t.real) for t in za]
        fi = []
        for i in range(1, n):
            d = ph[i] - ph[i - 1]
            while d > pi:
                d -= 2.0 * pi
            while d < -pi:
                d += 2.0 * pi
            fi.append(abs(d) * fs / (2.0 * pi))
        mf = fsum(fi) / len(fi) if fi else 0.0
        sd = sqrt(fsum((t - mf) ** 2 for t in fi) / len(fi)) if fi else 0.0
        e = fsum(t * t for t in c)
        feats.append({"energy": e, "relative_energy": e / tot,
                      "mean_freq": mf, "freq_std": sd,
                      "mean_amplitude": fsum(a) / n})
    dom = max(range(len(feats)), key=lambda i: feats[i]["energy"]) if feats else None
    return RichResult(
        payload={
            "imfs": imfs,
            "residual": res,
            "n_imfs": len(imfs),
            "features": feats,
            "dominant_imf": dom,
            "dominant_freq": feats[dom]["mean_freq"] if dom is not None else None,
            "method": "IMF-based characterisation of ventricular fibrillation, "
                      "Rangayyan & Krishnan (2024) Sec 8.16 with the EMD of "
                      "Sec 9.4 and eqs (9.8)-(9.11)",
        }
    )


rangayyan_emd_vf_detect = vfemd  # pre-policy spelling


# -- rgemg: EMG RMS envelope -- Rangayyan & Krishnan Sec 5.6.1, eq (5.24).
def rangayyan_emg_rms(x, window=64, fs=1.0):
    """Sliding-window RMS envelope.

    RMS[n] = sqrt( (1/W) Σ_{k=n-W+1}^{n} x[k]² ).

    Parameters
    ----------
    x : array-like
    window : int
        Window length in samples.
    fs : float
        Sampling rate (Hz, only for reporting).

    Returns
    -------
    RichResult with keys ``rms``, ``window``, ``fs``, ``mean_rms``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.6.1 "The RMS value", pp.283-284.
        The previous docstring cited Ch 8. Equation (5.23) is the global RMS
        over N samples; equation (5.24) is the running RMS this function
        computes,

            RMS(n) = [ (1/M) sum_{k=0}^{M-1} x^2(n-k) ]^(1/2),

        which is explicitly CAUSAL and therefore undefined for n < M-1.
    """
    x = np.asarray(x, dtype=float).ravel()
    W = int(window)
    if W < 1:
        raise ValueError("window must be >= 1")
    sq = x**2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    rms = np.full_like(x, np.nan)
    for i in range(W - 1, x.size):
        rms[i] = np.sqrt((csum[i + 1] - csum[i + 1 - W]) / W)
    # The first W-1 samples stay NaN. Equation (5.24) is a CAUSAL window --
    # RMS(n) averages x(n-k) for k = 0..M-1 -- so it is simply undefined until
    # n = M-1; the book defines no warm-up value.
    #
    # This previously back-filled rms[:W-1] with rms[W-1], a value computed
    # from samples that lie in the FUTURE of those positions. That destroys the
    # one property eq (5.24) exists to have. Measured: a signal that is exactly
    # zero until sample 20 and active thereafter reported envelope 0.7651 at
    # sample 0, i.e. the envelope rose 20 samples BEFORE the burst. EMG onset
    # detection is the main use of an RMS envelope, so the artefact lands
    # exactly where it does the most damage.
    #
    # mean_rms already uses np.nanmean, so the warm-up was always meant to be
    # NaN; the back-fill was the anomaly.
    res = RichResult(
        title="EMG RMS envelope",
        summary_lines=[
            ("Window (samples)", W),
            ("Fs (Hz)", float(fs)),
            ("Mean RMS", float(np.nanmean(rms))),
            ("Max RMS", float(np.nanmax(rms))),
        ],
        interpretation=f"Sliding-window RMS, W={W} samples ({W / fs:.3g} s).",
        payload={"rms": rms, "window": W, "fs": float(fs), "mean_rms": float(np.nanmean(rms))},
    )
    return with_describe_pointer(res, "rgemg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_emg_rms(rng.standard_normal(500), window=32)
# >>> r["rms"].shape == (500,)
# True


# -- rgentrwv: Wavelet entropy for measuring signal regularity.
def wtentropy(x, wavelet="db4", levels=3, base="e"):
    """Wavelet entropy: how evenly the signal's energy is spread over scales.

    Why this exists: subband energies alone say WHERE the energy is; entropy
    says how ORDERED the signal is.  A single clean rhythm concentrates its
    energy in one band and scores low; broadband noise or a disorganised
    waveform spreads across all bands and scores near the maximum.  That
    single scalar is what makes it usable as a state marker, e.g. for depth
    of anaesthesia or for seizure onset.

    The construction is the Shannon entropy of the relative wavelet energies

        p_j = E_j / sum_k E_k,   WE = -sum_j p_j log p_j,

    where E_j are the subband energies.  Rangayyan & Krishnan (2024) supplies
    the subband energy decomposition (Sec 8.15, Ex = Es1 + ... + EsN, over the
    eq 8.111-8.113 DWT) but does NOT define wavelet entropy.  The measure is
    from
      Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A.,
      Schurmann, M., & Basar, E. (2001). "Wavelet entropy: a new tool for
      analysis of short duration brain electrical signals." Journal of
      Neuroscience Methods 105(1):65-75.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Decomposition depth.  The maximum possible entropy is log(levels + 1).
    base : str
        ``'e'`` for nats or ``'2'`` for bits.

    Returns
    -------
    RichResult with keys ``entropy``, ``normalized_entropy`` (0 = all energy
    in one band, 1 = uniform), ``max_entropy``, ``relative_energy``,
    ``labels``, ``levels``, ``base``, ``method``.

    Raises
    ------
    ValueError
        If base is not 'e' or '2', or the signal carries no energy.
    """
    b = str(base).strip().lower()
    if b not in ("e", "2"):
        raise ValueError("base must be 'e' or '2'")
    r = wtenergy(x, wavelet=wavelet, levels=levels)
    p = r["relative"]
    if r["total_energy"] <= 0.0:
        raise ValueError("the signal has zero energy; wavelet entropy is undefined")
    ent = -fsum(q * log(q) for q in p if q > 0.0)
    mx = log(len(p))
    if b == "2":
        ent /= log(2.0)
        mx /= log(2.0)
    return RichResult(
        payload={
            "entropy": ent,
            "max_entropy": mx,
            "normalized_entropy": ent / mx if mx > 0.0 else 0.0,
            "relative_energy": p,
            "labels": r["labels"],
            "levels": int(levels),
            "base": b,
            "wavelet": str(wavelet),
            "method": "Wavelet (relative-energy Shannon) entropy, Rosso et al. "
                      "(2001) J. Neurosci. Methods 105(1):65-75, over the "
                      "Rangayyan & Krishnan (2024) eq (8.111)-(8.113) DWT",
        }
    )


rangayyan_wavelet_entropy = wtentropy  # pre-policy spelling


# -- rgenv: Hilbert-transform envelope -- Rangayyan & Krishnan Sec 5.5.3.
def rangayyan_envelope(x):
    """Analytic-signal envelope via the Hilbert transform.

    ``env(t) = |x(t) + j H{x(t)}|`` where H{·} is the discrete Hilbert
    transform (``scipy.signal.hilbert``).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    RichResult with keys ``envelope``, ``analytic``,
    ``instantaneous_phase``, ``instantaneous_freq``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.5.3 "The envelogram", p.281
        (Sec 5.5 "Envelope Extraction and Analysis", p.277).
    """
    from ._signal_core import hilbert

    x = np.asarray(x, dtype=float)
    z = hilbert(x)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase) / (2 * np.pi)
    res = RichResult(
        title="Hilbert envelope",
        summary_lines=[
            ("N samples", int(x.size)),
            ("Envelope mean", float(env.mean())),
            ("Envelope max", float(env.max())),
        ],
        interpretation=f"Analytic envelope mean {env.mean():.4g}, peak {env.max():.4g}.",
        payload={"envelope": env, "analytic": z, "instantaneous_phase": phase, "instantaneous_freq": inst_freq},
    )
    return with_describe_pointer(res, "rgenv")


# CANONICAL TEST
# >>> t = np.arange(100)/100.0
# >>> x = np.cos(2*np.pi*5*t) * (1 + 0.3*np.cos(2*np.pi*0.5*t))
# >>> r = rangayyan_envelope(x)
# >>> r["envelope"].shape == x.shape
# True


# -- rgenvgm: Envelogram.
def rangayyan_envelogram(pcg, ecg=None, fs=1000.0, r_peaks=None, n_beats=None):
    r"""Envelogram of a PCG signal (Rangayyan Ch. 3):

    .. math:: \mathrm{env}_{avg}[n] = \frac1M \sum_{k=1}^{M}
              \big| x_k(n) + j\,\mathcal H\{x_k(n)\} \big|,

    the ensemble-averaged analytic-signal magnitude. The Hilbert
    transform gives the instantaneous amplitude envelope, which is
    what makes S1 and S2 visible as smooth bumps rather than as
    oscillation. Alignment comes from the ECG R peaks -- averaging
    unaligned beats smears the envelope and is the usual way this
    goes wrong, so the R peaks are required rather than guessed.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram.
    ecg : array-like, optional
        ECG used only if r_peaks must be detected.
    fs : float, default 1000.0
        Sampling frequency.
    r_peaks : array-like of int, optional
        R-peak sample indices; detected from ecg when omitted.
    n_beats : int, optional
        Beat-count check.

    Returns
    -------
    RichResult
        keys: ``envelope`` (averaged), ``beats`` (M, L matrix),
        ``M``, ``beat_length``, ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (envelope extraction, envelograms).
    """
    from ._signal_core import signal as sig

    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if r_peaks is None:
        if ecg is None:
            raise ValueError(
                "supply r_peaks, or an ecg to detect them from -- averaging "
                "unaligned beats smears the envelope."
            )
        e = np.asarray(ecg, dtype=float).ravel()
        thr = float(np.mean(e) + 2.0 * np.std(e))
        pk, _ = sig.find_peaks(e, height=thr, distance=int(0.25 * fs))
        r_peaks = pk
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 2:
        raise ValueError("need at least 2 R peaks to segment beats.")
    if n_beats is not None and int(n_beats) != r.size:
        raise ValueError(f"n_beats = {n_beats} does not match {r.size} peaks.")
    L = int(np.min(np.diff(r)))
    if L < 8:
        raise ValueError("beats are too short to average.")
    env = np.abs(sig.hilbert(x))
    beats = np.array([env[p : p + L] for p in r[:-1] if p + L <= env.size])
    if beats.size == 0:
        raise ValueError("no complete beats within the signal.")
    return RichResult(payload={"envelope": beats.mean(axis=0), "beats": beats,
                               "M": int(beats.shape[0]), "beat_length": L, "fs": fs,
                               "method": "Hilbert envelope averaged over R-aligned beats"})


# -- rghaar: Haar wavelet transform (simplest orthogonal wavelet).
def dwt2tap(x, levels=3):
    """Two-tap orthogonal wavelet transform -- running sums and differences.

    Why this exists: this is the shortest orthogonal wavelet there is.  Its
    filters are (1, 1)/sqrt(2) and (1, -1)/sqrt(2), so a level of the
    transform is literally a normalised average and difference of adjacent
    samples.  That makes it the one wavelet whose coefficients can be read by
    hand, which is why it is the right tool for checking that a wavelet
    pipeline does what you think, and for detecting step discontinuities,
    where its discontinuous shape is an advantage rather than a defect.

    It is the order-1 member of the Daubechies family (see :func:`orthfilt`).
    Rangayyan & Krishnan (2024) Sec 8.8 gives the general dyadic DWT in
    eqs (8.111)-(8.113) but does NOT single this wavelet out or give its
    taps; the two-tap filter is the L = 2 case of
      Daubechies, I. (1992). Ten Lectures on Wavelets. SIAM, Table 6.1.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    levels : int
        Number of decomposition levels.

    Returns
    -------
    RichResult with keys ``approx``, ``details``, ``coeffs``, ``lengths``,
    ``levels``, ``energy``, ``input_energy``, ``method``.  Because the
    transform is orthonormal, ``energy`` equals ``input_energy`` for a
    signal of even length at every level -- a Parseval check the caller can
    read straight off the payload.

    Raises
    ------
    ValueError
        If levels exceeds what the signal length allows.
    """
    v = _tfneed(x, "x", 2)
    lv = int(levels)
    a, d, ln = _tfdwt(v, "db1", lv)
    coeffs = [list(a)] + [list(c) for c in d]
    return RichResult(
        payload={
            "approx": list(a),
            "details": [list(c) for c in d],
            "coeffs": coeffs,
            "lengths": ln,
            "levels": lv,
            "energy": fsum(fsum(t * t for t in c) for c in coeffs),
            "input_energy": fsum(t * t for t in v),
            "method": "Two-tap (Haar / db1) orthogonal DWT; the L=2 case of "
                      "Daubechies (1992) Table 6.1, dyadic grid per "
                      "Rangayyan & Krishnan (2024) eq (8.113)",
        }
    )


rangayyan_haar_wavelet = dwt2tap  # pre-policy spelling


# -- rghhtsp: Hilbert-Huang spectrum (HHS) via EMD + Hilbert transform.
def emdspec(x, fs=1.0, max_imfs=8, nfreq=32, tol=0.05):
    """Time-frequency spectrum built from EMD modes and their instantaneous
    frequencies.

    Why this exists: a Fourier or wavelet TFD spreads a frequency-modulated
    component over many bins, because the basis functions have fixed
    frequency and the component does not.  Once EMD has separated the signal
    into modes that each carry one oscillation at a time, each mode has a
    single meaningful instantaneous frequency, and the energy can be placed at
    exactly that frequency at every instant.  The result is a TFD with no
    uncertainty-principle smearing -- its resolution is limited by the
    decomposition, not by a window.

    Rangayyan & Krishnan (2024) Sec 9.4 gives the construction.  For an IMF
    c(n), eq (9.8) forms the analytic signal ca(n) = c(n) + j cH(n) with
    cH the Hilbert transform; eq (9.9) writes it as ca(n) = a(n) exp[j
    theta(n)]; eq (9.10) gives a(n) = sqrt(c^2(n) + cH^2(n)); eq (9.11) gives
    theta(n) = arctan(cH(n)/c(n)); and the text states that "the
    instantaneous frequency omega(n) is given by the derivative of theta(n)
    with respect to time".  Eq (9.12) expresses the original signal as
    x(n) = real{ sum_i a_i(n) exp[j theta_i(n)] }, noting that "the residual
    component is neglected".

    Energy a_i^2(n) is deposited into the bin containing omega_i(n) at each
    instant -- the sparse, non-smeared TFD that eq (9.12) implies.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    max_imfs : int
        Cap on the number of modes.
    nfreq : int
        Number of frequency bins over [0, fs/2).
    tol : float
        EMD sifting tolerance.

    Returns
    -------
    RichResult with keys ``spectrum`` (time x frequency), ``times``,
    ``freqs``, ``imfs``, ``amplitude`` (a_i(n) per IMF, eq 9.10),
    ``inst_freq`` (omega_i(n)/2pi in Hz per IMF), ``marginal`` (energy summed
    over time per frequency bin), ``n_imfs``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive or nfreq < 2.
    """
    v = _tfneed(x, "x", 8)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    nf = int(nfreq)
    if nf < 2:
        raise ValueError("nfreq must be >= 2")
    n = len(v)
    imfs, _res = _tfemd(v, int(max_imfs), float(tol))
    freqs = [k * fs / (2.0 * nf) for k in range(nf)]
    df = fs / (2.0 * nf)
    spec = [[0.0] * nf for _ in range(n)]
    amps, ifs = [], []
    for c in imfs:
        za = _tfanalytic(c)
        a = [abs(t) for t in za]                       # eq (9.10)
        ph = [atan2(t.imag, t.real) for t in za]       # eq (9.11)
        # omega(n) = d theta / dt, by centred differences on the unwrapped
        # phase.  Wrapping is undone modulo 2 pi before differencing.
        unw = [ph[0]]
        for i in range(1, n):
            d = ph[i] - ph[i - 1]
            while d > pi:
                d -= 2.0 * pi
            while d < -pi:
                d += 2.0 * pi
            unw.append(unw[-1] + d)
        fi = []
        for i in range(n):
            if i == 0:
                dth = unw[1] - unw[0]
            elif i == n - 1:
                dth = unw[n - 1] - unw[n - 2]
            else:
                dth = (unw[i + 1] - unw[i - 1]) / 2.0
            fi.append(abs(dth) * fs / (2.0 * pi))
        amps.append(a)
        ifs.append(fi)
        for i in range(n):
            k = int(fi[i] / df)
            if 0 <= k < nf:
                spec[i][k] += a[i] * a[i]
    marg = [fsum(row[k] for row in spec) for k in range(nf)]
    return RichResult(
        payload={
            "spectrum": spec,
            "times": [i / fs for i in range(n)],
            "freqs": freqs,
            "imfs": imfs,
            "amplitude": amps,
            "inst_freq": ifs,
            "marginal": marg,
            "n_imfs": len(imfs),
            "peak_freq": freqs[max(range(nf), key=lambda k: marg[k])] if nf else 0.0,
            "method": "EMD-based instantaneous-frequency spectrum, Rangayyan & "
                      "Krishnan (2024) Sec 9.4 eqs (9.8)-(9.12)",
        }
    )


rangayyan_hht_spectrum = emdspec  # pre-policy spelling


# -- rghrvtv: Time-varying HRV analysis via STFT of RR intervals.
def hrvtv(rr_intervals, fs_resamp=4.0, window_len=64, noverlap=None, standard="taskforce"):
    """Time-varying HRV band powers from a short-time spectrum of RR intervals.

    Why this exists: RR interval series recorded over hours are not
    stationary -- the whole clinical point is to catch the transient, e.g. the
    onset of an ischemic episode.  A single Fourier PSD over the whole
    recording averages the transient away.  Tracking LF and HF power in short
    windows keeps the time axis, which is what makes the sympathovagal balance
    a trend rather than a number.

    Rangayyan & Krishnan (2024) Sec 8.12 "Application: Time-varying Analysis
    of HRV" makes the case -- "when heart rate data such as beat-to-beat RR
    intervals are collected over long periods of time (several hours), the
    signal could be expected to be nonstationary" -- and reports the work of
    its reference [31] (Bianchi et al.), who derived a time-varying PSD "in
    order to study transient episodes related to ischemic attacks".  It gives
    both band definitions:

      Bianchi et al.: VLF 0-0.03 Hz, LF 0.03-0.15 Hz, HF 0.18-0.4 Hz.
      Task Force [84]: VLF <= 0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz,
        which the book gives explicitly as the published standard.

    Both are selectable; the Task Force bands are the default because the
    book flags them as the standard.  The RR series is irregularly sampled by
    construction (one value per beat), so it is resampled onto a uniform grid
    before the short-time transform -- that step is a prerequisite the book
    does not spell out but which the Fourier analysis requires.

    Parameters
    ----------
    rr_intervals : sequence of float
        Successive RR intervals in seconds.
    fs_resamp : float
        Uniform resampling rate in Hz.  4 Hz is ample for a 0.4 Hz band.
    window_len : int
        Short-time window length in resampled samples.
    noverlap : int or None
        Overlap in samples.  Default window_len // 2.
    standard : str
        ``'taskforce'`` or ``'bianchi'``.

    Returns
    -------
    RichResult with keys ``times``, ``lf`` , ``hf``, ``vlf``,
    ``lf_hf_ratio``, ``lf_percent``, ``hf_percent``, ``total_power``,
    ``bands``, ``mean_rr``, ``mean_hr``, ``resampled``, ``method``.

    Raises
    ------
    ValueError
        If any RR interval is non-positive, fewer than 4 intervals are given,
        the resampled series is shorter than the window, or the standard is
        unknown.
    """
    rr = _tfneed(rr_intervals, "rr_intervals", 4)
    for t in rr:
        if t <= 0.0:
            raise ValueError("all RR intervals must be positive (seconds)")
    st = str(standard).strip().lower()
    if st == "taskforce":
        bands = {"vlf": (0.0, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.40)}
    elif st == "bianchi":
        bands = {"vlf": (0.0, 0.03), "lf": (0.03, 0.15), "hf": (0.18, 0.40)}
    else:
        raise ValueError(f"standard must be 'taskforce' or 'bianchi', got {standard!r}")
    fsr = float(fs_resamp)
    if fsr <= 0.0:
        raise ValueError("fs_resamp must be positive")
    # Beat times are the cumulative RR sums; interpolate the tachogram onto a
    # uniform grid (linear -- the tachogram has no meaningful curvature
    # between beats).
    tt = [0.0]
    for t in rr:
        tt.append(tt[-1] + t)
    beat_t = tt[1:]
    dur = beat_t[-1]
    m = int(dur * fsr)
    if m < 4:
        raise ValueError("the RR series is too short to resample at this rate")
    grid = [i / fsr for i in range(m)]
    resamp = []
    j = 0
    for g in grid:
        while j < len(beat_t) - 2 and beat_t[j + 1] < g:
            j += 1
        t0, t1 = beat_t[j], beat_t[j + 1]
        y0, y1 = rr[j], rr[j + 1]
        w = 0.0 if t1 == t0 else (g - t0) / (t1 - t0)
        w = max(0.0, min(1.0, w))
        resamp.append(y0 + (y1 - y0) * w)
    mu = fsum(resamp) / m
    resamp = [t - mu for t in resamp]
    wl = int(window_len)
    if wl < 4:
        raise ValueError("window_len must be >= 4")
    if wl > m:
        raise ValueError(
            f"window_len={wl} exceeds the resampled length {m}; use a shorter "
            f"window or a higher fs_resamp"
        )
    sp = spectrogram(resamp, fs=fsr, nperseg=wl, noverlap=noverlap, window="hann")
    fr = sp["freqs"]
    out = {"vlf": [], "lf": [], "hf": []}
    tot = []
    for row in sp["spectrogram"]:
        s = {}
        for name, (a, b) in bands.items():
            s[name] = fsum(row[k] for k in range(len(fr)) if a <= fr[k] < b)
            out[name].append(s[name])
        tot.append(fsum(s.values()))
    return RichResult(
        payload={
            "times": sp["times"],
            "vlf": out["vlf"],
            "lf": out["lf"],
            "hf": out["hf"],
            "total_power": tot,
            "lf_hf_ratio": [out["lf"][i] / out["hf"][i] if out["hf"][i] > 0.0 else float("inf")
                            for i in range(len(tot))],
            "lf_percent": [100.0 * out["lf"][i] / tot[i] if tot[i] > 0.0 else 0.0
                           for i in range(len(tot))],
            "hf_percent": [100.0 * out["hf"][i] / tot[i] if tot[i] > 0.0 else 0.0
                           for i in range(len(tot))],
            "bands": bands,
            "standard": st,
            "mean_rr": fsum(rr) / len(rr),
            "mean_hr": 60.0 / (fsum(rr) / len(rr)),
            "resampled": resamp,
            "fs_resamp": fsr,
            "method": "Time-varying HRV band powers from the short-time "
                      "spectrum of the RR tachogram, Rangayyan & Krishnan "
                      "(2024) Sec 8.12; bands as given there (Task Force "
                      "standard or Bianchi et al.); STFT per eq (8.8)",
        }
    )


rangayyan_hrv_time_varying = hrvtv  # pre-policy spelling


# -- rgistft: Inverse STFT signal reconstruction from spectrogram.
def istft(stft, window="hann", hop=None):
    """Reconstruct a signal from its complex STFT by weighted overlap-add.

    Why this exists: the STFT of eq (8.8) is deliberately redundant -- with
    overlapping windows every sample is analysed several times.  That
    redundancy is what lets a signal be edited in the time-frequency plane
    (denoised, a band removed) and then put back together.  The synthesis
    used here divides the overlap-added frames by the overlap-added squared
    window, which is exact for any window and hop with no zeros in the
    denominator; it therefore round-trips an unmodified STFT to machine
    precision.

    Rangayyan & Krishnan (2024) Sec 8.7 defines the forward STFT (eq 8.8) and
    its continuous form (eq 8.9) but does NOT give an inversion formula.  The
    weighted-overlap-add synthesis implemented here is from
      Griffin, D. W., & Lim, J. S. (1984). "Signal estimation from modified
      short-time Fourier transform." IEEE Trans. ASSP 32(2):236-243, eq (6),
    which is the standard least-squares inverse of eq (8.8).

    Parameters
    ----------
    stft : sequence of sequences of complex
        Frames as produced by :func:`spectrogram` (payload key ``stft``).
    window : str or sequence
        The analysis window used, by name or as explicit taps.
    hop : int or None
        Frame advance in samples.  Default nperseg // 2.

    Returns
    -------
    RichResult with keys ``signal``, ``n``, ``hop``, ``valid_start``,
    ``valid_end``, ``n_frames``, ``method``.  Samples outside
    [valid_start, valid_end] carry no window weight and are returned as 0.

    Raises
    ------
    ValueError
        If the frames are ragged, or if the hop leaves gaps the window
        cannot cover (a zero in the overlap-added squared window).
    """
    frames = [list(f) for f in stft]
    if not frames:
        raise ValueError("stft must contain at least one frame")
    m = len(frames[0])
    if m < 2:
        raise ValueError("each STFT frame must have at least 2 bins")
    for f in frames:
        if len(f) != m:
            raise ValueError("all STFT frames must have the same length")
    h = m // 2 if hop is None else int(hop)
    if not 1 <= h <= m:
        raise ValueError(f"hop must satisfy 1 <= hop <= {m}, got {h}")
    if isinstance(window, str):
        w = _tfwin(window, m)
    else:
        w = aslist(window)
        if len(w) != m:
            raise ValueError("explicit window length must equal the frame length")
    n = (len(frames) - 1) * h + m
    num = [0.0] * n
    den = [0.0] * n
    for fi, F in enumerate(frames):
        seg = _tfidft([complex(c) for c in F])
        off = fi * h
        for i in range(m):
            num[off + i] += seg[i].real * w[i]
            den[off + i] += w[i] * w[i]
    # Division by the overlap-added squared window makes the synthesis exact
    # wherever the windows actually cover the sample -- no COLA condition on
    # the window/hop pair is needed.  A tapered window is zero at its ends,
    # so the first and last few samples of the reconstruction are covered by
    # a vanishing weight; those are reported as invalid rather than divided
    # by ~0.  A hop that leaves a hole in the INTERIOR is a real error and
    # raises.
    tol = 1e-10 * max(den) if max(den) > 0.0 else 0.0
    for i in range(m - 1, n - m + 1):
        if den[i] <= tol:
            raise ValueError(
                f"sample {i} lies in a gap between analysis windows "
                f"(hop={h} is too large for a window of length {m})"
            )
    out = [(num[i] / den[i]) if den[i] > tol else 0.0 for i in range(n)]
    lo = next((i for i in range(n) if den[i] > tol), 0)
    hi = next((i for i in range(n - 1, -1, -1) if den[i] > tol), n - 1)
    return RichResult(
        payload={
            "signal": out,
            "n": n,
            "hop": h,
            "valid_start": lo,
            "valid_end": hi,
            "n_frames": len(frames),
            "method": "Weighted overlap-add inverse STFT, Griffin & Lim (1984) "
                      "eq (6); forward transform is Rangayyan eq (8.8)",
        }
    )


rangayyan_istft = istft  # pre-policy spelling


# -- rgmra: Multiresolution analysis (MRA) decomposition.
def mra(x, wavelet="db4", levels=3):
    """Multiresolution analysis: split a signal into additive detail bands.

    Why this exists: :func:`dwt` returns coefficients, which live at
    different sampling rates and cannot be plotted against the original time
    axis or added together.  MRA reconstructs each subband separately back
    into the signal domain, so the output is a set of full-length components
    that literally sum back to the input.  That is the form in which a
    clinician looks at "the 4-8 Hz part of this EEG".

    Rangayyan & Krishnan (2024) Sec 8.8 gives the analysis in eq (8.111) with
    the dyadic grid of eq (8.113), and eq (8.114) gives the synthesis

        x~(t) = sum_m sum_n X~(m, n) psi_{m,n}(t),

    with the note that "the range of summation depends upon the choices made
    and the application".  Restricting that double sum to a single m is
    exactly the detail band D_m returned here.  The multiresolution framework
    itself is the book's reference [72], Mallat, S. G. (1989), "A theory for
    multiresolution signal decomposition: the wavelet representation", IEEE
    Trans. PAMI 11(7):674-693.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Number of resolution levels.

    Returns
    -------
    RichResult with keys ``approximation`` (the coarsest band A_J, full
    length), ``details`` (D_J .. D_1, coarse to fine, each full length),
    ``bands`` (approximation followed by details), ``reconstruction_error``
    (max |sum of bands - x|, which is a machine-epsilon check that the
    decomposition is complete), ``energy_per_band``, ``levels``, ``method``.

    Raises
    ------
    ValueError
        If levels exceeds what the signal length and filter allow.
    """
    v = _tfneed(x, "x", 2)
    lv = int(levels)
    a, d, ln = _tfdwt(v, wavelet, lv)
    n = len(v)

    def rebuild(which):
        aa = [0.0] * len(a)
        dd = [[0.0] * len(c) for c in d]
        if which is None:
            aa = list(a)
        else:
            dd[which] = list(d[which])
        return _tfidwt(aa, dd, ln, wavelet)[:n]

    approx = rebuild(None)
    details = [rebuild(i) for i in range(len(d))]
    total = [approx[i] + fsum(b[i] for b in details) for i in range(n)]
    err = max(abs(total[i] - v[i]) for i in range(n))
    return RichResult(
        payload={
            "approximation": approx,
            "details": details,
            "bands": [approx] + details,
            "reconstruction_error": err,
            "energy_per_band": [fsum(t * t for t in b) for b in [approx] + details],
            "levels": lv,
            "wavelet": str(wavelet),
            "method": "Multiresolution analysis, Rangayyan & Krishnan (2024) "
                      "eqs (8.111)-(8.114); Mallat (1989) IEEE PAMI 11(7)",
        }
    )


rangayyan_mra = mra  # pre-policy spelling


# -- rgpcgenl: Synchronized averaging of PCG envelopes (S1/S2 intensity analysis).
def pcgenvavg(pcg, ecg, fs=1000.0, cycle_len=None, envelope_smoothing=None):
    """Synchronised averaging of PCG envelopes, triggered from the ECG.

    Why this exists: a single heart sound is buried in noise and varies beat
    to beat, so S1/S2 intensity cannot be measured reliably from one cycle.
    Averaging many cycles suppresses the noise -- but only if the cycles are
    aligned, and the PCG itself has no reliable fiducial point.  The ECG does:
    the QRS complex is the sharpest event in either signal, so it supplies the
    trigger, and the PCG is averaged relative to it.  Averaging the ENVELOPE
    rather than the raw PCG matters because the acoustic phase is not
    reproducible from beat to beat; averaging raw waveforms would cancel the
    sounds it is meant to measure.

    Rangayyan & Krishnan (2024) Sec 3.5 gives synchronised averaging, stating
    that "the most important requirement in synchronized averaging" is the
    synchronisation itself, and that "if the noise is random with zero mean",
    averaging improves the SNR.  Sec 5.5 gives envelope extraction, with the
    complex-demodulation and Hilbert routes of Sec 5.5.1 and 5.5.3; the
    envelope used here is the analytic-signal magnitude of Sec 5.5.3.  Sec 4.9
    and Sec 5.5.2 discuss the use of the ECG (and carotid pulse) to segment
    and align the PCG.

    Parameters
    ----------
    pcg : sequence of float
        PCG samples.
    ecg : sequence of float
        Simultaneously recorded ECG samples, same length and rate.
    fs : float
        Sampling rate in Hz.
    cycle_len : int or None
        Length of the averaged cycle in samples.  Default is the median
        inter-trigger interval.
    envelope_smoothing : int or None
        Moving-average length applied to the envelope, in samples.  Default
        is about 20 ms, which suppresses the acoustic carrier without
        blurring S1 and S2 into each other.

    Returns
    -------
    RichResult with keys ``average_envelope``, ``n_cycles``, ``cycle_len``,
    ``triggers``, ``s1_index``, ``s1_amplitude``, ``s2_index``,
    ``s2_amplitude``, ``s2_s1_ratio``, ``snr_gain_db`` (the 10 log10(N)
    improvement synchronised averaging of N cycles gives against zero-mean
    noise), ``method``.

    Raises
    ------
    ValueError
        If the two signals differ in length, fs is not positive, or fewer
        than two complete cycles can be extracted.
    """
    p = _tfneed(pcg, "pcg", 16)
    e = _tfneed(ecg, "ecg", 16)
    if len(p) != len(e):
        raise ValueError(f"pcg and ecg must have the same length, got {len(p)} and {len(e)}")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    n = len(p)
    # Envelope of the PCG: analytic-signal magnitude (Sec 5.5.3), then a
    # short moving average to remove the acoustic carrier.
    env = [abs(t) for t in _tfanalytic(p)]
    w = max(1, int(0.02 * fs)) if envelope_smoothing is None else int(envelope_smoothing)
    if w < 1:
        raise ValueError("envelope_smoothing must be >= 1")
    sm = []
    for i in range(n):
        a = max(0, i - w // 2)
        b = min(n, i + w // 2 + 1)
        sm.append(fsum(env[a:b]) / (b - a))
    # QRS triggers: threshold the ECG with a refractory period.
    mx = max(abs(t) for t in e) or 1.0
    thr = 0.6 * mx
    refr = max(1, int(0.25 * fs))
    trig, i = [], 0
    while i < n:
        if abs(e[i]) >= thr:
            j = i
            while j < n and abs(e[j]) >= thr:
                j += 1
            trig.append(max(range(i, j), key=lambda q: abs(e[q])))
            i = j + refr
        else:
            i += 1
    if len(trig) < 2:
        raise ValueError(
            f"only {len(trig)} QRS triggers found in the ECG; synchronised "
            f"averaging needs at least two cycles"
        )
    gaps = sorted(trig[i + 1] - trig[i] for i in range(len(trig) - 1))
    cl = gaps[len(gaps) // 2] if cycle_len is None else int(cycle_len)
    if cl < 2:
        raise ValueError("cycle_len must be >= 2 samples")
    cycles = [sm[t:t + cl] for t in trig if t + cl <= n]
    if len(cycles) < 2:
        raise ValueError(
            f"only {len(cycles)} complete cycle(s) of {cl} samples fit in the "
            f"record; at least two are needed"
        )
    avg = [fsum(c[i] for c in cycles) / len(cycles) for i in range(cl)]
    # S1 is the first envelope peak after the trigger, S2 the largest peak in
    # the remainder of the cycle.
    half = max(1, cl // 3)
    s1 = max(range(half), key=lambda i: avg[i])
    s2 = max(range(half, cl), key=lambda i: avg[i]) if cl > half else s1
    return RichResult(
        payload={
            "average_envelope": avg,
            "n_cycles": len(cycles),
            "cycle_len": cl,
            "triggers": trig,
            "s1_index": s1,
            "s1_time": s1 / fs,
            "s1_amplitude": avg[s1],
            "s2_index": s2,
            "s2_time": s2 / fs,
            "s2_amplitude": avg[s2],
            "s2_s1_ratio": avg[s2] / avg[s1] if avg[s1] > 0.0 else float("inf"),
            "snr_gain_db": 10.0 * log(len(cycles)) / log(10.0),
            "method": "ECG-triggered synchronised averaging of PCG envelopes, "
                      "Rangayyan & Krishnan (2024) Sec 3.5 (synchronised "
                      "averaging) with the analytic-signal envelope of "
                      "Sec 5.5.3",
        }
    )


rangayyan_pcg_envelope_avg = pcgenvavg  # pre-policy spelling


# -- rgppgwt: Wavelet denoising of PPG signals.
def ppgwtden(ppg, fs=100.0, wavelet="db4", levels=4, threshold_type="soft"):
    """Wavelet denoising of a PPG signal against motion artifact.

    Why this exists: the classical fixes do not apply here.  A fixed
    band-pass filter cannot work because, as the book puts it, the artifact
    and the signal overlap in both time and frequency; adaptive filtering
    cannot work because it needs a clean reference channel that a wearable
    does not have.  Wavelet shrinkage needs neither: it separates on
    coefficient magnitude within each scale, not on band.

    Rangayyan & Krishnan (2024) Sec 8.14 "Application: Wavelet Denoising of
    PPG Signals" makes exactly that argument -- "given the temporal and
    spectral overlap of the signal of interest and the motion artifact signal,
    fixed filtering approaches will not be optimal.  The adaptive filtering
    technique needs a reference channel consisting of a clean version of the
    signal, which would be difficult to obtain.  A possible alternative would
    be to use a wavelet-based denoising technique."  The section reports the
    work of its reference [91] (Raghuram et al.), in which "five different
    types of wavelets: Daubechies, biorthogonal, reverse biorthogonal, symlet,
    and Coiflet were applied for denoising, and it was found that the
    Daubechies wavelets provided the best denoising result" -- hence the
    Daubechies default.  The shrinkage rules are eqs (8.103) and (8.104) and
    the reconstruction is eq (8.105).  The section further notes that the
    respiratory activity band of the PPG sits at low frequency and can be
    damaged by denoising, so the approximation band is reported separately
    and is never thresholded.

    Parameters
    ----------
    ppg : sequence of float
        PPG samples.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Default ``'db4'``, per the Daubechies finding in Sec 8.14.
    levels : int
        Decomposition depth.
    threshold_type : str
        ``'soft'`` (eq 8.104) or ``'hard'`` (eq 8.103).

    Returns
    -------
    RichResult with keys ``denoised``, ``artifact`` (what was removed),
    ``threshold``, ``sigma``, ``snr_improvement_db`` (10 log10 of the ratio
    of input energy to removed-component energy -- an artifact-level
    indicator, not a true SNR, since no clean reference exists),
    ``approx_energy`` (the untouched low-frequency band that carries the
    respiratory component), ``levels``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive, or from :func:`wtthresh` for a bad
        threshold_type or too many levels.
    """
    v = _tfneed(ppg, "ppg", 8)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    r = wtthresh(v, wavelet=wavelet, levels=int(levels),
                 threshold_type=threshold_type)
    den = r["denoised"]
    art = [v[i] - den[i] for i in range(len(v))]
    ein = fsum(t * t for t in v)
    ea = fsum(t * t for t in art)
    a, _d, _ln = _tfdwt(v, wavelet, int(levels))
    return RichResult(
        payload={
            "denoised": den,
            "artifact": art,
            "threshold": r["threshold"],
            "sigma": r["sigma"],
            "artifact_energy": ea,
            "snr_improvement_db": 10.0 * log(ein / ea) / log(10.0) if ea > 0.0 else float("inf"),
            "approx_energy": fsum(t * t for t in a),
            "levels": int(levels),
            "wavelet": str(wavelet),
            "fs": fs,
            "method": "Wavelet-shrinkage denoising of PPG, Rangayyan & Krishnan "
                      "(2024) Sec 8.14 (Daubechies wavelets best, per its "
                      "reference [91]) with eqs (8.103)-(8.105)",
        }
    )


rangayyan_ppg_wavelet = ppgwtden  # pre-policy spelling


# -- rgsclgr: Scalogram: energy density via squared CWT magnitudes.
def scalogram(x, fs=1.0, scales=None, wavelet="morlet", w0=5.0):
    """Scalogram: the squared-magnitude CWT, i.e. energy density in time-scale.

    Why this exists: the CWT coefficients are signed (and complex for a
    Morlet), so they are not directly readable as "how much signal is here".
    Squaring gives a nonnegative energy surface -- the wavelet analogue of the
    spectrogram -- which is what actually gets plotted and thresholded.

    Rangayyan & Krishnan (2024) Sec 8.8 gives the CWT in eq (8.107); Figure
    8.29 is captioned "Scalogram resulting from the CWT of the ECG signal in
    Figure 8.26 using Morlet wavelets", with "the vertical axis represent[ing]
    scale from 1 to 32" -- so the book's scalogram is a scale-indexed, not a
    frequency-indexed, display, which is the convention followed here.  The
    energy interpretation is the general TFD criterion of eq (8.117),
    integral TFD dt dw = integral |x(t)|^2 dt.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    scales : sequence of float or None
        Scales in samples.  Default 1, 2, 4, ... as in Figure 8.29.
    wavelet : str
        ``'morlet'``, ``'mexh'`` or ``'haar'``.
    w0 : float
        Morlet central frequency, eq (8.116).

    Returns
    -------
    RichResult with keys ``scalogram`` (scales x time, nonnegative),
    ``scales``, ``freqs``, ``times``, ``total_energy``,
    ``energy_per_scale``, ``ridge`` (peak scale index at each time),
    ``method``.

    Raises
    ------
    ValueError
        Propagated from :func:`cwt` for bad scales or an unknown wavelet.
    """
    r = cwt(x, fs=fs, wavelet=wavelet, scales=scales, w0=w0)
    co = r["coeffs"]
    sg = [[abs(c) ** 2 for c in row] for row in co]
    n = len(sg[0])
    ridge = [max(range(len(sg)), key=lambda s: sg[s][i]) for i in range(n)]
    return RichResult(
        payload={
            "scalogram": sg,
            "scales": r["scales"],
            "freqs": r["freqs"],
            "times": r["times"],
            "energy_per_scale": [fsum(row) for row in sg],
            "total_energy": fsum(fsum(row) for row in sg),
            "ridge": ridge,
            "method": "Scalogram (|CWT|^2), Rangayyan & Krishnan (2024) "
                      "eq (8.107) and Figure 8.29",
        }
    )


rangayyan_scalogram = scalogram  # pre-policy spelling


# -- rgseizwv: EEG epileptic seizure detection via wavelet energy.
def seizwt(eeg, fs=1.0, wavelet="db4", levels=5, scales=(3, 4, 5), threshold=None):
    """Seizure marker from the fluctuation intensity of EEG wavelet coefficients.

    Why this exists: ictal EEG is not simply larger than interictal EEG, it is
    rougher -- the coefficient sequence jumps around more from one sample to
    the next.  A plain energy or variance measure misses that, because a slow
    high-amplitude artifact scores just as high.  Summing the absolute
    first differences of the wavelet coefficients measures the roughness
    directly, and doing it per scale confines the measurement to the
    3-29 Hz band where intracranial seizure activity lives.

    Rangayyan & Krishnan (2024) Sec 8.17 "Application: Detection of Epileptic
    Seizures in EEG Signals" describes the method of its reference [110]
    (Zhou et al.): "EEG signals were subjected to DWT-based decomposition with
    the Daubechies 4 wavelet and five scales, and wavelet coefficients at
    scales 3, 4, and 5 were selected for further processing.  The selection of
    the scales was based on the observation that seizure signals in
    intracranial EEG signals usually occur in the range of 3 to 29 Hz."  The
    fluctuation intensity is eq (8.132),

        FI(s) = (1/N) sum_{i=1}^{N-1} |d(i+1) - d(i)|,

    "where N is the number of DWT coefficients d(i) for scale s".  The book
    reports that "the values of FI during seizures were typically greater than
    those during other periods" but gives no numerical threshold, so no
    default threshold is invented here: leave ``threshold`` as None and the
    verdict key is None rather than a guess.

    Parameters
    ----------
    eeg : sequence of float
        EEG samples.
    fs : float
        Sampling rate in Hz (used only to report each scale's band).
    wavelet : str
        Default ``'db4'``, as specified in Sec 8.17.
    levels : int
        Default 5, as specified in Sec 8.17.
    scales : sequence of int
        1-based detail levels to score.  Default (3, 4, 5) per Sec 8.17.
    threshold : float or None
        Optional FI cut-off.  None leaves ``seizure_detected`` as None.

    Returns
    -------
    RichResult with keys ``fi`` (FI per selected scale), ``fi_total``,
    ``scales``, ``bands`` (approximate Hz range per scale),
    ``energies``, ``seizure_detected``, ``threshold``, ``method``.

    Raises
    ------
    ValueError
        If a requested scale exceeds ``levels``, or fs is not positive.
    """
    v = _tfneed(eeg, "eeg", 8)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    lv = int(levels)
    sc = [int(s) for s in scales]
    for s in sc:
        if not 1 <= s <= lv:
            raise ValueError(f"scale {s} is outside 1..levels={lv}")
    _a, d, _ln = _tfdwt(v, wavelet, lv)
    # _tfdwt returns details coarse-to-fine; index detail level s (1 = finest)
    fine_to_coarse = list(reversed(d))
    fi, ener, bands = [], [], []
    for s in sc:
        c = fine_to_coarse[s - 1]
        n = len(c)
        if n < 2:
            raise ValueError(f"scale {s} has only {n} coefficient(s); FI needs >= 2")
        fi.append(fsum(abs(c[i + 1] - c[i]) for i in range(n - 1)) / n)
        ener.append(fsum(t * t for t in c))
        bands.append((fs / 2.0 ** (s + 1), fs / 2.0 ** s))
    tot = fsum(fi)
    det = None if threshold is None else bool(tot > float(threshold))
    return RichResult(
        payload={
            "fi": fi,
            "fi_total": tot,
            "scales": sc,
            "bands": bands,
            "energies": ener,
            "seizure_detected": det,
            "threshold": threshold,
            "wavelet": str(wavelet),
            "levels": lv,
            "method": "Fluctuation intensity of DWT coefficients, Rangayyan & "
                      "Krishnan (2024) Sec 8.17 eq (8.132), db4 with five "
                      "scales and scales 3-5 selected as specified there",
        }
    )


rangayyan_seizure_wavelet = seizwt  # pre-policy spelling


# -- rgstftp: STFT parameter selection (window length vs. time/freq resolution tradeoff).
def stftparam(fs, desired_t_res, desired_f_res):
    """Choose an STFT window length, and say whether the request is possible.

    Why this exists: window length is the one knob in short-time analysis and
    it trades the two resolutions against each other.  Asking for 5 ms time
    resolution and 1 Hz frequency resolution at the same time is not a tuning
    problem, it is forbidden; this function answers which of the two the
    caller has to give up.

    Rangayyan & Krishnan (2024) Sec 8.7, eq (8.10), states the time-bandwidth
    limit as

        delta_t * delta_omega >= 1/2,

    with delta_t and delta_omega defined as the second moments in eqs (8.11)
    and (8.13).  In ordinary frequency (omega = 2 pi f) that is
    delta_t * delta_f >= 1/(4 pi).  The same section states plainly that
    "increasing the time resolution of the STFT by making the analysis window
    short in duration will compromise frequency resolution; on the other
    hand, increasing the window duration will lead to a loss in time
    resolution".

    The practical (rectangular-window DFT) resolutions of an M-sample window
    are delta_t = M / fs and delta_f = fs / M, so delta_t * delta_f = 1
    exactly -- the product a real DFT achieves, well above the eq (8.10)
    floor.  Both window lengths implied by the request are reported, together
    with the feasibility verdict.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    desired_t_res : float
        Wanted time resolution in seconds.
    desired_f_res : float
        Wanted frequency resolution in Hz.

    Returns
    -------
    RichResult with keys ``nperseg_time``, ``nperseg_freq``, ``nperseg``,
    ``achieved_t_res``, ``achieved_f_res``, ``tf_product``,
    ``heisenberg_bound``, ``feasible``, ``method``.

    Raises
    ------
    ValueError
        If any argument is not strictly positive.
    """
    fs = float(fs)
    dt = float(desired_t_res)
    df = float(desired_f_res)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if dt <= 0.0:
        raise ValueError("desired_t_res must be positive")
    if df <= 0.0:
        raise ValueError("desired_f_res must be positive")
    m_t = max(2, int(dt * fs))
    m_f = max(2, int(fs / df + 0.5))
    feasible = dt * df >= 1.0
    # A DFT window of M samples delivers dt*df = 1; anything the caller asks
    # for below that cannot be had at any M.  Serve the finer of the two
    # requests when they conflict, and say so.
    m = m_f if not feasible else m_t
    return RichResult(
        payload={
            "nperseg_time": m_t,
            "nperseg_freq": m_f,
            "nperseg": m,
            "achieved_t_res": m / fs,
            "achieved_f_res": fs / m,
            "tf_product": dt * df,
            "heisenberg_bound": 1.0 / (4.0 * pi),
            "feasible": feasible,
            "method": "STFT window selection under the time-bandwidth limit, "
                      "Rangayyan & Krishnan (2024) eq (8.10)",
        }
    )


rangayyan_stft_params = stftparam  # pre-policy spelling


# -- rgstfts: STFT spectrogram (magnitude squared STFT).
def spectrogram(x, fs=1.0, nperseg=64, noverlap=None, window="hann"):
    """Spectrogram: the squared magnitude of the short-time Fourier transform.

    Why this exists: a single Fourier spectrum assumes the signal is
    stationary.  Biomedical signals are not -- the PCG changes character
    between S1, a systolic murmur and S2 within a fraction of a second.  The
    STFT restores time by taking the spectrum of one short window at a time,
    and the spectrogram is the resulting energy surface over the
    time-frequency plane.

    Rangayyan & Krishnan (2024) Sec 8.7 "The short-time Fourier transform":
    eq (8.8) defines the discrete STFT

        X(m, w) = sum_n [x(n) w(n - m)] exp(-j w n),

    and the text immediately after eq (8.9) states that "the squared
    magnitude of the STFT is known as the spectrogram of the signal".
    Eq (8.7) gives the rectangular window; the paragraph after eq (8.9) notes
    that any window from Sec 6.3.4 may be used, and the same section notes
    that it is "common practice to use a symmetrical noncausal window", which
    is the convention adopted here (the DFT index runs over the window, not
    over absolute time, so frames differ only in content and not by a linear
    phase ramp -- that is what makes them invertible by overlap-add).

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    nperseg : int
        Window length M in samples.
    noverlap : int or None
        Samples of overlap between consecutive windows.  Default nperseg // 2.
    window : str
        'rect' (eq 8.7), 'hann', 'hamming' or 'bartlett'.

    Returns
    -------
    RichResult with keys ``spectrogram`` (frames x frequency bins),
    ``stft`` (the complex frames, suitable for :func:`istft`), ``times``,
    ``freqs``, ``nperseg``, ``hop``, ``window``, ``method``.

    Raises
    ------
    ValueError
        If the window is longer than the signal or the overlap is invalid.
    """
    v = _tfneed(x, "x", 2)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    m = int(nperseg)
    if m < 2:
        raise ValueError("nperseg must be >= 2")
    if m > len(v):
        raise ValueError(f"nperseg={m} exceeds the signal length {len(v)}")
    ov = m // 2 if noverlap is None else int(noverlap)
    if not 0 <= ov < m:
        raise ValueError(f"noverlap must satisfy 0 <= noverlap < nperseg, got {ov}")
    hop = m - ov
    w = _tfwin(window, m)
    nf = m // 2 + 1
    freqs = [k * fs / m for k in range(nf)]
    frames, times, spec = [], [], []
    start = 0
    while start + m <= len(v):
        seg = [v[start + i] * w[i] for i in range(m)]
        X = _tfdft(seg)
        frames.append(X)
        spec.append([abs(X[k]) ** 2 for k in range(nf)])
        times.append((start + (m - 1) / 2.0) / fs)
        start += hop
    if not frames:
        raise ValueError("no complete analysis window fits in the signal")
    total = fsum(fsum(r) for r in spec)
    peak = max(range(nf), key=lambda k: fsum(r[k] for r in spec))
    return RichResult(
        payload={
            "spectrogram": spec,
            "stft": frames,
            "times": times,
            "freqs": freqs,
            "nperseg": m,
            "hop": hop,
            "window": str(window),
            "n_frames": len(frames),
            "total_energy": total,
            "peak_freq": freqs[peak],
            "method": "STFT spectrogram, Rangayyan & Krishnan (2024) eq (8.8); "
                      "|STFT|^2 per the definition following eq (8.9)",
        }
    )


rangayyan_stft_spectrogram = spectrogram  # pre-policy spelling


# -- rgswt: Stationary wavelet transform (SWT, undecimated DWT).
def swt(x, wavelet="db4", levels=3):
    """Stationary (undecimated) wavelet transform -- shift-invariant DWT.

    Why this exists: the decimated DWT of :func:`dwt` is shift variant.  Move
    the signal by one sample and the coefficients change completely, because
    the decimator keeps the even samples and throws the odd ones away.  For
    denoising and for detection that is a real defect: the answer depends on
    where you happened to start recording, and thresholding in a shift-variant
    basis produces Gibbs-like oscillations near discontinuities.  Dropping the
    decimation and upsampling the filters instead fixes both, at the cost of
    an L-times redundant representation.

    Rangayyan & Krishnan (2024) Sec 8.8 acknowledges the problem in the
    remark following eq (8.113) -- "the critically sampled DWT is a
    shift-variant transform" -- and points to its reference [80] (Bradley,
    A. P., "Shift-invariance in the discrete wavelet transform") but does NOT
    define the undecimated transform.  The algorithm implemented is from
      Nason, G. P., & Silverman, B. W. (1995). "The stationary wavelet
      transform and some statistical applications." Lecture Notes in
      Statistics 103:281-299,
    the a-trous scheme in which the level-j filters are the base filters with
    2^j - 1 zeros inserted between taps.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Number of levels.

    Returns
    -------
    RichResult with keys ``approx`` (full length), ``details`` (fine to
    coarse, all full length), ``levels``, ``redundancy``,
    ``energy_per_level``, ``method``.

    Raises
    ------
    ValueError
        If levels < 1 or 2^levels exceeds the signal length.
    """
    v = _tfneed(x, "x", 4)
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")
    if 2 ** lv > len(v):
        raise ValueError(
            f"levels={lv} needs a signal of at least {2 ** lv} samples, "
            f"got {len(v)}"
        )
    a, det, _ap = _tfswt(v, wavelet, lv)
    return RichResult(
        payload={
            "approx": a,
            "details": det,
            "levels": lv,
            "wavelet": str(wavelet),
            "redundancy": lv + 1,
            "energy_per_level": [fsum(t * t for t in c) for c in det],
            "method": "Stationary (undecimated, a-trous) wavelet transform, "
                      "Nason & Silverman (1995); shift variance of the "
                      "decimated DWT noted by Rangayyan & Krishnan (2024) "
                      "after eq (8.113)",
        }
    )


rangayyan_swt = swt  # pre-policy spelling


# -- rgswtden: SWT-based denoising (shift-invariant, no Gibbs oscillation).
def swtden(x, wavelet="db4", levels=3, threshold=None, threshold_type="soft"):
    """Shift-invariant wavelet denoising via the undecimated transform.

    Why this exists: thresholding in the decimated DWT basis (:func:`wtthresh`)
    produces pseudo-Gibbs oscillations around sharp features, and the exact
    pattern of those oscillations depends on the alignment of the signal with
    the decimation grid.  Thresholding the undecimated coefficients and
    averaging the reconstructions over all shifts removes that dependence --
    the classical cycle-spinning result.  For an ECG, where the sharp feature
    IS the diagnostic content, that matters.

    Rangayyan & Krishnan (2024) supplies the thresholding rules, eqs (8.103)
    and (8.104), and notes after eq (8.113) that "the critically sampled DWT
    is a shift-variant transform".  The book does NOT define the undecimated
    transform or cycle spinning.  Primary sources:
      Nason, G. P., & Silverman, B. W. (1995). Lecture Notes in Statistics
      103:281-299 (the SWT itself), and
      Coifman, R. R., & Donoho, D. L. (1995). "Translation-invariant
      de-noising." Lecture Notes in Statistics 103:125-150 (cycle spinning).

    The implementation averages the thresholded reconstruction over all
    2^levels circular shifts of the input -- literally the cycle-spinning
    definition -- rather than inventing an inverse for the redundant
    transform.  That reuses the exactly invertible decimated bank of
    :func:`dwt`, so the result cannot drift away from a valid reconstruction.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Number of undecimated levels.
    threshold : float or None
        Explicit T; None uses the universal threshold (Donoho & Johnstone
        1994) computed from the finest undecimated detail level.
    threshold_type : str
        ``'soft'`` or ``'hard'``.

    Returns
    -------
    RichResult with keys ``denoised``, ``threshold``, ``sigma``,
    ``n_zeroed``, ``n_coeffs``, ``n_shifts``, ``residual_energy``,
    ``levels``, ``method``.

    Raises
    ------
    ValueError
        For an unknown threshold_type, a negative threshold, or levels < 1.
    """
    v = _tfneed(x, "x", 4)
    tt = str(threshold_type).strip().lower()
    if tt not in ("soft", "hard"):
        raise ValueError(f"threshold_type must be 'soft' or 'hard', got {threshold_type!r}")
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")
    if 2 ** lv > len(v):
        raise ValueError(f"levels={lv} needs at least {2 ** lv} samples")
    n = len(v)
    nshift = 2 ** lv
    # True cycle spinning (Coifman & Donoho 1995): threshold and reconstruct
    # under every circular shift that the decimator can distinguish, then
    # average.  This reuses the exactly-invertible decimated bank rather than
    # inventing an SWT inverse, so the estimate cannot drift.
    _a0, det0, _ln0 = _tfdwt(v, wavelet, lv)
    med = sorted(abs(t) for t in det0[-1])
    sigma = med[len(med) // 2] / 0.6745
    if threshold is None:
        T = sigma * sqrt(2.0 * log(n))
    else:
        T = float(threshold)
        if T < 0.0:
            raise ValueError("threshold must be non-negative")

    def shrink(w):
        if abs(w) < T:
            return 0.0
        if tt == "hard":
            return w
        return (1.0 if w > 0 else -1.0) * (abs(w) - T)

    acc = [0.0] * n
    zeroed = 0
    ncoef = 0
    for sh in range(nshift):
        rolled = [v[(i + sh) % n] for i in range(n)]
        a, d, ln = _tfdwt(rolled, wavelet, lv)
        nd = []
        for c in d:
            row = [shrink(w) for w in c]
            zeroed += sum(1 for w in row if w == 0.0)
            ncoef += len(row)
            nd.append(row)
        rec = _tfidwt(a, nd, ln, wavelet)[:n]
        for i in range(n):
            acc[(i + sh) % n] += rec[i]
    cur = [t / nshift for t in acc]
    return RichResult(
        payload={
            "denoised": cur,
            "threshold": T,
            "sigma": sigma,
            "n_zeroed": zeroed,
            "n_coeffs": ncoef,
            "n_shifts": nshift,
            "residual_energy": fsum((v[i] - cur[i]) ** 2 for i in range(n)),
            "levels": lv,
            "wavelet": str(wavelet),
            "threshold_type": tt,
            "method": "SWT (cycle-spinning) denoising: thresholds of "
                      "Rangayyan & Krishnan (2024) eqs (8.103)-(8.104) applied "
                      "to the Nason & Silverman (1995) undecimated transform; "
                      "translation invariance per Coifman & Donoho (1995)",
        }
    )


rangayyan_swt_denoise = swtden  # pre-policy spelling


# -- rgvmd: Variational mode decomposition (VMD) into K band-limited modes.
def vmodes(x, K=3, alpha=2000.0, tau=0.0, init="uniform", tol=1e-7, max_iter=300):
    """Variational mode decomposition into K band-limited modes.

    Why this exists: EMD is an algorithm without an objective function -- it
    is defined by what the sifting loop does, so it has no convergence theory,
    it is sensitive to noise, and it decides the number of modes for you.  VMD
    poses the same job as an optimisation: find K modes, each compact around
    its own centre frequency, whose sum is the signal.  That makes the mode
    count an explicit choice, the bandwidth an explicit penalty, and the
    result reproducible.

    Rangayyan & Krishnan (2024) does NOT cover variational mode decomposition
    anywhere -- Sec 9.4.1 "Variants of empirical mode decomposition" discusses
    ensemble EMD and multivariate EMD only.  The primary source is
      Dragomiretskiy, K., & Zosso, D. (2014). "Variational mode
      decomposition." IEEE Trans. Signal Processing 62(3):531-544,
    whose alternating-direction update equations (their eqs 13, 15 and 16) are
    implemented here:

        u_k^ = (f^ - sum_{i != k} u_i^ + lambda^/2)
               / (1 + 2 alpha (omega - omega_k)^2)
        omega_k = sum_omega omega |u_k^|^2 / sum_omega |u_k^|^2
        lambda^ <- lambda^ + tau (f^ - sum_k u_k^)

    on the one-sided spectrum, iterated to a relative-change tolerance.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    K : int
        Number of modes.
    alpha : float
        Bandwidth penalty; larger means narrower modes.
    tau : float
        Dual ascent step.  0 disables the Lagrangian update, which is the
        right choice when the signal is noisy (exact reconstruction is then
        not enforced).
    init : str
        ``'uniform'`` spreads the initial centre frequencies evenly over
        [0, 0.5); ``'zero'`` starts them all at DC.
    tol : float
        Convergence tolerance on the summed relative mode change.
    max_iter : int
        Iteration cap.

    Returns
    -------
    RichResult with keys ``modes`` (K full-length signals, sorted by centre
    frequency), ``center_freqs`` (normalised, cycles/sample), ``K``,
    ``alpha``, ``tau``, ``iterations``, ``converged``,
    ``reconstruction_error``, ``residual_energy``, ``method``.

    Raises
    ------
    ValueError
        If K < 1, alpha <= 0, tau < 0, max_iter < 1, or init is unknown.
    """
    v = _tfneed(x, "x", 8)
    k = int(K)
    if k < 1:
        raise ValueError("K must be >= 1")
    al = float(alpha)
    if al <= 0.0:
        raise ValueError("alpha must be positive")
    ta = float(tau)
    if ta < 0.0:
        raise ValueError("tau must be non-negative")
    mi = int(max_iter)
    if mi < 1:
        raise ValueError("max_iter must be >= 1")
    ini = str(init).strip().lower()
    if ini not in ("uniform", "zero"):
        raise ValueError(f"init must be 'uniform' or 'zero', got {init!r}")
    n = len(v)
    # Work on the one-sided spectrum of the analytic signal: the modes of
    # Dragomiretskiy & Zosso are analytic by construction.
    F = _tfdft(v)
    half = n // 2 + 1
    fh = [F[i] for i in range(half)]
    om = [i / float(n) for i in range(half)]        # cycles per sample
    if ini == "uniform":
        wk = [0.5 * (j + 0.5) / k for j in range(k)]
    else:
        wk = [0.0] * k
    uk = [[0j] * half for _ in range(k)]
    lam = [0j] * half
    it = 0
    conv = False
    for it in range(1, mi + 1):
        change = 0.0
        for j in range(k):
            others = [sum(uk[q][i] for q in range(k) if q != j) for i in range(half)]
            new = []
            for i in range(half):
                den = 1.0 + 2.0 * al * (om[i] - wk[j]) ** 2
                new.append((fh[i] - others[i] + lam[i] / 2.0) / den)
            num = fsum(om[i] * abs(new[i]) ** 2 for i in range(half))
            den = fsum(abs(new[i]) ** 2 for i in range(half))
            if den > 0.0:
                wk[j] = num / den
            prev = fsum(abs(uk[j][i]) ** 2 for i in range(half))
            change += (fsum(abs(new[i] - uk[j][i]) ** 2 for i in range(half))
                       / (prev if prev > 0.0 else 1.0))
            uk[j] = new
        if ta > 0.0:
            tot = [sum(uk[q][i] for q in range(k)) for i in range(half)]
            lam = [lam[i] + ta * (fh[i] - tot[i]) for i in range(half)]
        if change < float(tol):
            conv = True
            break
    # Back to the time domain: rebuild the full Hermitian spectrum from the
    # one-sided modes, so each mode is a real signal.
    modes = []
    for j in range(k):
        full = [0j] * n
        for i in range(half):
            full[i] = uk[j][i]
        for i in range(1, (n + 1) // 2):
            full[n - i] = uk[j][i].conjugate()
        modes.append([t.real for t in _tfidft(full)])
    order = sorted(range(k), key=lambda j: wk[j])
    modes = [modes[j] for j in order]
    wk = [wk[j] for j in order]
    tot = [fsum(m[i] for m in modes) for i in range(n)]
    err = max(abs(tot[i] - v[i]) for i in range(n))
    return RichResult(
        payload={
            "modes": modes,
            "center_freqs": wk,
            "K": k,
            "alpha": al,
            "tau": ta,
            "iterations": it,
            "converged": conv,
            "reconstruction_error": err,
            "residual_energy": fsum((tot[i] - v[i]) ** 2 for i in range(n)),
            "method": "Variational mode decomposition, Dragomiretskiy & Zosso "
                      "(2014) IEEE TSP 62(3):531-544 eqs (13), (15), (16) -- "
                      "not covered by Rangayyan & Krishnan (2024)",
        }
    )


rangayyan_vmd = vmodes  # pre-policy spelling


# -- rgwavstr: Wavelet-based structure detection in biomedical signals (CWT ridges).
def cwtridge(x, fs=1.0, scales=None, wavelet="mexh", w0=5.0, min_prominence=0.1):
    """Locate transient structures in a signal by tracking scalogram ridges.

    Why this exists: a wavelet coefficient is large exactly where the signal
    locally looks like the wavelet.  So the maxima of the scalogram, followed
    across scales, mark where the interesting transients are -- a QRS complex,
    a click in a PCG, a spike in an EEG -- and at what characteristic width.
    This turns the CWT from a picture into a list of detected events.

    Rangayyan & Krishnan (2024) Sec 8.8 provides the machinery: eq (8.107)
    for the CWT and eqs (8.115)-(8.116) for the mother wavelets, and states
    that "the use of appropriate wavelets can assist in the identification and
    analysis of transient, aperiodic, multicomponent, and nonstationary
    signals".  The book illustrates ECG structure detection with the CWT in
    Figures 8.26-8.29 and cites Li, Zheng & Tai, "Detection of ECG
    characteristic points using wavelet transforms", IEEE Trans. Biomedical
    Engineering (ref [76]), for the detection procedure.  The book does NOT
    specify a ridge-extraction algorithm; the local-maximum-with-prominence
    rule used here is the plain reading of that description and is stated as
    such rather than attributed to the book.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    scales : sequence of float or None
        Scales in samples.  Default the dyadic ladder.
    wavelet : str
        ``'mexh'`` (eq 8.115) by default -- it is real and symmetric, so its
        maxima sit on the transient rather than beside it.
    w0 : float
        Morlet central frequency if ``wavelet='morlet'``.
    min_prominence : float
        A structure is reported only if its scalogram peak exceeds this
        fraction of the global scalogram maximum.

    Returns
    -------
    RichResult with keys ``structures`` (list of dicts with ``time``,
    ``sample``, ``scale``, ``freq``, ``energy``), ``n_structures``,
    ``scalogram``, ``scales``, ``times``, ``method``.

    Raises
    ------
    ValueError
        If min_prominence is outside (0, 1].
    """
    p = float(min_prominence)
    if not 0.0 < p <= 1.0:
        raise ValueError("min_prominence must lie in (0, 1]")
    r = scalogram(x, fs=fs, scales=scales, wavelet=wavelet, w0=w0)
    sg = r["scalogram"]
    ns, n = len(sg), len(sg[0])
    gmax = max(max(row) for row in sg)
    if gmax <= 0.0:
        raise ValueError("the signal has no wavelet energy at any scale")
    found = []
    for si in range(ns):
        row = sg[si]
        for i in range(1, n - 1):
            if row[i] > row[i - 1] and row[i] >= row[i + 1] and row[i] >= p * gmax:
                # keep it only if this scale is also a local best across scales
                if si > 0 and sg[si - 1][i] > row[i]:
                    continue
                if si < ns - 1 and sg[si + 1][i] > row[i]:
                    continue
                found.append({"sample": i, "time": r["times"][i],
                              "scale": r["scales"][si], "freq": r["freqs"][si],
                              "energy": row[i]})
    found.sort(key=lambda d: -d["energy"])
    return RichResult(
        payload={
            "structures": found,
            "n_structures": len(found),
            "scalogram": sg,
            "scales": r["scales"],
            "times": r["times"],
            "min_prominence": p,
            "method": "CWT ridge detection of transient structures, "
                      "Rangayyan & Krishnan (2024) Sec 8.8 (eqs 8.107, 8.115, "
                      "8.116); ridge rule is local-maximum-with-prominence, "
                      "not specified by the book",
        }
    )


rangayyan_wavelet_struct = cwtridge  # pre-policy spelling


# -- rgwvcor: Wavelet cross-correlation between two signals at each scale.
def wtxcor(x, y, wavelet="db4", levels=3, max_lag=0):
    """Wavelet cross-correlation between two signals, scale by scale.

    Why this exists: two biomedical signals can be strongly coupled at one
    time scale and unrelated at another -- respiration couples to heart rate
    in the 0.15-0.4 Hz band while the slow trends drift independently.  An
    ordinary correlation coefficient averages all of that into one number and
    can report nearly zero for a pair that is tightly coupled in the band you
    care about.  Correlating the wavelet coefficients level by level keeps the
    scales separate.

    The decomposition is the undecimated transform of :func:`swt`, so that
    both signals keep the same time base at every level and a lag in samples
    means the same thing at every scale (with the decimated DWT a lag of one
    coefficient means 2^j samples, which makes cross-scale comparison
    meaningless).

    Rangayyan & Krishnan (2024) provides the wavelet decomposition, eqs
    (8.111)-(8.113), but does NOT define a wavelet cross-correlation.  The
    scale-by-scale correlation of undecimated wavelet coefficients is from
      Whitcher, B., Guttorp, P., & Percival, D. B. (2000). "Wavelet analysis
      of covariance with application to atmospheric time series." Journal of
      Geophysical Research 105(D11):14941-14962.

    Parameters
    ----------
    x, y : sequence of float
        The two signals; they must have the same length.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Number of scales.
    max_lag : int
        If greater than zero, the correlation is maximised over lags in
        [-max_lag, +max_lag] and the best lag is reported per scale.

    Returns
    -------
    RichResult with keys ``correlations`` (per scale), ``best_lags``,
    ``covariances``, ``scales``, ``overall_correlation`` (the plain
    Pearson correlation of the raw signals, for comparison), ``levels``,
    ``method``.

    Raises
    ------
    ValueError
        If the signals differ in length, are too short, or max_lag is
        negative or exceeds the signal length.
    """
    a = _tfneed(x, "x", 4)
    b = _tfneed(y, "y", 4)
    if len(a) != len(b):
        raise ValueError(f"x and y must have the same length, got {len(a)} and {len(b)}")
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")
    ml = int(max_lag)
    n = len(a)
    if ml < 0 or ml >= n:
        raise ValueError(f"max_lag must satisfy 0 <= max_lag < {n}, got {ml}")
    if 2 ** lv > n:
        raise ValueError(f"levels={lv} needs at least {2 ** lv} samples")
    _aa, da, _ = _tfswt(a, wavelet, lv)
    _ab, db, _ = _tfswt(b, wavelet, lv)

    def corr(u, w, lag):
        m = len(u)
        idx = [i for i in range(m) if 0 <= i + lag < m]
        if len(idx) < 2:
            return 0.0, 0.0
        mu = fsum(u[i] for i in idx) / len(idx)
        mw = fsum(w[i + lag] for i in idx) / len(idx)
        cov = fsum((u[i] - mu) * (w[i + lag] - mw) for i in idx) / len(idx)
        su = sqrt(fsum((u[i] - mu) ** 2 for i in idx) / len(idx))
        sw = sqrt(fsum((w[i + lag] - mw) ** 2 for i in idx) / len(idx))
        return (cov / (su * sw) if su > 0.0 and sw > 0.0 else 0.0), cov

    cors, lags, covs = [], [], []
    for j in range(lv):
        best, bl, bc = -2.0, 0, 0.0
        for lag in range(-ml, ml + 1):
            c, cv = corr(da[j], db[j], lag)
            if abs(c) > abs(best) or best == -2.0:
                best, bl, bc = c, lag, cv
        cors.append(best)
        lags.append(bl)
        covs.append(bc)
    mu = fsum(a) / n
    mw = fsum(b) / n
    sa = sqrt(fsum((t - mu) ** 2 for t in a) / n)
    sb = sqrt(fsum((t - mw) ** 2 for t in b) / n)
    ov = (fsum((a[i] - mu) * (b[i] - mw) for i in range(n)) / n / (sa * sb)
          if sa > 0.0 and sb > 0.0 else 0.0)
    return RichResult(
        payload={
            "correlations": cors,
            "best_lags": lags,
            "covariances": covs,
            "scales": [2 ** j for j in range(lv)],
            "overall_correlation": ov,
            "levels": lv,
            "max_lag": ml,
            "wavelet": str(wavelet),
            "method": "Scale-by-scale wavelet cross-correlation, Whitcher, "
                      "Guttorp & Percival (2000) JGR 105(D11), on the "
                      "undecimated transform of Nason & Silverman (1995); "
                      "wavelet basis per Rangayyan & Krishnan (2024) eq (8.113)",
        }
    )


rangayyan_wavelet_corr = wtxcor  # pre-policy spelling


# -- rgwvd: Wigner-Ville distribution (bilinear TFD).
def wvdist(x, fs=1.0, nfreq=None):
    """Wigner-Ville distribution: the bilinear time-frequency distribution.

    Why this exists: the spectrogram's resolution is capped by the analysis
    window (eq 8.10) -- you buy time resolution with frequency resolution.
    The WVD uses no window at all.  It correlates the signal with a
    time-reversed copy of itself about each instant and Fourier transforms
    that instantaneous autocorrelation, which gives the sharpest possible
    localisation of a single chirping component.  The price is cross terms.

    Rangayyan & Krishnan (2024) Sec 8.9 "Bilinear TFDs", eq (8.123):

        WVD(t, w) = integral x(t + tau/2) x*(t - tau/2) exp(-j w tau) dtau,

    and the sentence immediately following it warns that "a drawback of the
    WVD is that, in the case of multicomponent signals, cross-terms (referred
    to as 'ghost frequencies') are generated in the TFD that could lead to
    misinterpretation".  Those cross terms are real output here, not a bug --
    :func:`gtfd` is the tool for suppressing them.

    The lag variable is discretised on the ANALYTIC signal (Sec 5.5.3), which
    is standard practice: it removes the interference between the positive-
    and negative-frequency images of a real signal and lets the lag be
    sampled at the signal rate rather than twice it.  The resulting kernel is
    the Claasen-Mecklenbrauker form

        W[n, k] = 2 sum_tau z[n+tau] z*[n-tau] exp(-j 4 pi tau k / N),

    so the frequency axis spans 0 to fs/2.  Cost is O(N^2) in the transform
    -- inherent to a bilinear distribution, not an implementation shortcut.

    Parameters
    ----------
    x : sequence of float
        Signal samples.  Keep this short (tens to a few hundred samples).
    fs : float
        Sampling rate in Hz.
    nfreq : int or None
        Number of frequency bins over [0, fs/2).  Default len(x).

    Returns
    -------
    RichResult with keys ``tfd`` (time x frequency, real), ``times``,
    ``freqs``, ``peak_freq``, ``total_energy``, ``method``.

    Raises
    ------
    ValueError
        If fs is not positive or the signal is shorter than 4 samples.
    """
    v = _tfneed(x, "x", 4)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    nf = int(nfreq or len(v))
    if nf < 2:
        raise ValueError("nfreq must be >= 2")
    tfd, freqs = _tfwvd(v, fs, nf)
    times = [i / fs for i in range(len(v))]
    col = [fsum(row[k] for row in tfd) for k in range(nf)]
    peak = max(range(nf), key=lambda k: col[k])
    return RichResult(
        payload={
            "tfd": tfd,
            "times": times,
            "freqs": freqs,
            "peak_freq": freqs[peak],
            "total_energy": fsum(fsum(r) for r in tfd),
            "method": "Wigner-Ville distribution, Rangayyan & Krishnan (2024) "
                      "eq (8.123), analytic-signal (Claasen-Mecklenbrauker) form",
        }
    )


rangayyan_wigner_ville = wvdist  # pre-policy spelling


# -- rgwvener: Wavelet energy per subband (scale).
def wtenergy(x, wavelet="db4", levels=3):
    """Energy of the wavelet coefficients in each subband.

    Why this exists: this is the workhorse feature of every wavelet-based
    classifier in the book -- how the signal's energy is distributed across
    scales, expressed as a handful of numbers instead of a whole transform.
    It is what separates a seizure EEG segment from a normal one, or an
    organised ventricular fibrillation waveform from a disorganised one.

    Rangayyan & Krishnan (2024) Sec 8.15 states the decomposition directly:
    "If Ex is the signal's total energy and s1 to sN are the wavelet scales
    that completely characterize the signal, the signal energy may be written
    as Ex = Es1 + Es2 + Es3 + ... + EsN".  Because the transform of eqs
    (8.111)-(8.113) is orthonormal, that split is exact -- the subband
    energies returned here sum to the energy of the input, which is the
    Parseval identity and is reported as ``energy_balance``.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Decomposition depth.

    Returns
    -------
    RichResult with keys ``energies`` (approximation first, then details
    coarse to fine), ``relative`` (fractions summing to 1), ``labels``,
    ``total_energy``, ``input_energy``, ``energy_balance``,
    ``dominant_band``, ``method``.

    Raises
    ------
    ValueError
        If levels exceeds what the signal length and filter allow.
    """
    v = _tfneed(x, "x", 2)
    lv = int(levels)
    a, d, _ln = _tfdwt(v, wavelet, lv)
    bands = [list(a)] + [list(c) for c in d]
    labels = [f"A{lv}"] + [f"D{lv - i}" for i in range(lv)]
    ener = [fsum(t * t for t in c) for c in bands]
    tot = fsum(ener)
    ein = fsum(t * t for t in v)
    return RichResult(
        payload={
            "energies": ener,
            "relative": [e / tot if tot > 0.0 else 0.0 for e in ener],
            "labels": labels,
            "total_energy": tot,
            "input_energy": ein,
            "energy_balance": abs(tot - ein),
            "dominant_band": labels[max(range(len(ener)), key=lambda i: ener[i])],
            "levels": lv,
            "wavelet": str(wavelet),
            "method": "Wavelet subband energy, Rangayyan & Krishnan (2024) "
                      "Sec 8.15 (Ex = Es1 + Es2 + ... + EsN) over the "
                      "eq (8.111)-(8.113) orthonormal DWT",
        }
    )


rangayyan_wavelet_energy = wtenergy  # pre-policy spelling


# -- rgwvmom: Wavelet coefficient moments (energy, variance, mean) per scale.
def wtmoment(x, wavelet="db4", levels=3):
    """Mean, variance, energy, skewness and kurtosis of each wavelet subband.

    Why this exists: subband energy (:func:`wtenergy`) reduces a whole band
    to one number and throws away its shape.  Two bands can carry identical
    energy and look nothing alike -- one a steady oscillation, the other a
    single spike.  The higher moments tell them apart: kurtosis in particular
    is high exactly when the band's energy is concentrated in a few large
    coefficients, which is the signature of a transient.

    The subbands are those of the orthonormal DWT of Rangayyan & Krishnan
    (2024) eqs (8.111)-(8.113); the book's own subband feature is the energy
    of Sec 8.15.  The moments themselves are the ordinary sample moments and
    are not attributed to the book.  The absolute-difference feature of
    eq (8.132) is a related coefficient-domain statistic and is available
    separately as :func:`seizwt`.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Decomposition depth.

    Returns
    -------
    RichResult with keys ``moments`` (one dict per band with ``label``,
    ``n``, ``mean``, ``variance``, ``energy``, ``skewness``, ``kurtosis``),
    ``labels``, ``levels``, ``method``.  Skewness and kurtosis are reported
    as None for a band with fewer than 3 coefficients or zero variance, where
    they are undefined rather than zero.

    Raises
    ------
    ValueError
        If levels exceeds what the signal length and filter allow.
    """
    v = _tfneed(x, "x", 2)
    lv = int(levels)
    a, d, _ln = _tfdwt(v, wavelet, lv)
    bands = [list(a)] + [list(c) for c in d]
    labels = [f"A{lv}"] + [f"D{lv - i}" for i in range(lv)]
    out = []
    for lab, c in zip(labels, bands):
        n = len(c)
        mu = fsum(c) / n
        var = fsum((t - mu) ** 2 for t in c) / n
        sd = sqrt(var)
        sk = ku = None
        if n >= 3 and sd > 0.0:
            sk = fsum(((t - mu) / sd) ** 3 for t in c) / n
            ku = fsum(((t - mu) / sd) ** 4 for t in c) / n
        out.append({"label": lab, "n": n, "mean": mu, "variance": var,
                    "energy": fsum(t * t for t in c), "skewness": sk,
                    "kurtosis": ku})
    return RichResult(
        payload={
            "moments": out,
            "labels": labels,
            "levels": lv,
            "wavelet": str(wavelet),
            "method": "Per-subband sample moments of the Rangayyan & Krishnan "
                      "(2024) eq (8.111)-(8.113) DWT coefficients; band energy "
                      "per Sec 8.15",
        }
    )


rangayyan_wavelet_moments = wtmoment  # pre-policy spelling


# -- rgwvpkt: Wavelet packet decomposition (full binary tree).
def wpt(x, wavelet="db4", levels=3):
    """Wavelet packet decomposition: the full binary filter-bank tree.

    Why this exists: the DWT only ever splits the LOW band again, so its
    frequency tiling is logarithmic and it resolves high frequencies coarsely.
    A signal whose information sits in a narrow high band -- a murmur, an EMG
    burst, a seizure rhythm -- is badly served by that.  The wavelet packet
    tree splits both bands at every level, giving a uniform tiling of
    2^levels equal subbands, and lets a best-basis search pick whichever
    subtree suits the signal.

    Rangayyan & Krishnan (2024) Sec 8.8.1 states that "the approach of
    selecting the best basis among a dictionary of bases by minimizing a cost
    function is known as the method of wavelet packet", that the approach
    "uses a family of orthogonal bases that include different types of
    localized TF functions (also known as TF atoms)" and "decomposes the
    given signal into TF atoms that are adapted to the TF structures expected
    or known to be present in the signal", citing its reference [81]
    (Wickerhauser, M. V., Adapted Wavelet Analysis from Theory to Software,
    IEEE Press, 1994).  The book gives no equation for the tree; the recursion
    is the direct one implied by that description, applying the eq
    (8.111)-(8.113) filter bank to both children.

    Node ordering is natural (Paley) order, i.e. the order in which the tree
    is generated, NOT frequency order -- the odd nodes at each level have
    reversed frequency sense because of the highpass branch.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Tree depth; the leaf level has 2^levels nodes.

    Returns
    -------
    RichResult with keys ``nodes`` (dict level -> list of coefficient lists),
    ``leaves`` (the 2^levels terminal nodes), ``n_leaves``,
    ``energy_per_leaf``, ``dominant_leaf``, ``entropy`` (Shannon entropy of
    the normalised leaf energies, the usual best-basis cost), ``levels``,
    ``method``.

    Raises
    ------
    ValueError
        If levels < 1 or the signal is too short for the requested depth.
    """
    v = _tfneed(x, "x", 4)
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")
    h, g, _, _ = _tffilters(wavelet)
    if len(v) < len(h) * (2 ** (lv - 1)):
        raise ValueError(
            f"signal of length {len(v)} is too short for {lv} packet levels "
            f"with a length-{len(h)} filter"
        )
    nodes = {0: [list(v)]}
    for lev in range(1, lv + 1):
        cur = []
        for node in nodes[lev - 1]:
            lo, hi = _tfdwtstep(node, h, g)
            cur.append(lo)
            cur.append(hi)
        nodes[lev] = cur
    leaves = nodes[lv]
    ener = [fsum(t * t for t in c) for c in leaves]
    tot = fsum(ener)
    ent = 0.0
    if tot > 0.0:
        for e in ener:
            p = e / tot
            if p > 0.0:
                ent -= p * log(p)
    return RichResult(
        payload={
            "nodes": nodes,
            "leaves": leaves,
            "n_leaves": len(leaves),
            "energy_per_leaf": ener,
            "dominant_leaf": max(range(len(ener)), key=lambda i: ener[i]),
            "entropy": ent,
            "levels": lv,
            "wavelet": str(wavelet),
            "method": "Wavelet packet decomposition (full binary tree, natural "
                      "order), Rangayyan & Krishnan (2024) Sec 8.8.1 and its "
                      "reference [81], Wickerhauser (1994)",
        }
    )


rangayyan_wavelet_packet = wpt  # pre-policy spelling


# -- rgwvth: Wavelet denoising via soft/hard thresholding.
def wtthresh(x, wavelet="db4", levels=3, threshold_type="soft", threshold=None):
    """Wavelet denoising by hard or soft thresholding of the coefficients.

    Why this exists: white noise spreads its energy evenly over every wavelet
    coefficient, while a structured signal concentrates its energy into a few
    large ones.  So "keep the big coefficients, zero the rest" separates the
    two without needing a frequency band to filter on -- which is the whole
    point for signals such as the ECG or PPG whose useful content and whose
    artifact occupy the same band.

    Rangayyan & Krishnan (2024) Sec 8.8.1, eq (8.103), hard thresholding:

        w~_m = w_m if |w_m| >= T, else 0,

    and eq (8.104), soft thresholding:

        w~_m = sgn(w_m)(|w_m| - T) if |w_m| >= T, else 0,

    with eq (8.105) reconstructing x~(t) = sum w~_m psi_m(t).  The book uses a
    fixed value (it reports T = 0.5 for the illustration in Sec 9.6) and does
    NOT give a rule for choosing T.  When ``threshold`` is None the universal
    threshold T = sigma sqrt(2 ln N) is used, with sigma estimated as the
    median absolute deviation of the finest detail level divided by 0.6745,
    from
      Donoho, D. L., & Johnstone, I. M. (1994). "Ideal spatial adaptation by
      wavelet shrinkage." Biometrika 81(3):425-455.
    That source is cited explicitly because it is not the book's.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'haar'``/``'db1'`` .. ``'db10'``.
    levels : int
        Decomposition depth.
    threshold_type : str
        ``'soft'`` (eq 8.104) or ``'hard'`` (eq 8.103).
    threshold : float or None
        Explicit T.  None selects the universal threshold.

    Returns
    -------
    RichResult with keys ``denoised``, ``threshold``, ``sigma``,
    ``n_zeroed``, ``n_coeffs``, ``sparsity``, ``noise_removed``,
    ``threshold_type``, ``method``.

    Raises
    ------
    ValueError
        For an unknown threshold_type or a negative explicit threshold.
    """
    v = _tfneed(x, "x", 4)
    tt = str(threshold_type).strip().lower()
    if tt not in ("soft", "hard"):
        raise ValueError(f"threshold_type must be 'soft' or 'hard', got {threshold_type!r}")
    a, d, ln = _tfdwt(v, wavelet, int(levels))
    finest = d[-1]
    med = sorted(abs(t) for t in finest)
    m = med[len(med) // 2] if med else 0.0
    sigma = m / 0.6745
    if threshold is None:
        T = sigma * sqrt(2.0 * log(len(v))) if len(v) > 1 else 0.0
    else:
        T = float(threshold)
        if T < 0.0:
            raise ValueError("threshold must be non-negative")
    zeroed = 0
    newd = []
    for c in d:
        row = []
        for w in c:
            if abs(w) < T:
                row.append(0.0)
                zeroed += 1
            elif tt == "hard":
                row.append(w)
            else:
                row.append((1.0 if w > 0 else -1.0) * (abs(w) - T))
        newd.append(row)
    den = _tfidwt(a, newd, ln, wavelet)[:len(v)]
    ncoef = sum(len(c) for c in d)
    return RichResult(
        payload={
            "denoised": den,
            "threshold": T,
            "sigma": sigma,
            "n_zeroed": zeroed,
            "n_coeffs": ncoef,
            "sparsity": zeroed / ncoef if ncoef else 0.0,
            "noise_removed": fsum((v[i] - den[i]) ** 2 for i in range(len(v))),
            "threshold_type": tt,
            "wavelet": str(wavelet),
            "method": "Wavelet shrinkage, Rangayyan & Krishnan (2024) eqs "
                      "(8.103)-(8.105); universal threshold from Donoho & "
                      "Johnstone (1994) when T is not supplied",
        }
    )


rangayyan_wavelet_threshold = wtthresh  # pre-policy spelling


# -- rgwvvar: Wavelet variance (Allan variance) by scale.
def wtvar(x, wavelet="db1", levels=3):
    """Wavelet variance: how the variance of a signal decomposes by scale.

    Why this exists: the variance of a nonstationary or long-memory signal is
    not one number -- how much it wobbles depends entirely on the time scale
    you look over.  The wavelet variance answers "how much variance lives at
    scale 2^j", and it does so for signals whose overall variance may not even
    converge (drifting baselines, 1/f noise).  With the two-tap wavelet it is
    the Allan variance used for clock and sensor stability.

    The transform is the undecimated one (see :func:`swt`), because the
    decimated DWT gives only N/2^j coefficients at scale j and a badly biased
    estimate at the coarse scales.  Boundary coefficients -- those whose
    filter support wraps around the periodic extension -- are excluded, which
    is what makes the estimator unbiased.

    Rangayyan & Krishnan (2024) does NOT define the wavelet variance or the
    Allan variance.  Primary sources:
      Percival, D. B. (1995). "On estimation of the wavelet variance."
      Biometrika 82(3):619-631 (the unbiased scale-by-scale estimator), and
      Allan, D. W. (1966). "Statistics of atomic frequency standards."
      Proceedings of the IEEE 54(2):221-230 (the two-tap special case).

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    wavelet : str
        ``'db1'`` (default) reproduces the Allan variance; ``'db2'``..
        ``'db10'`` give higher-order wavelet variances that tolerate
        polynomial trends of correspondingly higher degree.
    levels : int
        Number of scales.

    Returns
    -------
    RichResult with keys ``variances`` (one per scale), ``scales``
    (2^j in samples), ``n_used`` (unbiased coefficient count per scale),
    ``total_variance``, ``sample_variance``, ``dominant_scale``,
    ``is_allan``, ``method``.

    Raises
    ------
    ValueError
        If levels < 1, or a scale has no interior coefficients left after
        boundary exclusion (the signal is too short for that scale).
    """
    v = _tfneed(x, "x", 4)
    lv = int(levels)
    if lv < 1:
        raise ValueError("levels must be >= 1")
    if 2 ** lv > len(v):
        raise ValueError(f"levels={lv} needs at least {2 ** lv} samples")
    h, _g, _, _ = _tffilters(wavelet)
    L = len(h)
    n = len(v)
    _a, det, _ap = _tfswt(v, wavelet, lv)
    variances, used = [], []
    for j, c in enumerate(det):
        # Boundary coefficients of the level-j filter span (L-1)*2^j + 1
        # samples; drop them so the estimate uses only interior data.
        span = (L - 1) * (2 ** j)
        keep = [c[i] for i in range(span, n)]
        if not keep:
            raise ValueError(
                f"level {j + 1} has no interior coefficients for a signal of "
                f"length {n}; reduce levels or use a shorter filter"
            )
        variances.append(fsum(t * t for t in keep) / (2.0 ** (j + 1) * len(keep)))
        used.append(len(keep))
    mu = fsum(v) / n
    return RichResult(
        payload={
            "variances": variances,
            "scales": [2 ** j for j in range(lv)],
            "n_used": used,
            "total_variance": fsum(variances),
            "sample_variance": fsum((t - mu) ** 2 for t in v) / n,
            "dominant_scale": 2 ** max(range(lv), key=lambda j: variances[j]),
            "is_allan": _tfdbname(wavelet) == 1,
            "wavelet": str(wavelet),
            "method": "Unbiased wavelet variance by scale, Percival (1995) "
                      "Biometrika 82(3):619-631, on the Nason & Silverman "
                      "(1995) undecimated transform; db1 gives the Allan "
                      "variance of Allan (1966)",
        }
    )


rangayyan_wavelet_variance = wtvar  # pre-policy spelling


# -- rng246: Two-impulse input modeling a wavelet plus echo..
def echoimp(a, n_0, n):
    """Two-impulse excitation modelling a wavelet and one echo.

    Why this exists: the whole homomorphic-filtering argument of Chapter 4
    rests on writing a signal with an echo as a convolution, y = x * h, where
    h is the basic wavelet and x is a two-impulse train.  This function IS
    that x: the excitation whose second impulse creates the echo.  Convolving
    it with any wavelet produces the composite signal, which is why it is the
    natural starting point for constructing test signals for cepstral echo
    detection.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.74):

        x(n) = delta(n) + a delta(n - n0),

    "with a and n0 being two constants".  The book adds that "the coefficient
    a indicates the magnitude of the second appearance of the basic wavelet
    (called an echo in seismic applications), and n0 indicates its delay
    (pitch in the case of a voiced speech signal)", and notes that "the
    sampling interval T is ignored, or assumed to be normalized to unity in
    this example".

    Parameters
    ----------
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    n : int or sequence of int
        Sample index or indices at which to evaluate x(n).  A plain int N is
        treated as the range 0..N-1, so ``echoimp(0.5, 10, 32)`` returns a
        32-sample excitation.

    Returns
    -------
    RichResult with keys ``x`` (the excitation, a list), ``n`` (the indices),
    ``a``, ``n_0``, ``method``.

    Raises
    ------
    ValueError
        If n_0 is not a positive integer, or n is empty / not integral.
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    if isinstance(n, int):
        if n < 1:
            raise ValueError("n must be a positive length or a sequence of indices")
        idx = list(range(n))
    else:
        idx = [int(t) for t in n]
        if not idx:
            raise ValueError("n must not be empty")
    x = [(1.0 if i == 0 else 0.0) + (a if i == n0 else 0.0) for i in idx]
    return RichResult(
        payload={
            "x": x,
            "n": idx,
            "a": a,
            "n_0": n0,
            "method": "Two-impulse echo excitation, Rangayyan & Krishnan "
                      "(2024) eq (4.74)",
        }
    )


rangayyan_ch4_signal_with_echo_input = echoimp  # pre-policy spelling


# -- rng247: Time-domain expression for a wavelet h(n) plus an echo at delay n_0..
def echosig(h, a, n_0, n=None):
    """Composite signal: a basic wavelet plus a delayed, scaled copy of itself.

    Why this exists: this is what an echo actually looks like in the time
    domain, and looking at it is the point -- for a delay shorter than the
    wavelet the two copies overlap and the echo is not visible by eye at all.
    That is precisely why the cepstral machinery of the rest of Sec 4.8 is
    needed, and this function generates the signals it is tested on.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.75):

        y(n) = h(n) + a h(n - n0),

    which the book derives as the convolution of the wavelet h(n) with the
    two-impulse excitation of eq (4.74).  "The signal thus has two occurrences
    of the basic wavelet h(n) at n = 0 and n = n0."  The book's illustration
    (Figure 4.26) uses a = 0.5 and n0 = 0.01125 s.

    Parameters
    ----------
    h : sequence of float
        The basic wavelet h(n), starting at n = 0.
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    n : int or sequence of int or None
        Output length or explicit indices.  Default len(h) + n_0, which is
        just long enough to hold the whole echo.

    Returns
    -------
    RichResult with keys ``y``, ``n``, ``h``, ``a``, ``n_0``,
    ``echo_visible`` (False when n_0 < len(h), i.e. the two copies overlap
    and no separate echo can be seen in the waveform), ``method``.

    Raises
    ------
    ValueError
        If h is empty or n_0 is not a positive integer.
    """
    hh = aslist(h)
    if not hh:
        raise ValueError("h must contain at least one sample")
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    if n is None:
        idx = list(range(len(hh) + n0))
    elif isinstance(n, int):
        if n < 1:
            raise ValueError("n must be a positive length or a sequence of indices")
        idx = list(range(n))
    else:
        idx = [int(t) for t in n]
        if not idx:
            raise ValueError("n must not be empty")

    def hv(i):
        return hh[i] if 0 <= i < len(hh) else 0.0

    y = [hv(i) + a * hv(i - n0) for i in idx]
    return RichResult(
        payload={
            "y": y,
            "n": idx,
            "h": hh,
            "a": a,
            "n_0": n0,
            "echo_visible": bool(n0 >= len(hh)),
            "method": "Wavelet plus echo in the time domain, Rangayyan & "
                      "Krishnan (2024) eq (4.75)",
        }
    )


rangayyan_ch4_signal_with_echo_output = echosig  # pre-policy spelling


# -- rng248: Z-transform of a signal with a wavelet and an echo..
def echoz(a, n_0, z, H=None):
    """z-transform of a signal with a wavelet and an echo.

    Why this exists: the echo, which is a shift-and-add in time, becomes a
    simple multiplicative factor in z.  That factorisation -- signal =
    wavelet times echo comb -- is what makes the logarithm of the next
    equation turn the product into a sum, and the sum is what a cepstral
    filter can separate.  The whole homomorphic argument is set up here.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.76), obtained by "taking the
    z-transform of the signal in Equation 4.75":

        Y(z) = (1 + a z^(-n0)) H(z).

    Parameters
    ----------
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    z : complex or sequence of complex
        Point(s) in the z plane at which to evaluate.
    H : complex or sequence of complex or None
        The wavelet transform H(z) at the same point(s).  If None, H = 1 is
        used and the result is the echo factor alone.

    Returns
    -------
    RichResult with keys ``Y``, ``echo_factor`` (the (1 + a z^-n0) term),
    ``z``, ``H``, ``a``, ``n_0``, ``method``.

    Raises
    ------
    ValueError
        If n_0 is not a positive integer, H has a different length from z,
        or any z is zero (z^-n0 is not defined there).
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    zs = [complex(z)] if isinstance(z, (int, float, complex)) else [complex(t) for t in z]
    if not zs:
        raise ValueError("z must not be empty")
    for t in zs:
        if t == 0:
            raise ValueError("z = 0 is outside the region of convergence of z^(-n_0)")
    if H is None:
        hs = [1.0 + 0j] * len(zs)
    elif isinstance(H, (int, float, complex)):
        hs = [complex(H)] * len(zs)
    else:
        hs = [complex(t) for t in H]
        if len(hs) != len(zs):
            raise ValueError(f"H has length {len(hs)} but z has length {len(zs)}")
    fac = [1.0 + a * t ** (-n0) for t in zs]
    Y = [fac[i] * hs[i] for i in range(len(zs))]
    return RichResult(
        payload={
            "Y": Y,
            "echo_factor": fac,
            "z": zs,
            "H": hs,
            "a": a,
            "n_0": n0,
            "method": "z-transform of a wavelet with an echo, Rangayyan & "
                      "Krishnan (2024) eq (4.76)",
        }
    )


rangayyan_ch4_z_transform_signal_echo = echoz  # pre-policy spelling


# -- rng249: Fourier-domain expression for a signal with a wavelet plus echo..
def echospec(a, n_0, omega, H=None):
    """Fourier spectrum of a signal with a wavelet and an echo.

    Why this exists: evaluating eq (4.76) on the unit circle is what turns
    the algebra into something measurable.  The echo factor becomes a comb
    with period 2 pi / n0 in frequency, so the echo shows up as periodic
    ripple in the spectrum -- the effect the log-spectrum and cepstral
    methods later exploit to read the delay off directly.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.77): "If the z-transform is
    evaluated on the unit circle, we get the Fourier-transform-based
    expression

        Y(w) = [1 + a exp(-j w n0)] H(w)."

    Parameters
    ----------
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    omega : float or sequence of float
        Radian frequency (or frequencies) at which to evaluate.
    H : complex or sequence of complex or None
        H(w) at the same frequencies.  None gives H = 1.

    Returns
    -------
    RichResult with keys ``Y``, ``echo_factor``, ``magnitude``, ``phase``,
    ``omega``, ``ripple_period`` (2 pi / n0, the frequency spacing of the
    comb the echo imposes), ``a``, ``n_0``, ``method``.

    Raises
    ------
    ValueError
        If n_0 is not a positive integer, omega is empty, or H does not match
        omega in length.
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    ws = [float(omega)] if isinstance(omega, (int, float)) else [float(t) for t in omega]
    if not ws:
        raise ValueError("omega must not be empty")
    if H is None:
        hs = [1.0 + 0j] * len(ws)
    elif isinstance(H, (int, float, complex)):
        hs = [complex(H)] * len(ws)
    else:
        hs = [complex(t) for t in H]
        if len(hs) != len(ws):
            raise ValueError(f"H has length {len(hs)} but omega has length {len(ws)}")
    fac = [1.0 + a * cmath.exp(-1j * w * n0) for w in ws]
    Y = [fac[i] * hs[i] for i in range(len(ws))]
    return RichResult(
        payload={
            "Y": Y,
            "echo_factor": fac,
            "magnitude": [abs(t) for t in Y],
            "phase": [atan2(t.imag, t.real) for t in Y],
            "omega": ws,
            "ripple_period": 2.0 * pi / n0,
            "a": a,
            "n_0": n0,
            "method": "Fourier spectrum of a wavelet with an echo, Rangayyan "
                      "& Krishnan (2024) eq (4.77)",
        }
    )


rangayyan_ch4_fourier_signal_echo = echospec  # pre-policy spelling


# -- rng250: Complex log of the spectrum of a signal with a wavelet plus echo..
def echologsp(a, n_0, omega, H_hat=None, n_terms=None):
    """Complex log spectrum of a signal with a wavelet plus echo.

    Why this exists: this is the step that makes the whole method work.  In
    eq (4.77) the wavelet and the echo are multiplied together; taking the
    logarithm makes them ADD, and addition is something a linear filter can
    undo.  That is the definition of homomorphic filtering, and everything
    after this equation is just working out what the two additive parts look
    like.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.78): "Taking the logarithm, we
    have

        Y^(w) = H^(w) + log[1 + a exp(-j w n0)]."

    The book then notes that "if a < 1, the log term may be expanded in a
    power series, to get" eq (4.79),

        Y^(w) = H^(w) + a exp(-j w n0) - (a^2/2) exp(-2 j w n0)
                + (a^3/3) exp(-3 j w n0) - ...,

    which is the form that shows the echo contributes impulses at multiples of
    n0.  Both the exact logarithm and the truncated series of eq (4.79) are
    returned so the truncation can be checked rather than trusted.

    Parameters
    ----------
    a : float
        Echo amplitude.  |a| >= 1 makes the eq (4.79) series divergent; the
        exact log is still returned but ``series`` is None and
        ``series_valid`` is False.
    n_0 : int
        Echo delay in samples.
    omega : float or sequence of float
        Radian frequency or frequencies.
    H_hat : complex or sequence of complex or None
        The wavelet's own complex log spectrum H^(w).  None gives 0.
    n_terms : int or None
        Number of terms of the eq (4.79) expansion.  Default 10.

    Returns
    -------
    RichResult with keys ``Y_hat`` (exact, eq 4.78), ``echo_log`` (just the
    log term), ``series`` (eq 4.79 truncation, or None), ``series_valid``,
    ``series_error`` (max |exact - series|, or None), ``omega``, ``a``,
    ``n_0``, ``n_terms``, ``method``.

    Raises
    ------
    ValueError
        If n_0 is not positive, omega is empty, n_terms < 1, H_hat does not
        match omega in length, or 1 + a exp(-j w n0) vanishes at some omega
        (the logarithm is then undefined -- this happens when a = -1 or
        a = 1 with the right phase).
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    ws = [float(omega)] if isinstance(omega, (int, float)) else [float(t) for t in omega]
    if not ws:
        raise ValueError("omega must not be empty")
    nt = 10 if n_terms is None else int(n_terms)
    if nt < 1:
        raise ValueError("n_terms must be >= 1")
    if H_hat is None:
        hs = [0j] * len(ws)
    elif isinstance(H_hat, (int, float, complex)):
        hs = [complex(H_hat)] * len(ws)
    else:
        hs = [complex(t) for t in H_hat]
        if len(hs) != len(ws):
            raise ValueError(f"H_hat has length {len(hs)} but omega has length {len(ws)}")
    elog = []
    for w in ws:
        t = 1.0 + a * cmath.exp(-1j * w * n0)
        if abs(t) < 1e-12:
            raise ValueError(
                f"1 + a exp(-j w n_0) vanishes at omega={w}; the complex "
                f"logarithm of eq (4.78) is undefined there"
            )
        elog.append(cmath.log(t))
    Yh = [hs[i] + elog[i] for i in range(len(ws))]
    valid = abs(a) < 1.0
    ser = None
    err = None
    if valid:
        ser = []
        for i, w in enumerate(ws):
            acc = 0j
            for k in range(1, nt + 1):
                # log(1+u) = u - u^2/2 + u^3/3 - ... with u = a exp(-j w n0)
                acc += ((-1.0) ** (k + 1)) * (a ** k) / k * cmath.exp(-1j * k * w * n0)
            ser.append(hs[i] + acc)
        err = max(abs(Yh[i] - ser[i]) for i in range(len(ws)))
    return RichResult(
        payload={
            "Y_hat": Yh,
            "echo_log": elog,
            "series": ser,
            "series_valid": valid,
            "series_error": err,
            "omega": ws,
            "a": a,
            "n_0": n0,
            "n_terms": nt,
            "method": "Complex log spectrum of a wavelet with an echo, "
                      "Rangayyan & Krishnan (2024) eq (4.78), with the "
                      "eq (4.79) power-series expansion",
        }
    )


rangayyan_ch4_log_signal_echo = echologsp  # pre-policy spelling


# -- rng252: Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0)..
def echocep(h_hat, a, n_0, n=None, n_terms=None):
    """Complex cepstrum of a wavelet plus echo: an impulse train at multiples of n0.

    Why this exists: this is the payoff of the whole Sec 4.8 derivation.  An
    echo that is invisible in the waveform and only a faint ripple in the
    spectrum becomes, in the cepstrum, a clean spike at the echo delay --
    readable directly, with its height telling you the echo amplitude.  That
    is why cepstral analysis is the tool for pitch detection in speech and
    for echo arrival times in seismic and heart-sound work.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.80), obtained by "taking the
    inverse Fourier transform" of eq (4.79):

        y^(n) = h^(n) + a delta(n - n0) - (a^2/2) delta(n - 2 n0)
                + (a^3/3) delta(n - 3 n0) - ...

    The book states the conclusion explicitly: "the complex cepstrum of a
    signal with a basic wavelet and an echo is equal to the complex cepstrum
    of the basic wavelet plus a series of impulses at the echo delay and
    integral multiples thereof.  The amplitudes of the impulses are
    proportional to the echo amplitude (the factor a) and decay for the
    higher-order repetitions (if a < 1)."  It also notes the shortpass /
    longpass separation that follows: assuming h^(n) "decays to negligible
    values" beyond a cutoff, a window on |n| < nc recovers the wavelet and the
    remainder recovers the excitation.

    Parameters
    ----------
    h_hat : sequence of float
        The complex cepstrum h^(n) of the basic wavelet, indexed from n = 0.
        Pass an empty sequence to see the echo impulse train alone.
    a : float
        Echo amplitude.  The series of eq (4.80) follows eq (4.79) and so
        requires |a| < 1; a larger a raises ValueError rather than returning
        a divergent train.
    n_0 : int
        Echo delay in samples.
    n : int or sequence of int or None
        Output length or explicit indices.  Default len(h_hat) extended to
        cover n_terms echo impulses.
    n_terms : int or None
        Number of impulses in the train.  Default is however many fit in the
        output range.

    Returns
    -------
    RichResult with keys ``y_hat``, ``n``, ``impulses`` (list of
    (index, amplitude) pairs, the a^k/k series), ``n_impulses``,
    ``echo_delay``, ``a``, ``method``.

    Raises
    ------
    ValueError
        If |a| >= 1 (eq 4.79 diverges), n_0 is not positive, or n is empty.
    """
    hh = aslist(h_hat)
    a = float(a)
    if abs(a) >= 1.0:
        raise ValueError(
            f"|a| = {abs(a)} >= 1; the power series of eq (4.79) that gives "
            f"eq (4.80) requires a < 1"
        )
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    if n is None:
        nt0 = 4 if n_terms is None else int(n_terms)
        idx = list(range(max(len(hh), n0 * nt0 + 1)))
    elif isinstance(n, int):
        if n < 1:
            raise ValueError("n must be a positive length or a sequence of indices")
        idx = list(range(n))
    else:
        idx = [int(t) for t in n]
        if not idx:
            raise ValueError("n must not be empty")
    top = max(idx)
    nt = (top // n0) if n_terms is None else int(n_terms)
    if nt < 1:
        raise ValueError("n_terms must be >= 1")
    imp = {}
    pairs = []
    for k in range(1, nt + 1):
        amp = ((-1.0) ** (k + 1)) * (a ** k) / k
        imp[k * n0] = amp
        pairs.append((k * n0, amp))
    y = []
    for i in idx:
        v = hh[i] if 0 <= i < len(hh) else 0.0
        y.append(v + imp.get(i, 0.0))
    return RichResult(
        payload={
            "y_hat": y,
            "n": idx,
            "impulses": pairs,
            "n_impulses": len(pairs),
            "echo_delay": n0,
            "a": a,
            "method": "Complex cepstrum of a wavelet with an echo, Rangayyan "
                      "& Krishnan (2024) eq (4.80)",
        }
    )


rangayyan_ch4_complex_cepstrum_signal_with_echo = echocep  # pre-policy spelling


# -- rng256: Squared magnitude (power spectrum) of a signal with wavelet plus echo..
def echopsd(H, a, n_0, z):
    """Squared magnitude of the z-transform of a wavelet with an echo.

    Why this exists: the power cepstrum route avoids phase unwrapping, which
    is the fragile part of the complex cepstrum -- but it can only be built on
    the squared magnitude, so this is where that branch of Sec 4.8 starts.
    The trade is stated by the book itself: no unwrapping needed, but the
    signal components can no longer be separated, only detected.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.84), for "a signal with two
    occurrences of a basic wavelet h(n) at n = 0 and n = n0 as in Equations
    4.74 and 4.75":

        |Y(z)|^2 = |H(z)|^2 |1 + a z^(-n0)|^2.

    This is the multiplicative decomposition that eq (4.81) turns into the
    power cepstrum and eq (4.82) turns into the additive form
    y^p(n) = x^p(n) + h^p(n).

    Parameters
    ----------
    H : complex or sequence of complex
        H(z) at the evaluation point(s).
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    z : complex or sequence of complex
        Point(s) in the z plane.

    Returns
    -------
    RichResult with keys ``power`` (|Y(z)|^2), ``wavelet_power`` (|H(z)|^2),
    ``echo_power`` (|1 + a z^-n0|^2), ``z``, ``a``, ``n_0``, ``method``.

    Raises
    ------
    ValueError
        If n_0 is not positive, H and z differ in length, or any z is zero.
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    zs = [complex(z)] if isinstance(z, (int, float, complex)) else [complex(t) for t in z]
    if not zs:
        raise ValueError("z must not be empty")
    for t in zs:
        if t == 0:
            raise ValueError("z = 0 is outside the region of convergence of z^(-n_0)")
    if isinstance(H, (int, float, complex)):
        hs = [complex(H)] * len(zs)
    else:
        hs = [complex(t) for t in H]
        if len(hs) != len(zs):
            raise ValueError(f"H has length {len(hs)} but z has length {len(zs)}")
    hp = [abs(t) ** 2 for t in hs]
    ep = [abs(1.0 + a * zs[i] ** (-n0)) ** 2 for i in range(len(zs))]
    return RichResult(
        payload={
            "power": [hp[i] * ep[i] for i in range(len(zs))],
            "wavelet_power": hp,
            "echo_power": ep,
            "z": zs,
            "a": a,
            "n_0": n0,
            "method": "Power spectrum of a wavelet with an echo, Rangayyan & "
                      "Krishnan (2024) eq (4.84)",
        }
    )


rangayyan_ch4_power_spectrum_signal_echo = echopsd  # pre-policy spelling


# -- rng257: Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation..
def echologpsd(H, a, n_0, omega):
    """Log power spectrum of a wavelet with an echo: the sinusoidal ripple.

    Why this exists: this equation names the visible symptom of an echo.  The
    log PSD of the composite signal is the log PSD of the wavelet plus a
    constant plus a cosine ripple whose PERIOD in frequency is set by the echo
    delay and whose DEPTH is set by the echo amplitude.  So an echo can be
    read straight off a log spectrum, and the cepstrum can be understood as
    the machine that measures the period of that ripple.

    Rangayyan & Krishnan (2024) Sec 4.8, eq (4.85), obtained "by substituting
    z = exp(jw) and taking the logarithm of both sides" of eq (4.84):

        log|Y(w)|^2 = log|H(w)|^2 + log[1 + a^2 + 2 a cos(w n0)]
                    = log|H(w)|^2 + log(1 + a^2)
                      + log[1 + (2a / (1 + a^2)) cos(w n0)].

    The book draws the conclusion directly: "it is now seen that the logarithm
    of the PSD of the signal has sinusoidal components (ripples) due to the
    presence of an echo.  The amplitudes and frequencies of the sinusoidal
    modulation or ripples are related to the amplitude a of the echo and its
    time delay n0."  Both the one-line and the three-term forms of eq (4.85)
    are computed, and their agreement is reported.

    Parameters
    ----------
    H : complex or sequence of complex
        H(w) at the evaluation frequencies.
    a : float
        Echo amplitude.
    n_0 : int
        Echo delay in samples.
    omega : float or sequence of float
        Radian frequency or frequencies.

    Returns
    -------
    RichResult with keys ``log_power``, ``wavelet_log_power``,
    ``echo_log_power``, ``dc_term`` (log(1 + a^2)), ``ripple`` (the
    log[1 + (2a/(1+a^2)) cos(w n0)] term), ``modulation_index``
    (2a/(1 + a^2)), ``ripple_period`` (2 pi / n0), ``decomposition_error``
    (agreement of the two forms of eq 4.85), ``omega``, ``a``, ``n_0``,
    ``method``.

    Raises
    ------
    ValueError
        If n_0 is not positive, H and omega differ in length, any |H| is
        zero (log 0), or 1 + a^2 + 2 a cos(w n0) is non-positive (which
        happens only at a = -1 or a = 1 with the right phase, where the two
        wavelet copies cancel exactly and the log PSD is -infinity).
    """
    a = float(a)
    n0 = int(n_0)
    if n0 <= 0:
        raise ValueError(f"n_0 must be a positive delay in samples, got {n_0!r}")
    ws = [float(omega)] if isinstance(omega, (int, float)) else [float(t) for t in omega]
    if not ws:
        raise ValueError("omega must not be empty")
    if isinstance(H, (int, float, complex)):
        hs = [complex(H)] * len(ws)
    else:
        hs = [complex(t) for t in H]
        if len(hs) != len(ws):
            raise ValueError(f"H has length {len(hs)} but omega has length {len(ws)}")
    for t in hs:
        if abs(t) == 0.0:
            raise ValueError("|H(w)| = 0 at some frequency; log|H|^2 is undefined there")
    hl = [log(abs(t) ** 2) for t in hs]
    el, rip = [], []
    dc = log(1.0 + a * a)
    mi = 2.0 * a / (1.0 + a * a)
    for w in ws:
        v = 1.0 + a * a + 2.0 * a * cos(w * n0)
        if v <= 0.0:
            raise ValueError(
                f"1 + a^2 + 2 a cos(w n_0) = {v} at omega={w}; the two wavelet "
                f"copies cancel exactly there and log|Y|^2 is -infinity"
            )
        el.append(log(v))
        r = 1.0 + mi * cos(w * n0)
        rip.append(log(r) if r > 0.0 else float("-inf"))
    lp = [hl[i] + el[i] for i in range(len(ws))]
    err = max(abs(el[i] - (dc + rip[i])) for i in range(len(ws)))
    return RichResult(
        payload={
            "log_power": lp,
            "wavelet_log_power": hl,
            "echo_log_power": el,
            "dc_term": dc,
            "ripple": rip,
            "modulation_index": mi,
            "ripple_period": 2.0 * pi / n0,
            "decomposition_error": err,
            "omega": ws,
            "a": a,
            "n_0": n0,
            "method": "Log power spectrum of a wavelet with an echo, "
                      "Rangayyan & Krishnan (2024) eq (4.85)",
        }
    )


rangayyan_ch4_log_power_spectrum_signal_echo = echologpsd  # pre-policy spelling


# -- wvdst: Wigner-Ville distribution for time-frequency analysis.
def wigner_ville(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nfft: int | None = None,
) -> DescriptiveResult:
    r"""Compute the Wigner-Ville distribution (WVD).

    The WVD is a bilinear time-frequency representation defined as:

    .. math::

        W_x(t, f) = \\int_{-\\infty}^{\\infty} x(t + \\tau/2) \\,
        x^*(t - \\tau/2) \\, e^{-j 2\\pi f \\tau} \\, d\\tau

    Provides perfect time-frequency resolution but suffers from
    cross-term interference for multi-component signals.

    Parameters
    ----------
    x : array-like
        1-D input signal (real or analytic).
    fs : float
        Sampling frequency in Hz (default 1.0).
    nfft : int or None
        FFT size for frequency axis.  Defaults to ``2 * len(x)``.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wvd`` (n_freq x n_time), ``times``,
        ``frequencies``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Claasen, T.A.C.M. & Mecklenbr\"auker, W.F.G. (1980). The Wigner
    distribution -- A tool for time-frequency signal analysis. *Philips
    J. Res.*, 35, 217--250.
    """
    from ._signal_core import hilbert

    x_raw = np.asarray(x, dtype=float).ravel()
    n = len(x_raw)
    if nfft is None:
        nfft = 2 * n

    xa = hilbert(x_raw)

    wvd = np.zeros((nfft, n))
    for t_idx in range(n):
        tau_max = min(t_idx, n - 1 - t_idx)
        kernel = np.zeros(nfft, dtype=complex)
        for tau in range(-tau_max, tau_max + 1):
            kernel[tau % nfft] = xa[t_idx + tau] * np.conj(xa[t_idx - tau])
        wvd[:, t_idx] = np.real(np.fft.fft(kernel, n=nfft))

    times = np.arange(n) / fs
    half = nfft // 2
    frequencies = np.arange(half) * (fs / (2 * nfft))
    wvd = wvd[:half, :]

    return DescriptiveResult(
        name="wigner_ville",
        value=float(n),
        extra={
            "wvd": wvd,
            "times": times,
            "frequencies": frequencies,
        },
    )


wvdst = wigner_ville


# compact alias per ledger/NAMING.md
wignerville = wigner_ville


_CHEATSHEET = [
    'hhtrf: Hilbert-Huang Transform (EMD + instantaneous frequency/amplitude).',
    'Complex demodulation: time-varying amplitude and phase at f0 -- Rangayyan eqs (5.16)-(5.19).',
    'Symmetric linear-phase biorthogonal 5/3 DWT via lifting; exact reconstruction.',
    "Exponential-kernel Cohen's-class TFD (Choi & Williams 1989); cuts cross terms.",
    'Wavelet scale-distribution width of a fibrillation waveform -- Rangayyan Sec 8.15.',
    'Continuous wavelet transform over scales -- Rangayyan eq (8.107).',
    "Cohen's-class TFD as a separably smoothed WVD -- Rangayyan eqs (8.124)-(8.127).",
    'Daubechies db1-db10 filter-bank taps with their orthonormality checks.',
    'Adaptive TFD: matching pursuit onto Gabor atoms, then per-atom WVD -- Rangayyan eq (9.15).',
    'Decimated dyadic DWT filter bank -- Rangayyan eqs (8.111)-(8.113).',
    'Noise-assisted ensemble EMD against mode mixing -- Rangayyan eq (9.13).',
    'EMD sifting into intrinsic mode functions -- Rangayyan Sec 9.4 steps 1-6.',
    'Sift one IMF and check both Sec 9.4 admissibility conditions.',
    'T-wave alternans amplitude from odd/even beat averages after EMD detrending -- Rangayyan Sec 9.2.3.',
    'Per-IMF energy and instantaneous-frequency descriptors of a fibrillation electrogram -- Rangayyan Sec 8.16.',
    'rgemg: EMG RMS envelope -- Rangayyan & Krishnan Sec 5.6.1, eq (5.24).',
    'Shannon entropy of the relative wavelet energies (Rosso et al. 2001).',
    'rgenv: Hilbert-transform envelope -- Rangayyan & Krishnan Sec 5.5.3.',
    'rgenvgm: Envelogram.',
    'Two-tap (Haar) orthogonal DWT -- normalised sums and differences, Parseval-exact.',
    "Time-frequency spectrum from EMD modes' instantaneous frequency -- Rangayyan eqs (9.8)-(9.12).",
    'Time-varying HRV LF/HF band powers via STFT of the RR tachogram -- Rangayyan Sec 8.12.',
    'Inverse STFT by weighted overlap-add; round-trips the forward STFT exactly.',
    'MRA: additive full-length detail bands that sum back to the signal -- Rangayyan eq (8.114).',
    'ECG-triggered synchronised averaging of PCG envelopes for S1/S2 intensity -- Rangayyan Sec 3.5.',
    'PPG motion-artifact removal by Daubechies wavelet shrinkage -- Rangayyan Sec 8.14.',
    'Scalogram |CWT|^2 energy density in time-scale -- Rangayyan Figure 8.29.',
    'EEG seizure marker: fluctuation intensity of db4 wavelet coefficients -- Rangayyan eq (8.132).',
    'STFT window length from wanted time/frequency resolution; flags the eq (8.10) conflict.',
    'Spectrogram |STFT|^2 over a sliding window -- Rangayyan eq (8.8).',
    'Shift-invariant undecimated wavelet transform (Nason & Silverman 1995).',
    'Shift-invariant (cycle-spinning) wavelet denoising -- no pseudo-Gibbs ringing.',
    'Variational mode decomposition into K band-limited modes (Dragomiretskiy & Zosso 2014).',
    'Detect transient structures as prominent CWT scalogram ridges -- Rangayyan Sec 8.8.',
    'Scale-by-scale wavelet cross-correlation of two signals (Whitcher et al. 2000).',
    'Wigner-Ville distribution of the analytic signal -- Rangayyan eq (8.123).',
    'Per-subband wavelet energy; sums exactly to the signal energy (Parseval).',
    'Mean/variance/energy/skewness/kurtosis of each DWT subband.',
    'Full wavelet packet tree with uniform subbands and a best-basis entropy cost.',
    'Wavelet denoising by soft/hard coefficient thresholding -- Rangayyan eqs (8.103)-(8.105).',
    'Unbiased scale-by-scale wavelet variance (Percival 1995); db1 = Allan variance.',
    'Two-impulse excitation delta(n) + a delta(n-n0) -- Rangayyan eq (4.74).',
    'Composite signal h(n) + a h(n-n0) -- Rangayyan eq (4.75).',
    'Y(z) = (1 + a z^-n0) H(z) -- Rangayyan eq (4.76).',
    'Y(w) = [1 + a exp(-j w n0)] H(w) -- Rangayyan eq (4.77).',
    'Y^(w) = H^(w) + log[1 + a exp(-j w n0)] -- Rangayyan eq (4.78), plus the eq (4.79) series.',
    'Complex cepstrum h^(n) plus a^k/k impulses at k*n0 -- Rangayyan eq (4.80).',
    '|Y(z)|^2 = |H(z)|^2 |1 + a z^-n0|^2 -- Rangayyan eq (4.84).',
    "log|Y(w)|^2 with the echo's cosine ripple -- Rangayyan eq (4.85).",
    'wvdst: Wigner-Ville distribution for time-frequency analysis.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)

# Pre-policy run-together spellings.  These were in the lazy
# map but not in the module, so morie.fn.<name> raised
# AttributeError.  Restored rather than dropped, because the
# map is the public flat namespace.
rangayyancwt = rangayyan_cwt  # pre-policy spelling, kept live
rangayyandwt = rangayyan_dwt  # pre-policy spelling, kept live
rangayyaneemd = rangayyan_eemd  # pre-policy spelling, kept live
rangayyanemd = rangayyan_emd  # pre-policy spelling, kept live
rangayyanistft = rangayyan_istft  # pre-policy spelling, kept live
rangayyanmra = rangayyan_mra  # pre-policy spelling, kept live
rangayyanswt = rangayyan_swt  # pre-policy spelling, kept live
rangayyanvmd = rangayyan_vmd  # pre-policy spelling, kept live
