# morie.fn -- bsaphys (rootcoder007/morie)
"""Physiological models and clinical applications: membrane and neuron models, PCG, EMG, EEG, VAG, respiratory and sleep signal analysis.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 38
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2, cos, exp, fsum, hypot, log, pi, sin, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._rgcore import aslist, aslistc
from ._rgcore import aslist, gridint
from ._richresult import RichResult
from ._sci_core import CubicSpline
from .bsacorr import rangayyan_welch_psd

__all__ = [
    'emd',
    'apwave',
    'rangayyan_action_potential',
    'bidomain',
    'rangayyan_cardiac_elecphys',
    'cadacou',
    'rangayyan_coronary_ad',
    'corsound',
    'rangayyan_coronary_sound',
    'infantcry',
    'rangayyan_infant_cry',
    'eggfeat',
    'rangayyan_egg',
    'rangayyan_heart_elasticity',
    'engcap',
    'rangayyan_eng',
    'seizdet',
    'rangayyan_epilepsy_detect',
    'erpfeat',
    'rangayyan_erp_features',
    'erders',
    'rangayyan_feature_extract_bci',
    'cadspec',
    'rangayyan_freq_domain_feat',
    'ghk',
    'rangayyan_goldman_eqn',
    'hhgate',
    'rangayyan_hh_gating',
    'hhmodel',
    'rangayyan_hodgkin_huxley',
    'fhn',
    'rangayyan_fitzhugh_nagumo',
    'rcmemb',
    'rangayyan_membrane_potential',
    'vagclean',
    'rangayyan_muscle_artifact',
    'muapmodel',
    'rangayyan_muap',
    'murmspec',
    'rangayyan_murmur_analysis',
    'nernst',
    'rangayyan_nernst_potential',
    'oaefeat',
    'rangayyan_oae',
    'pdmonitor',
    'rangayyan_parkinson_multimodal',
    'pcgeeg',
    'rangayyan_pcg_eeg_coupling',
    'murmdet',
    'rangayyan_pcg_murmur_detect',
    'psgstage',
    'rangayyan_polysomnography',
    'ieistats',
    'rangayyan_point_process',
    'valvepcg',
    'rangayyan_prosthetic_valve',
    'respfeat',
    'rangayyan_respiration_features',
    'respsound',
    'rangayyan_respiratory_sound',
    'apneadet',
    'rangayyan_sleep_apnea_detect',
    'speechfeat',
    'rangayyan_speech_features',
    'vagfeat',
    'rangayyan_vag_analysis',
    'vagknee',
    'rangayyan_vag_knee_cartilage',
    'deltadecomp',
    'rangayyan_ch3_signal_as_delta_decomposition',
    'clogprod',
    'rangayyan_ch4_complex_log_of_product',
    'clogpz',
    'rangayyan_ch4_complex_log_x_z',
    'rangayyanegg',
    'rangayyaneng',
    'rangayyanmuap',
    'rangayyanoae',
]

# -- shared helpers for the biophysical signal-generation blocks --------------
# Pure standard library: no numpy, no scipy.  Every helper here is used by
# several of the application blocks below (spectra, band powers, peak
# picking, autocorrelation, linear prediction).

_BSA_R_GAS = 8.314462618        # J/(mol K), CODATA 2018
_BSA_FARADAY = 96485.33212      # C/mol,     CODATA 2018


def _bsafft(re, im):
    """In-place iterative radix-2 Cooley-Tukey FFT; len must be a power of 2."""
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= n:
        ang = -2.0 * pi / length
        wr, wi = cos(ang), sin(ang)
        for i in range(0, n, length):
            cr, ci = 1.0, 0.0
            half = length >> 1
            for k in range(i, i + half):
                ur, ui = re[k], im[k]
                vr = re[k + half] * cr - im[k + half] * ci
                vi = re[k + half] * ci + im[k + half] * cr
                re[k], im[k] = ur + vr, ui + vi
                re[k + half], im[k + half] = ur - vr, ui - vi
                cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
        length <<= 1
    return re, im


def _bsapsd(x, fs, detrend=True):
    """Periodogram of *x* sampled at *fs* Hz with a Hann window.

    Returns (freqs_hz, power) over 0 .. fs/2.  Power is in signal-units^2
    per bin, normalised by the window energy so that band powers are
    comparable across window lengths.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 4:
        raise ValueError("need at least 4 samples for a spectrum")
    if not (fs > 0):
        raise ValueError("fs must be positive (Hz)")
    if detrend:
        mu = fsum(xs) / n
        xs = [v - mu for v in xs]
    w = [0.5 - 0.5 * cos(2.0 * pi * i / (n - 1)) for i in range(n)]
    wsum2 = fsum(v * v for v in w)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    re = [xs[i] * w[i] for i in range(n)] + [0.0] * (nfft - n)
    im = [0.0] * nfft
    _bsafft(re, im)
    m = nfft // 2 + 1
    freqs = [k * fs / nfft for k in range(m)]
    power = [(re[k] * re[k] + im[k] * im[k]) / wsum2 for k in range(m)]
    return freqs, power


def _bsabandpow(freqs, power, lo, hi):
    """Total power in the half-open band [lo, hi) Hz."""
    return fsum(p for f, p in zip(freqs, power) if lo <= f < hi)


def _bsapeaks(freqs, power, count=3, minsep=0.0):
    """Local maxima of *power*, strongest first, at least *minsep* Hz apart."""
    cand = []
    for k in range(1, len(power) - 1):
        if power[k] > power[k - 1] and power[k] >= power[k + 1]:
            cand.append((power[k], freqs[k]))
    cand.sort(reverse=True)
    out = []
    for p, f in cand:
        if all(abs(f - g) >= minsep for _, g in out):
            out.append((p, f))
        if len(out) >= count:
            break
    return [(f, p) for p, f in out]


def _bsaacf(x, maxlag):
    """Biased autocorrelation of the mean-removed *x*, lags 0..maxlag."""
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 samples")
    mu = fsum(xs) / n
    xs = [v - mu for v in xs]
    maxlag = min(maxlag, n - 1)
    return [fsum(xs[i] * xs[i + k] for i in range(n - k)) / n
            for k in range(maxlag + 1)]


def _bsalpc(x, order):
    """Levinson-Durbin linear prediction; returns (a[1..p], error_power).

    The all-pole model is  x[n] = -sum_k a[k] x[n-k] + e[n].
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    r = _bsaacf(x, order)
    if r[0] <= 0.0:
        raise ValueError("signal is constant; no linear-prediction model")
    a = [0.0] * (order + 1)
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + fsum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / e
        newa = list(a)
        newa[i] = k
        for j in range(1, i):
            newa[j] = a[j] + k * a[i - j]
        a = newa
        e *= (1.0 - k * k)
        if e <= 0.0:
            e = 1e-30
            break
    return a[1:], e


def _bsalpcspec(a, fs, npts=1024):
    """Magnitude-squared response of the all-pole LPC filter, 0..fs/2 Hz."""
    freqs, power = [], []
    for i in range(npts):
        f = 0.5 * fs * i / (npts - 1)
        w = 2.0 * pi * f / fs
        dr, di = 1.0, 0.0
        for k, ak in enumerate(a, start=1):
            dr += ak * cos(-w * k)
            di += ak * sin(-w * k)
        d2 = dr * dr + di * di
        freqs.append(f)
        power.append(1.0 / d2 if d2 > 1e-30 else 1e30)
    return freqs, power


def _bsapsdmom(freqs, power):
    """Moments of a PSD treated as a density, Rangayyan (2024) Section 6.4.1.

    Total power Ep is eq. (6.32), mean frequency eq. (6.34), median
    frequency eq. (6.35), variance fm2 eq. (6.37), spectral skewness
    eq. (6.38) with eq. (6.40), spectral kurtosis eq. (6.41) with
    eq. (6.43).  All frequencies in Hz, moments in Hz^k.
    """
    Ep = fsum(power)
    if Ep <= 0.0:
        raise ValueError("PSD has zero total power; nothing to characterise")
    fmean = fsum(f * p for f, p in zip(freqs, power)) / Ep
    run, fmed = 0.0, freqs[-1]
    for f, p in zip(freqs, power):
        run += p
        if run >= 0.5 * Ep:
            fmed = f
            break
    fm2 = fsum((f - fmean) ** 2 * p for f, p in zip(freqs, power)) / Ep
    fm3 = fsum((f - fmean) ** 3 * p for f, p in zip(freqs, power)) / Ep
    fm4 = fsum((f - fmean) ** 4 * p for f, p in zip(freqs, power)) / Ep
    sk = fm3 / fm2 ** 1.5 if fm2 > 0.0 else 0.0
    ku = fm4 / fm2 ** 2 if fm2 > 0.0 else 0.0
    return {"total_power": Ep, "mean_freq_hz": fmean, "median_freq_hz": fmed,
            "fm2_hz2": fm2, "spread_hz": sqrt(fm2), "spectral_skewness": sk,
            "spectral_kurtosis": ku}


def _bsaqfactor(freqs, power, fpeak):
    """-3 dB bandwidth and quality factor Q = f_peak / bandwidth of the peak
    at *fpeak*, as used in Rangayyan (2024) Section 6.4.2 (after Durand et
    al., Section 6.5).  Returns (bandwidth_hz, Q) or (None, None)."""
    if not freqs:
        return None, None
    i = min(range(len(freqs)), key=lambda k: abs(freqs[k] - fpeak))
    half = power[i] / 2.0
    lo = hi = None
    for k in range(i, -1, -1):
        if power[k] <= half:
            lo = freqs[k]
            break
    for k in range(i, len(freqs)):
        if power[k] <= half:
            hi = freqs[k]
            break
    if lo is None or hi is None or hi <= lo:
        return None, None
    bw = hi - lo
    return bw, (freqs[i] / bw if bw > 0.0 else None)


def _bsahjorth(x):
    """Hjorth activity, mobility and form factor, Rangayyan (2024)
    eqs. (5.25) and (5.26), Section 5.6.4.  Derivatives are first
    differences; the form factor of a pure sinusoid is 1."""
    xs = [float(v) for v in x]
    if len(xs) < 4:
        raise ValueError("need at least 4 samples for Hjorth parameters")

    def var(v):
        n = len(v)
        mu = fsum(v) / n
        return fsum((a - mu) ** 2 for a in v) / n

    d1 = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    a0, a1, a2 = var(xs), var(d1), var(d2)
    if a0 <= 0.0 or a1 <= 0.0:
        raise ValueError("signal is constant; Hjorth parameters undefined")
    mob = sqrt(a1 / a0)
    return {"activity": a0, "mobility": mob,
            "form_factor": sqrt(a2 / a1) / mob}


def _bsarms(x):
    xs = [float(v) for v in x]
    if not xs:
        raise ValueError("empty signal")
    return sqrt(fsum(v * v for v in xs) / len(xs))


def _bsamoments(x):
    """(mean, variance, skewness, kurtosis) -- kurtosis is the raw fourth
    standardised moment (3.0 for a Gaussian), as used in Rangayyan (2024)
    Section 5.12.3 for the screening of VAG signals."""
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 samples")
    mu = fsum(xs) / n
    d = [v - mu for v in xs]
    m2 = fsum(v * v for v in d) / n
    if m2 <= 0.0:
        return mu, 0.0, 0.0, 0.0
    s = sqrt(m2)
    m3 = fsum(v ** 3 for v in d) / n
    m4 = fsum(v ** 4 for v in d) / n
    return mu, m2, m3 / s ** 3, m4 / m2 ** 2


def _bsaenvelope(x, fs, win_s):
    """Short-time RMS envelope, non-overlapping windows of *win_s* seconds."""
    xs = [float(v) for v in x]
    w = max(1, int(round(win_s * fs)))
    return [_bsarms(xs[i:i + w]) for i in range(0, len(xs) - w + 1, w)], w / fs


def _bsahhrates(v):
    """Hodgkin-Huxley (1952) alpha/beta rate constants, in 1/ms, for the
    membrane potential *v* in mV on the modern sign convention with a
    resting potential of -65 mV.

    Hodgkin AL, Huxley AF, "A quantitative description of membrane current
    and its application to conduction and excitation in nerve", Journal of
    Physiology 117(4):500-544, 1952 (Table 3 / eqns 12, 13, 16, 17, 20, 21),
    re-expressed for absolute membrane potential.  Removable singularities
    at v = -55 mV and v = -40 mV are replaced by their limits.
    """
    d = v + 55.0
    am = None
    an = 0.1 if abs(d) < 1e-6 else 0.01 * d / (1.0 - exp(-d / 10.0))
    bn = 0.125 * exp(-(v + 65.0) / 80.0)
    d = v + 40.0
    am = 1.0 if abs(d) < 1e-6 else 0.1 * d / (1.0 - exp(-d / 10.0))
    bm = 4.0 * exp(-(v + 65.0) / 18.0)
    ah = 0.07 * exp(-(v + 65.0) / 20.0)
    bh = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
    return am, bm, ah, bh, an, bn



# -- emdsg: Empirical Mode Decomposition (standalone).
def _count_zero_crossings(x: np.ndarray) -> int:
    """Count zero crossings in signal *x*."""
    return int(np.sum(np.diff(np.sign(x)) != 0))


def _count_extrema(x: np.ndarray) -> int:
    """Count total number of local extrema."""
    max_idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
    min_idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]
    return len(max_idx) + len(min_idx)


def emd(
    x: np.ndarray,
    *,
    max_imfs: int = 12,
    max_sift_iter: int = 300,
    sd_threshold: float = 0.05,
) -> DescriptiveResult:
    r"""Empirical Mode Decomposition.

    Decomposes a signal into a set of Intrinsic Mode Functions (IMFs)
    through an iterative sifting process.  Each IMF satisfies two
    conditions:

    1. The number of extrema and zero crossings differ by at most one.
    2. The local mean of the upper and lower envelopes is zero.

    The sifting stopping criterion uses the normalized squared
    difference:

    .. math::

        SD = \\frac{\\sum_t |h_{k-1}(t) - h_k(t)|^2}
             {\\sum_t h_{k-1}^2(t)}

    Parameters
    ----------
    x : array-like
        1-D input signal.
    max_imfs : int
        Maximum number of IMFs to extract (default 12).
    max_sift_iter : int
        Maximum sifting iterations per IMF (default 300).
    sd_threshold : float
        Sifting convergence threshold (default 0.05).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs`` (list of arrays), ``residue``,
        ``n_imfs``, ``sift_counts`` (iterations per IMF),
        ``is_imf`` (bool per IMF -- satisfies IMF conditions).

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
    residue = x.copy()
    imfs = []
    sift_counts = []

    for _ in range(max_imfs):
        h = residue.copy()
        n_sifts = 0

        for s in range(max_sift_iter):
            n_sifts = s + 1
            t = np.arange(len(h))
            max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
            min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1

            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
            lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
            mean_env = (upper + lower) / 2.0

            prev = h.copy()
            h = h - mean_env

            sd = np.sum((prev - h) ** 2) / (np.sum(prev**2) + 1e-12)
            if sd < sd_threshold:
                break

        if np.max(np.abs(h)) < 1e-10:
            break

        imfs.append(h)
        sift_counts.append(n_sifts)
        residue = residue - h

        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    is_imf = []
    for imf in imfs:
        zc = _count_zero_crossings(imf)
        ne = _count_extrema(imf)
        is_imf.append(abs(ne - zc) <= 1)

    return DescriptiveResult(
        name="emd",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "n_imfs": len(imfs),
            "sift_counts": sift_counts,
            "is_imf": is_imf,
        },
    )


emdsg = emd


# -- rgap: Idealized action potential waveform model (depolarization/repolarization).
def apwave(t, v_rest=-70.0, v_peak=30.0, t_rise=0.5, t_fall=1.0, t_onset=0.0,
           v_undershoot=None, t_recover=None):
    """Idealised action-potential waveform: linear rise, exponential fall.

    Rangayyan (2024) Section 1.2.2 ("The action potential of a neuron")
    describes the
    neuronal action potential as an upstroke of depolarisation followed by
    a return to the resting potential via repolarisation, and Figure 1.11
    reproduces the first recorded axonal action potential of Hodgkin and
    Huxley (1939) with a vertical axis running from -70 to +40 mV.  The
    book gives no closed-form waveform equation; this block implements the
    conventional ramp-and-decay template consistent with that description:

        V(t) = V_rest                                       t < t_onset
             = V_rest + (V_peak - V_rest) (t - t_onset)/t_rise
                                             t_onset <= t < t_onset+t_rise
             = V_final + (V_peak - V_final) exp(-(t - t_peak)/t_fall)
                                             t >= t_onset + t_rise

    where V_final is V_rest, or the after-hyperpolarisation level if
    ``v_undershoot`` is given, in which case the trace then relaxes from
    the undershoot back to V_rest with time constant ``t_recover``.

    WHY: a template like this is not a mechanism -- for that use
    ``hhmodel`` -- but it is what is wanted when a synthetic spike train,
    an ENG or an MUAP has to be built from a known, exactly reproducible
    waveform, because every feature (peak, width, decay) is a parameter
    rather than an emergent property.

    Parameters
    ----------
    t : array-like
        Time points in milliseconds (ms).
    v_rest, v_peak : float
        Resting and peak potentials in millivolts (mV).  v_peak must
        exceed v_rest.
    t_rise : float
        Depolarisation (upstroke) duration in milliseconds (ms), positive.
    t_fall : float
        Repolarisation time constant in milliseconds (ms), positive.
    t_onset : float
        Time of the start of the upstroke, in milliseconds (ms).
    v_undershoot : float or None
        After-hyperpolarisation level in millivolts (mV); must be below
        v_rest when given.  ``None`` disables the undershoot.
    t_recover : float or None
        Time constant in milliseconds (ms) for the recovery from the
        undershoot back to rest.  Required when v_undershoot is given.

    Returns
    -------
    RichResult
        ``t_ms``, ``V_mV`` waveform; ``amplitude_mV`` = v_peak - v_rest;
        ``peak_time_ms``; ``width_half_ms`` -- the full width at half the
        peak amplitude, measured on the returned samples, or None if the
        sampled span does not cover it.
    """
    ts = [float(v) for v in aslist(t)]
    if not ts:
        raise ValueError("t must contain at least one time point (ms)")
    v_rest, v_peak = float(v_rest), float(v_peak)
    if v_peak <= v_rest:
        raise ValueError("v_peak must exceed v_rest (mV)")
    t_rise, t_fall = float(t_rise), float(t_fall)
    if t_rise <= 0.0:
        raise ValueError("t_rise must be positive (ms)")
    if t_fall <= 0.0:
        raise ValueError("t_fall must be positive (ms)")
    t_onset = float(t_onset)
    if v_undershoot is None:
        v_us = None
    else:
        v_us = float(v_undershoot)
        if v_us >= v_rest:
            raise ValueError("v_undershoot must lie below v_rest (mV)")
        if t_recover is None or float(t_recover) <= 0.0:
            raise ValueError("t_recover must be positive (ms) when v_undershoot is set")
    t_peak = t_onset + t_rise
    v_floor = v_rest if v_us is None else v_us
    Vs = []
    for ti in ts:
        if ti < t_onset:
            v = v_rest
        elif ti < t_peak:
            v = v_rest + (v_peak - v_rest) * (ti - t_onset) / t_rise
        else:
            v = v_floor + (v_peak - v_floor) * exp(-(ti - t_peak) / t_fall)
            if v_us is not None:
                v += (v_rest - v_us) * (1.0 - exp(-(ti - t_peak) / float(t_recover)))
        Vs.append(v)
    amp = v_peak - v_rest
    half = v_rest + 0.5 * amp
    above = [ts[i] for i in range(len(ts)) if Vs[i] >= half]
    width = (above[-1] - above[0]) if len(above) > 1 else None
    return RichResult(payload={
        "t_ms": ts, "V_mV": Vs,
        "amplitude_mV": amp, "peak_time_ms": t_peak,
        "v_rest_mV": v_rest, "v_peak_mV": v_peak,
        "t_rise_ms": t_rise, "t_fall_ms": t_fall,
        "width_half_ms": width,
        "units": {"V": "mV", "t": "ms"},
        "method": "Idealised ramp-and-decay action potential; Rangayyan (2024) Section 1.2.2 is descriptive and gives no waveform equation",
    })


rangayyan_action_potential = apwave  # pre-policy spelling


# -- rgcardep: Cardiac electrophysiology tissue/organ-level model.
def bidomain(n_nodes=100, dx_cm=0.02, duration_ms=60.0, dt_ms=0.005,
             sigma_i=1.0, sigma_e=2.0, C_m=1.0, Sv=1000.0,
             I_ion=None, stim_nodes=5, I_stim=50.0, stim_ms=1.0,
             v_rest=-85.0, v_peak=20.0, I_ion_peak=10.0, threshold_frac=0.25):
    """One-dimensional monodomain propagation with the bidomain extracellular field.

    Rangayyan (2024) Section 7.8.2 ("Electrophysiological modeling at the
    tissue and organ levels") gives the monodomain model as

        dV_m/dt = div(D grad V_m) - (I_ion + I_applied)/C_m,     eq. (7.143)
        D = G / (S_v C_m)                                        eq. (7.144)
        n . (D grad V_m) = 0            (no-flux boundary)       eq. (7.145)

    and the bidomain model as

        V_m = phi_i - phi_e                                      eq. (7.146)
        div((D_i + D_e) grad phi_e) = -div(D_i grad V_m)         eq. (7.147)
        div(D_i grad V_m) + div(D_i grad phi_e) = -S_v I_m        eq. (7.148)
        I_m = C_m dV_m/dt + I_ion                                eq. (7.149)

    This block solves those equations on a one-dimensional cable: V_m is
    advanced with explicit Euler using eq. (7.143) and the no-flux
    boundary of eq. (7.145); at each output time the extracellular
    potential phi_e is obtained by solving the elliptic eq. (7.147) with a
    tridiagonal (Thomas) solve, gauged to zero mean; and I_m is evaluated
    from eq. (7.149).  The intracellular potential follows from
    eq. (7.146).

    WHY: the monodomain model reduces to a single reaction-diffusion
    equation only because it assumes the intracellular and extracellular
    anisotropy ratios are equal.  That is enough to get propagation, but
    it cannot represent an externally injected current, because such a
    current lives in the extracellular domain.  The bidomain form keeps
    the two domains separate, and that is exactly why -- as the book notes
    -- it is the model used for defibrillation studies, at the cost of a
    much heavier computation.

    Parameters
    ----------
    n_nodes : int
        Number of nodes along the cable; >= 5.
    dx_cm : float
        Node spacing in centimetres (cm); positive.  0.02 cm = 200 um is a
        typical cardiac tissue discretisation.
    duration_ms, dt_ms : float
        Simulated time and explicit-Euler step, in milliseconds (ms).
        The step is checked against the diffusion stability limit
        dt <= dx^2 / (2 D) and a ValueError is raised if it is violated,
        because an unstable explicit run returns numbers that look like
        results.
    sigma_i, sigma_e : float
        Intracellular and extracellular conductivities in millisiemens
        per centimetre (mS/cm); both positive.
    C_m : float
        Membrane capacitance in microfarads per square centimetre
        (uF/cm^2); positive.
    Sv : float
        Surface-to-volume ratio in reciprocal centimetres (1/cm);
        positive.  The monodomain diffusion coefficient is
        D = sigma_harmonic / (Sv C_m) per eq. (7.144), with
        sigma_harmonic = sigma_i sigma_e / (sigma_i + sigma_e), the
        standard bulk conductivity of two conductors in series.
    I_ion : callable or None
        I_ion(V_mV) -> current density in uA/cm^2, the cell model.
        ``None`` installs a cubic excitable current normalised to
        v_rest and v_peak; the book states only that I_ion is "the ion
        channel current specified by the cell model utilized", and does
        not prescribe one.  The default cubic is of the FitzHugh (1961)
        excitable type, without a recovery variable, so it propagates a
        front rather than a full action potential.
    stim_nodes : int
        Number of leftmost nodes receiving the stimulus; >= 1.
    I_stim : float
        Applied stimulus current density in uA/cm^2 (positive
        depolarises).
    stim_ms : float
        Stimulus duration in milliseconds (ms).
    v_rest, v_peak : float
        Resting and plateau potentials in millivolts (mV) used by the
        default cubic current; v_peak must exceed v_rest.
    I_ion_peak : float
        Peak magnitude of the default cubic current over the interval
        [v_rest, v_peak], in uA/cm^2; it scales both the upstroke rate and
        the conduction velocity.  Ignored when I_ion is supplied.
    threshold_frac : float
        Position of the excitation threshold of the default cubic within
        [v_rest, v_peak], dimensionless, strictly between 0 and 0.5.  It
        must NOT be 0.5: a cubic whose threshold sits exactly at the
        midpoint has a stationary front and nothing propagates, since the
        Nagumo front velocity is proportional to (v_rest + v_peak - 2
        threshold).  0.25 is the usual choice.

    Returns
    -------
    RichResult
        ``x_cm`` node positions; ``Vm_mV``, ``phi_e_mV``, ``phi_i_mV``,
        ``Im_uA_cm2`` at the final time; ``activation_ms`` -- the time
        each node first crossed the midpoint potential, or None if it
        never did; ``cv_cm_per_ms`` -- conduction velocity from a
        least-squares fit of position against activation time over the
        activated nodes; ``D_cm2_per_ms`` -- eq. (7.144).
    """
    n = int(n_nodes)
    if n < 5:
        raise ValueError("n_nodes must be at least 5")
    dx, dt = float(dx_cm), float(dt_ms)
    if dx <= 0.0:
        raise ValueError("dx_cm must be positive (cm)")
    if dt <= 0.0:
        raise ValueError("dt_ms must be positive (ms)")
    dur = float(duration_ms)
    if dur <= 0.0:
        raise ValueError("duration_ms must be positive (ms)")
    si, se = float(sigma_i), float(sigma_e)
    if si <= 0.0 or se <= 0.0:
        raise ValueError("sigma_i and sigma_e must be positive (mS/cm)")
    C_m, Sv = float(C_m), float(Sv)
    if C_m <= 0.0:
        raise ValueError("C_m must be positive (uF/cm^2)")
    if Sv <= 0.0:
        raise ValueError("Sv must be positive (1/cm)")
    v_rest, v_peak = float(v_rest), float(v_peak)
    if v_peak <= v_rest:
        raise ValueError("v_peak must exceed v_rest (mV)")
    ns = int(stim_nodes)
    if not 1 <= ns <= n:
        raise ValueError("stim_nodes must be between 1 and n_nodes")
    # mS/cm divided by (1/cm * uF/cm^2) gives cm^2/ms
    sigma_bulk = si * se / (si + se)
    D = sigma_bulk / (Sv * C_m)                      # eq. (7.144), cm^2/ms
    lim = dx * dx / (2.0 * D)
    if dt > lim:
        raise ValueError("dt_ms=%g exceeds the explicit stability limit %g ms "
                         "for D=%g cm^2/ms and dx=%g cm" % (dt, lim, D, dx))
    tf = float(threshold_frac)
    if not 0.0 < tf < 0.5:
        raise ValueError("threshold_frac must be strictly between 0 and 0.5; "
                         "0.5 gives a stationary front that never propagates")
    amp = v_peak - v_rest
    thresh = v_rest + tf * amp
    if I_ion is None:
        ipk = float(I_ion_peak)
        if ipk <= 0.0:
            raise ValueError("I_ion_peak must be positive (uA/cm^2)")
        raw = max(abs((v_rest + amp * k / 500.0 - v_rest)
                      * (v_rest + amp * k / 500.0 - thresh)
                      * (v_rest + amp * k / 500.0 - v_peak))
                  for k in range(501))
        gain = ipk / raw

        def I_ion(v):
            # cubic excitable current, uA/cm^2; zero at rest, at the
            # threshold, and at the plateau v_peak
            return gain * (v - v_rest) * (v - thresh) * (v - v_peak)
    elif not callable(I_ion):
        raise ValueError("I_ion must be callable: I_ion(V_mV) -> uA/cm^2")
    V = [v_rest] * n
    act = [None] * n
    nsteps = int(round(dur / dt))
    stim_steps = int(round(float(stim_ms) / dt))
    I_stim = float(I_stim)
    for step in range(nsteps):
        lap = []
        for i in range(n):
            lo = V[1] if i == 0 else V[i - 1]        # eq. (7.145), no flux
            hi = V[n - 2] if i == n - 1 else V[i + 1]
            lap.append((lo - 2.0 * V[i] + hi) / (dx * dx))
        newV, dVdt = [], []
        for i in range(n):
            iapp = -I_stim if (step < stim_steps and i < ns) else 0.0
            rate = D * lap[i] - (I_ion(V[i]) + iapp) / C_m
            dVdt.append(rate)
            newV.append(V[i] + dt * rate)
        V = newV
        t = (step + 1) * dt
        for i in range(n):
            if act[i] is None and V[i] >= thresh:
                act[i] = t
    # eq. (7.147): d/dx((Di+De) dphi_e/dx) = -d/dx(Di dVm/dx), 1-D uniform
    # coefficients -> (si+se) L phi_e = -si L Vm  with L the Laplacian.
    rhs = []
    for i in range(n):
        lo = V[1] if i == 0 else V[i - 1]
        hi = V[n - 2] if i == n - 1 else V[i + 1]
        rhs.append(-si * (lo - 2.0 * V[i] + hi) / (dx * dx))
    # Thomas algorithm on (si+se)/dx^2 * tridiag(1, -2, 1), Neumann + zero
    # mean; pin node 0 to zero, then remove the mean afterwards.
    k = (si + se) / (dx * dx)
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = list(rhs)
    for i in range(n):
        if i == 0:
            b[i], c[i], d[i] = 1.0, 0.0, 0.0        # gauge pin
        elif i == n - 1:
            a[i], b[i] = k, -k
        else:
            a[i], b[i], c[i] = k, -2.0 * k, k
    for i in range(1, n):
        if b[i - 1] == 0.0:
            raise ValueError("singular extracellular system; check conductivities")
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    phie = [0.0] * n
    if b[n - 1] == 0.0:
        raise ValueError("singular extracellular system; check conductivities")
    phie[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        phie[i] = (d[i] - c[i] * phie[i + 1]) / b[i]
    mu = fsum(phie) / n
    phie = [v - mu for v in phie]
    phii = [V[i] + phie[i] for i in range(n)]        # eq. (7.146)
    # eq. (7.149): I_m = C_m dV_m/dt + I_ion, using the last computed rate
    Im = [C_m * dVdt[i] + I_ion(V[i]) for i in range(n)]
    xs = [i * dx for i in range(n)]
    hit = [(xs[i], act[i]) for i in range(n) if act[i] is not None]
    cv = None
    if len(hit) > 2:
        tt = [h[1] for h in hit]
        xx = [h[0] for h in hit]
        mt = fsum(tt) / len(tt)
        mx = fsum(xx) / len(xx)
        den = fsum((v - mt) ** 2 for v in tt)
        if den > 0.0:
            cv = fsum((tt[i] - mt) * (xx[i] - mx) for i in range(len(tt))) / den
    return RichResult(payload={
        "x_cm": xs, "Vm_mV": V, "phi_e_mV": phie, "phi_i_mV": phii,
        "Im_uA_cm2": Im,
        "activation_ms": act,
        "n_activated": len(hit),
        "cv_cm_per_ms": cv,
        "D_cm2_per_ms": D, "sigma_bulk_mS_cm": sigma_bulk,
        "dt_ms": dt, "dx_cm": dx, "stability_limit_ms": lim,
        "units": {"V": "mV", "x": "cm", "t": "ms", "sigma": "mS/cm",
                  "C_m": "uF/cm^2", "Sv": "1/cm", "I": "uA/cm^2",
                  "D": "cm^2/ms", "cv": "cm/ms"},
        "method": "Rangayyan (2024) eqs. (7.143)-(7.149), Section 7.8.2, monodomain propagation with the bidomain extracellular field, 1-D",
    })


rangayyan_cardiac_elecphys = bidomain  # pre-policy spelling


# -- rgcorad: Coronary artery disease detection from acoustic signals.
def cadacou(coronary_sound, fs, order=8, hf_band=(300.0, 900.0),
            ref_band=(50.0, 300.0)):
    """Coronary artery disease detection from a diastolic acoustic segment.

    Rangayyan (2024) Section 7.10 ("Application: Coronary Artery Disease")
    applies parametric (autoregressive) spectral modelling to sounds
    recorded from the chest, the underlying generative model being that of
    Section 7.7.2 for turbulent flow past a stenosis, eq. (7.136).  The
    decision measures used here are the spectral power ratio of
    eq. (6.44), Section 6.4.2, and the PSD moments of Section 6.4.1.
    The AR model itself is the all-pole model of Section 7.5
    ("Autoregressive or All-pole Modeling"), fitted by the Levinson-Durbin
    recursion.

    WHY: the diastolic interval is acoustically quiet in a healthy heart,
    because the valves are shut and flow through the coronaries is smooth.
    A stenosis makes that flow turbulent, and the resulting wideband
    murmur is small, buried in noise, and best characterised by an AR
    model that concentrates it into a few poles rather than by a raw
    periodogram.  A high-frequency-to-reference power ratio then reads out
    directly the upward frequency shift that eq. (7.136) predicts as the
    stenosis tightens.

    Parameters
    ----------
    coronary_sound : array-like
        Diastolic segment of the recorded chest-wall sound, in arbitrary
        amplitude units.  At least 4 * order samples.
    fs : float
        Sampling rate in hertz (Hz); positive.
    order : int
        Order of the all-pole model; >= 2.  Values of 6 to 12 are usual
        for diastolic heart-sound segments.
    hf_band : (float, float)
        High-frequency band in hertz (Hz) expected to carry the turbulent
        murmur energy.
    ref_band : (float, float)
        Reference band in hertz (Hz) common to normal and abnormal
        segments, playing the role of the "constant area" of eq. (6.45).

    Returns
    -------
    RichResult
        ``power_ratio`` -- HF band power divided by reference band power,
        computed on the AR spectrum (dimensionless); the larger it is, the
        more the energy has shifted upward;
        ``hf_fraction`` -- HF band power as a fraction of total AR-spectrum
        power, eq. (6.44);
        ``ar_coeffs`` and ``prediction_error``;
        ``ar_peaks_hz`` -- the resonance frequencies of the AR model,
        strongest first, with their powers;
        ``mean_freq_hz``, ``median_freq_hz``, ``spread_hz``,
        ``spectral_skewness``, ``spectral_kurtosis`` of the AR spectrum;
        ``freq_hz`` and ``ar_psd``.
    """
    xs = [float(v) for v in aslist(coronary_sound)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    order = int(order)
    if order < 2:
        raise ValueError("order must be at least 2")
    if len(xs) < 4 * order:
        raise ValueError("need at least 4*order samples to fit the AR model")
    for lo, hi in (hf_band, ref_band):
        if float(hi) <= float(lo):
            raise ValueError("each band must have hi > lo (Hz)")
        if float(hi) > fs / 2.0:
            raise ValueError("band upper edge exceeds the Nyquist frequency")
    a, err = _bsalpc(xs, order)
    freqs, psd = _bsalpcspec(a, fs)
    psd = [err * v for v in psd]
    hf = _bsabandpow(freqs, psd, float(hf_band[0]), float(hf_band[1]))
    rf = _bsabandpow(freqs, psd, float(ref_band[0]), float(ref_band[1]))
    if rf <= 0.0:
        raise ValueError("reference band carries no power; choose another band")
    mom = _bsapsdmom(freqs, psd)
    out = dict(mom)
    out.update({
        "power_ratio": hf / rf,
        "hf_fraction": hf / mom["total_power"],
        "hf_band_hz": (float(hf_band[0]), float(hf_band[1])),
        "ref_band_hz": (float(ref_band[0]), float(ref_band[1])),
        "ar_coeffs": a, "prediction_error": err, "order": order,
        "ar_peaks_hz": _bsapeaks(freqs, psd, count=4, minsep=fs / 200.0),
        "freq_hz": freqs, "ar_psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "ratios": "dimensionless"},
        "method": "Rangayyan (2024) Section 7.10 with AR modelling of Section 7.5 and the spectral power ratio of eq. (6.44), Section 6.4.2",
    })
    return RichResult(payload=out)


rangayyan_coronary_ad = cadacou  # pre-policy spelling


# -- rgcorart: Coronary artery sound generation model (turbulent flow).
def corsound(diameter, flow_velocity, stenosis_pct=0.0, p2max=1.0,
             freqs=None, nu=3.5e-6):
    """Turbulent-flow sound spectrum of a stenosed coronary artery segment.

    Rangayyan (2024) Section 7.7.2 ("Modeling sound generation in coronary
    arteries") reproduces the model of Wang et al. (IEEE Transactions on
    Biomedical Engineering 37(11):1087-1094, 1990).  The wideband spectrum
    of the sound associated with turbulent flow is eq. (7.136), after
    Fredberg:

        S(f) = 0.7 (d/U) <P^2>_max / [ 1 + 0.5 f (d/U) ]^(10/3),

    where U is the velocity of blood in a NORMAL segment, d is the
    diameter of the stenotic segment and f is frequency in Hz.  The book
    also gives eq. (7.135) for the Reynolds-type parameter of the stenotic
    segment,

        x = 10^-3 (u d / nu) (D / d)^0.75,

    with u the blood velocity in the stenotic segment.

    NOTE ON THE SOURCE TEXT: in the flattened text of eq. (7.136) the
    ratio d/U appears as the run-together "Ud".  It must be d/U, not the
    product: f (d/U) is dimensionless, as the bracket of a power law
    requires, whereas f U d is not, and 0.7 (d/U) <P^2>_max carries units
    of pressure-squared per hertz, as a spectral density must.

    WHY: a stenosis accelerates blood through the narrowed lumen; past a
    critical Reynolds number the flow breaks into turbulence, and the
    pressure fluctuations of that turbulence radiate as a wideband sound
    that can be picked up on the chest wall.  The corner frequency of the
    spectrum scales as U/d, so a tighter stenosis pushes acoustic energy
    to HIGHER frequencies -- which is the physical basis of acoustic
    detection of coronary artery disease.

    Parameters
    ----------
    diameter : float
        Diameter D of the normal (unstenosed) segment, in metres (m).
        Must be positive.
    flow_velocity : float
        Velocity U of blood in the normal segment, in metres per second
        (m/s).  Must be positive.
    stenosis_pct : float
        Percentage reduction in cross-sectional AREA at the stenosis, in
        percent, 0 <= stenosis_pct < 100.  The stenotic diameter follows
        from the area, d = D sqrt(1 - stenosis_pct/100), and continuity
        gives the stenotic velocity u = U / (1 - stenosis_pct/100).
    p2max : float
        <P^2>_max, the peak mean-square pressure fluctuation of the
        turbulence, in pascals squared (Pa^2).  Sets the overall level
        only.
    freqs : array-like or None
        Frequencies in hertz (Hz) at which to evaluate S(f).  ``None``
        gives 1 Hz to 1000 Hz in 1 Hz steps, the band used for coronary
        acoustic detection.
    nu : float
        Kinematic viscosity of blood in square metres per second (m^2/s);
        default 3.5e-6, i.e. about 3.5 cSt.

    Returns
    -------
    RichResult
        ``freq_hz`` and ``psd_Pa2_per_Hz`` -- the spectrum S(f);
        ``d_stenotic_m``, ``u_stenotic_m_s``;
        ``corner_freq_hz`` = 2 U / d, the frequency at which the bracket
        of eq. (7.136) reaches 2 and the spectrum begins its -10/3 roll
        off;
        ``reynolds_param_x`` -- eq. (7.135);
        ``reynolds_number`` = u d / nu.
    """
    D = float(diameter)
    U = float(flow_velocity)
    if D <= 0.0:
        raise ValueError("diameter must be positive (m)")
    if U <= 0.0:
        raise ValueError("flow_velocity must be positive (m/s)")
    s = float(stenosis_pct)
    if not 0.0 <= s < 100.0:
        raise ValueError("stenosis_pct must be in [0, 100) percent")
    p2max = float(p2max)
    if p2max <= 0.0:
        raise ValueError("p2max must be positive (Pa^2)")
    nu = float(nu)
    if nu <= 0.0:
        raise ValueError("nu must be positive (m^2/s)")
    open_frac = 1.0 - s / 100.0
    d = D * sqrt(open_frac)
    u = U / open_frac
    if freqs is None:
        fs_hz = [float(k) for k in range(1, 1001)]
    else:
        fs_hz = [float(v) for v in aslist(freqs)]
        if any(v < 0.0 for v in fs_hz):
            raise ValueError("frequencies must be non-negative (Hz)")
    tau = d / U                       # seconds
    psd = [0.7 * tau * p2max / (1.0 + 0.5 * f * tau) ** (10.0 / 3.0)
           for f in fs_hz]
    x = 1e-3 * (u * d / nu) * (D / d) ** 0.75
    return RichResult(payload={
        "freq_hz": fs_hz, "psd_Pa2_per_Hz": psd,
        "D_normal_m": D, "d_stenotic_m": d,
        "U_normal_m_s": U, "u_stenotic_m_s": u,
        "stenosis_pct": s,
        "corner_freq_hz": 2.0 / tau,
        "reynolds_param_x": x,
        "reynolds_number": u * d / nu,
        "total_power_Pa2": fsum(psd) * (fs_hz[1] - fs_hz[0] if len(fs_hz) > 1 else 1.0),
        "units": {"freq": "Hz", "psd": "Pa^2/Hz", "diameter": "m",
                  "velocity": "m/s", "nu": "m^2/s"},
        "method": "Rangayyan (2024) eqs. (7.135) and (7.136), Section 7.7.2, after Wang et al. (1990) and Fredberg",
    })


rangayyan_coronary_sound = corsound  # pre-policy spelling


# -- rgcry: Infant cry signal analysis: formants and fundamental frequency.
def infantcry(cry, fs, window_ms=40.0, f0_range=(200.0, 1000.0), order=None,
              flat_tolerance=0.06):
    """Infant cry analysis: fundamental frequency track and cry melody.

    Rangayyan (2024) Section 8.13 ("Application: Analysis of Crying
    Sounds of Infants") reports the method of Varallyay, who "divided cry
    segments into short-time windows of duration 40 ms" and followed the
    fundamental frequency over time, calling that track the CRY MELODY.
    The book states: the fundamental frequencies analysed "were found to
    vary between 200 and 1,000 Hz, with the most common values being
    between 300 and 600 Hz"; three fundamental units of cry melody
    pattern were defined as falling (-1), flat (0) and rising (+1), and
    melodies were coded as sequences of these.  The book further reports
    mean F0 of 425.51 +/- 78.10 Hz for infants with hearing impairment
    against 408.74 +/- 64.77 Hz for controls, and (after Hirschberg)
    high-pitched melodies at 1,000-2,000 Hz in dysphonia against
    400-500 Hz in healthy infants.  Formants come from the all-pole model
    of Section 7.5, as in Section 7.2.3 for speech.

    WHY: a cry is produced by the same source-filter system as speech, so
    the fundamental frequency is set by the larynx and the formants by the
    vocal tract.  What is diagnostic is the TRAJECTORY of F0 rather than
    any single value: an infant who cannot hear itself cannot control its
    own phonation, and a diseased larynx cannot sustain a smooth pitch.
    Coding the track into rise/flat/fall units is what makes those
    trajectories comparable between recordings of different length.

    Parameters
    ----------
    cry : array-like
        Cry recording, in arbitrary amplitude units.
    fs : float
        Sampling rate in hertz (Hz); positive.  Must exceed twice the
        upper edge of f0_range.
    window_ms : float
        Short-time analysis window in milliseconds (ms); positive.  40 ms
        is the value the book quotes.  Windows do not overlap.
    f0_range : (float, float)
        Search range for F0 in hertz (Hz).  Default 200-1000 Hz is the
        range the book reports.
    order : int or None
        All-pole order for the formant estimate; ``None`` uses
        2 + fs/1000.
    flat_tolerance : float
        Relative F0 change below which consecutive voiced windows are
        coded flat (0) rather than rising (+1) or falling (-1);
        dimensionless, in (0, 1).  The default 0.06 is about one
        semitone.  This is a CALIBRATION KNOB: with the book's 40 ms
        windows a genuine glide of a few hundred hertz per second moves
        F0 by only a percent or two per window, so a tolerance set for
        long windows will code a real melody as flat.  Lower it when the
        window is short or the glides are slow.

    Returns
    -------
    RichResult
        ``f0_track_hz`` -- one F0 (or None where unvoiced) per window;
        ``t_track_s`` -- window start times;
        ``melody`` -- the sequence of -1 (falling), 0 (flat), +1 (rising)
        units, one per consecutive pair of voiced windows;
        ``mean_f0_hz``, ``sd_f0_hz``, ``min_f0_hz``, ``max_f0_hz``,
        ``f0_range_semitones`` -- the melody excursion;
        ``voiced_fraction``;
        ``in_common_band_fraction`` -- fraction of voiced windows with F0
        in 300-600 Hz, the book's most common range;
        ``high_pitched`` -- True when the mean F0 exceeds 1000 Hz, the
        Hirschberg dysphonia range;
        ``formants_hz`` -- formants of the loudest voiced window.
    """
    xs = [float(v) for v in aslist(cry)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    flo, fhi = float(f0_range[0]), float(f0_range[1])
    if not 0.0 < flo < fhi:
        raise ValueError("f0_range must satisfy 0 < lo < hi (Hz)")
    if fhi * 2.0 > fs:
        raise ValueError("fs must exceed twice the upper edge of f0_range")
    window_ms = float(window_ms)
    if window_ms <= 0.0:
        raise ValueError("window_ms must be positive (ms)")
    w = int(round(window_ms * fs / 1000.0))
    if w < int(2.0 * fs / flo):
        raise ValueError("a %g ms window holds fewer than two periods at %g Hz"
                         % (window_ms, flo))
    nwin = len(xs) // w
    if nwin < 2:
        raise ValueError("need at least 2 whole analysis windows")
    tol = float(flat_tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("flat_tolerance must be a relative change in (0, 1)")
    lag_lo = max(1, int(fs / fhi))
    lag_hi = min(w - 1, int(fs / flo))
    if lag_hi <= lag_lo:
        raise ValueError("window is too short for the requested f0_range")
    track, tt, best_seg, best_rms = [], [], None, -1.0
    for i in range(nwin):
        seg = xs[i * w:(i + 1) * w]
        tt.append(i * w / fs)
        try:
            acf = _bsaacf(seg, lag_hi)
        except ValueError:
            track.append(None)
            continue
        if acf[0] <= 0.0:
            track.append(None)
            continue
        k = max(range(lag_lo, lag_hi + 1), key=lambda j: acf[j])
        if acf[k] / acf[0] < 0.3:
            track.append(None)
            continue
        track.append(fs / k)
        r = _bsarms(seg)
        if r > best_rms:
            best_rms, best_seg = r, seg
    voiced = [v for v in track if v is not None]
    if not voiced:
        raise ValueError("no voiced window found; check f0_range and the recording")
    melody = []
    prev = None
    for v in track:
        if v is None:
            continue
        if prev is not None:
            ratio = v / prev
            melody.append(0 if abs(ratio - 1.0) < tol
                          else (1 if ratio > 1.0 else -1))
        prev = v
    n = len(voiced)
    mf = fsum(voiced) / n
    sf = sqrt(fsum((v - mf) ** 2 for v in voiced) / (n - 1)) if n > 1 else 0.0
    fmt = []
    if best_seg is not None:
        p = int(order) if order is not None else int(2 + fs / 1000.0)
        if len(best_seg) >= 4 * p and p >= 4:
            try:
                a, err = _bsalpc(best_seg, p)
                fr, ps = _bsalpcspec(a, fs, npts=1024)
                fmt = sorted(f for f, _ in _bsapeaks(fr, ps, count=4, minsep=150.0))
            except ValueError:
                fmt = []
    return RichResult(payload={
        "f0_track_hz": track, "t_track_s": tt, "melody": melody,
        "melody_units": {"-1": "falling", "0": "flat", "1": "rising"},
        "mean_f0_hz": mf, "sd_f0_hz": sf,
        "min_f0_hz": min(voiced), "max_f0_hz": max(voiced),
        "f0_range_semitones": 12.0 * log(max(voiced) / min(voiced), 2.0),
        "voiced_fraction": n / nwin,
        "in_common_band_fraction": sum(1 for v in voiced
                                       if 300.0 <= v <= 600.0) / n,
        "high_pitched": mf > 1000.0,
        "formants_hz": fmt,
        "window_ms": window_ms, "n_windows": nwin, "fs_hz": fs,
        "flat_tolerance": tol,
        "units": {"frequency": "Hz", "time": "s",
                  "f0_range_semitones": "semitones"},
        "method": "Rangayyan (2024) Section 8.13 after Varallyay: 40 ms windows, F0 track as the cry melody coded falling/flat/rising",
    })


rangayyan_infant_cry = infantcry  # pre-policy spelling


# -- rgegg: Electrogastrogram (EGG) feature extraction (dominant frequency, power).
def eggfeat(egg, fs, normal_band=(0.0333, 0.0667)):
    """Electrogastrogram features: dominant frequency and power distribution.

    Rangayyan (2024) Section 1.2.8 ("The electrogastrogram (EGG)") states
    that the electrical activity of the stomach consists of rhythmic waves
    of depolarisation and repolarisation of its smooth muscle cells "with
    intervals of about 20 s in humans", i.e. a normal gastric slow-wave
    frequency of about 3 cycles per minute (0.05 Hz), and that Chen et al.
    recorded cutaneous EGG filtered to 0.02-0.3 Hz and sampled at 2 Hz.
    The book notes that "gastric dysrhythmia or arrhythmia may be detected
    via analysis of the EGG".  The PSD measures are those of Section 6.4.1
    (eqs. 6.32, 6.34, 6.35) and the band-power fraction is eq. (6.44),
    Section 6.4.2.

    WHY: the diagnostic quantity in an EGG is not amplitude -- cutaneous
    EGG amplitude depends entirely on electrode placement and body
    habitus -- but FREQUENCY.  Normogastria sits near 3 cpm; bradygastria
    (below about 2 cpm) and tachygastria (above about 4 cpm) are the
    dysrhythmias.  So the useful output is the dominant frequency and the
    fraction of power falling in the normal band, both of which are
    invariant to recording gain.

    Parameters
    ----------
    egg : array-like
        Cutaneous EGG recording, in arbitrary amplitude units
        (microvolts at the amplifier output in the usual setup).  At
        least 4 samples, and long enough to resolve 0.05 Hz -- several
        minutes.
    fs : float
        Sampling rate in hertz (Hz); positive.  2 Hz in the protocol the
        book describes.
    normal_band : (float, float)
        Normogastric band in hertz (Hz).  Default 0.0333-0.0667 Hz, i.e.
        2-4 cycles per minute, bracketing the 3 cpm the book's 20 s
        interval implies.

    Returns
    -------
    RichResult
        ``dominant_freq_hz`` and ``dominant_freq_cpm`` (cycles per
        minute); ``normal_fraction``, ``brady_fraction`` and
        ``tachy_fraction`` -- band power fractions below, within and above
        the normal band, all dimensionless and summing to 1 over
        0 to fs/2; ``rhythm`` -- "normogastria", "bradygastria" or
        "tachygastria" from where the dominant frequency falls;
        ``mean_freq_hz``, ``median_freq_hz``, ``spread_hz``,
        ``total_power``; ``duration_s``; ``freq_hz`` and ``psd``.
    """
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    lo, hi = float(normal_band[0]), float(normal_band[1])
    if not 0.0 < lo < hi:
        raise ValueError("normal_band must satisfy 0 < lo < hi (Hz)")
    if hi > fs / 2.0:
        raise ValueError("normal_band upper edge exceeds the Nyquist frequency")
    xs = [float(v) for v in aslist(egg)]
    dur = len(xs) / fs
    if dur < 2.0 / lo:
        raise ValueError("recording of %.1f s is too short to resolve %g Hz; "
                         "need at least %.0f s" % (dur, lo, 2.0 / lo))
    freqs, psd = _bsapsd(xs, fs)
    mom = _bsapsdmom(freqs, psd)
    tot = mom["total_power"]
    fr_norm = _bsabandpow(freqs, psd, lo, hi) / tot
    fr_brady = _bsabandpow(freqs, psd, 0.0, lo) / tot
    fr_tachy = _bsabandpow(freqs, psd, hi, fs / 2.0 + 1.0) / tot
    band = [(f, p) for f, p in zip(freqs, psd) if 0.0 < f <= min(0.5, fs / 2.0)]
    if not band:
        raise ValueError("no spectral bins in the gastric frequency range")
    fdom = max(band, key=lambda t: t[1])[0]
    rhythm = ("normogastria" if lo <= fdom < hi
              else ("bradygastria" if fdom < lo else "tachygastria"))
    out = dict(mom)
    out.update({
        "dominant_freq_hz": fdom, "dominant_freq_cpm": fdom * 60.0,
        "normal_fraction": fr_norm, "brady_fraction": fr_brady,
        "tachy_fraction": fr_tachy, "rhythm": rhythm,
        "normal_band_hz": (lo, hi), "normal_band_cpm": (lo * 60.0, hi * 60.0),
        "duration_s": dur, "fs_hz": fs,
        "freq_hz": freqs, "psd": psd,
        "units": {"frequency": "Hz", "dominant_freq_cpm": "cycles/minute",
                  "fractions": "dimensionless", "duration": "s"},
        "method": "Rangayyan (2024) Section 1.2.8 with the PSD measures of Section 6.4.1 and the band fraction of eq. (6.44)",
    })
    return RichResult(payload=out)


rangayyan_egg = eggfeat  # pre-policy spelling


# -- rgelast: Heart-sound spectral stiffness index.
def rangayyan_heart_elasticity(pcg, fs, s1_window=None):
    r"""Spectral index of myocardial stiffness from S1 (Rangayyan
    Ch. 3).

    Higher stiffness shifts the S1 spectrum upward, so the dominant
    frequency and spectral centroid of the first heart sound track
    elasticity. This returns those descriptors -- it does NOT return a
    stiffness value in physical units: the relationship is monotone
    but the calibration is subject- and instrument-specific, and
    inventing an absolute number would be a fabrication.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram, or an isolated S1 segment.
    fs : float
        Sampling frequency.
    s1_window : (int, int), optional
        Sample range holding S1; the whole record if omitted.

    Returns
    -------
    RichResult
        keys: ``dominant_frequency``, ``spectral_centroid``,
        ``bandwidth_3db``, ``freqs``, ``psd``, ``calibrated`` (False),
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (heart sounds; spectral indices).
    """
    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if s1_window is not None:
        a, b = int(s1_window[0]), int(s1_window[1])
        if not 0 <= a < b <= x.size:
            raise ValueError(f"s1_window ({a}, {b}) is out of range.")
        x = x[a:b]
    if x.size < 16:
        raise ValueError(f"need at least 16 samples, got {x.size}.")
    w = rangayyan_welch_psd(x, fs=fs, nperseg=min(256, x.size))
    f, p = w["freqs"], w["psd"]
    tot = float(p.sum())
    centroid = float(np.sum(f * p) / tot) if tot > 0 else np.nan
    ipk = int(np.argmax(p))
    above = np.flatnonzero(p >= p[ipk] / 2.0)
    bw = float(f[above[-1]] - f[above[0]]) if above.size else 0.0
    return RichResult(payload={"dominant_frequency": float(f[ipk]),
                               "spectral_centroid": centroid, "bandwidth_3db": bw,
                               "freqs": f, "psd": p, "calibrated": False,
                               "method": "S1 spectral descriptors; monotone in stiffness, NOT calibrated"})


# -- rgengn: Electroneurogram (ENG) compound action potential model.
def engcap(t, distance_m=0.1, n_fibers=40, cv_range=(45.0, 70.0),
           amp_range=(0.5, 2.0), width_ms=0.3):
    """Electroneurogram compound action potential from a fibre population.

    Rangayyan (2024) Section 1.2.3 ("The electroneurogram (ENG)") defines
    the ENG as the electrical signal observed as a stimulus and its
    associated action potential propagate along a nerve, and states that
    conduction velocity is measured by stimulating a motor nerve and
    recording the response at two points a known distance apart.  The book
    gives no compound-action-potential equation, so the synthesis used
    here is stated rather than cited: each fibre class contributes a
    CAUSAL biphasic potential

        s(u) = u (2 - u) exp(-u)   for u >= 0,   0 for u < 0,

    with u = (t - latency)/width and latency = distance / velocity.  The
    template is causal on purpose -- a Gaussian-derivative pulse spreads
    symmetrically about its centre and would make the CAP appear to begin
    before the fastest fibre could possibly have arrived, corrupting the
    onset-latency conduction velocity that is the clinical measurement.

    WHY: a peripheral nerve is not one conductor but a bundle of fibres
    with a distribution of diameters and therefore of conduction
    velocities.  The recorded CAP is their sum, so a faster fibre class
    arrives earlier than a slower one and the CAP disperses and flattens
    with recording distance.  Demyelination slows the fast fibres and
    shows up exactly as a longer latency and a lower, broader CAP -- which
    is why the two-point latency measurement the book describes is the
    clinical test.

    Parameters
    ----------
    t : array-like
        Time (latency) axis in milliseconds (ms) measured from the
        stimulus artefact.
    distance_m : float
        Distance between the stimulation and recording sites, in metres
        (m).  Must be positive.
    n_fibers : int
        Number of fibre classes spread evenly across cv_range.  >= 1.
    cv_range : (float, float)
        Slowest and fastest conduction velocities in metres per second
        (m/s).  Both positive, low <= high.  45-70 m/s is the normal
        range for large myelinated motor fibres.
    amp_range : (float, float)
        Contribution of the slowest and fastest fibre class, in
        microvolts (uV); intermediate classes are interpolated linearly.
    width_ms : float
        Gaussian width sigma of a single fibre class potential, in
        milliseconds (ms); must be positive.

    Returns
    -------
    RichResult
        ``t_ms``, ``cap_uV`` compound action potential;
        ``peak_uV``, ``peak_latency_ms``, ``onset_latency_ms``
        (first crossing of 5 percent of peak);
        ``cv_from_peak_m_s`` = distance / peak latency and
        ``cv_from_onset_m_s`` = distance / onset latency -- the onset
        figure estimates the velocity of the FASTEST fibres, which is the
        clinically reported conduction velocity;
        ``latencies_ms`` -- the per-class arrival latencies.
    """
    ts = [float(v) for v in aslist(t)]
    if len(ts) < 3:
        raise ValueError("t must contain at least 3 time points (ms)")
    distance_m = float(distance_m)
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive (m)")
    n_fibers = int(n_fibers)
    if n_fibers < 1:
        raise ValueError("n_fibers must be at least 1")
    lo, hi = (float(v) for v in cv_range)
    if lo <= 0.0 or hi <= 0.0 or lo > hi:
        raise ValueError("cv_range must be positive with low <= high (m/s)")
    alo, ahi = (float(v) for v in amp_range)
    width_ms = float(width_ms)
    if width_ms <= 0.0:
        raise ValueError("width_ms must be positive (ms)")
    cvs = [lo] if n_fibers == 1 else [lo + (hi - lo) * i / (n_fibers - 1)
                                      for i in range(n_fibers)]
    amps = [alo] if n_fibers == 1 else [alo + (ahi - alo) * i / (n_fibers - 1)
                                        for i in range(n_fibers)]
    # distance in m over velocity in m/s gives seconds; times 1000 -> ms
    lats = [1000.0 * distance_m / v for v in cvs]
    wave = []
    for ti in ts:
        acc = 0.0
        for lat, a in zip(lats, amps):
            u = (ti - lat) / width_ms
            if u < 0.0 or u > 30.0:
                continue
            acc += a * u * (2.0 - u) * exp(-u)
        wave.append(acc)
    pk = max(abs(v) for v in wave)
    if pk <= 0.0:
        raise ValueError("CAP is identically zero; t must cover latencies "
                         + repr((min(lats), max(lats))) + " ms")
    ipk = max(range(len(wave)), key=lambda i: abs(wave[i]))
    onset = None
    for i in range(len(wave)):
        if abs(wave[i]) >= 0.05 * pk:
            onset = ts[i]
            break
    return RichResult(payload={
        "t_ms": ts, "cap_uV": wave,
        "peak_uV": wave[ipk], "peak_latency_ms": ts[ipk],
        "onset_latency_ms": onset,
        "cv_from_peak_m_s": 1000.0 * distance_m / ts[ipk] if ts[ipk] > 0 else None,
        "cv_from_onset_m_s": (1000.0 * distance_m / onset)
                             if onset and onset > 0 else None,
        "latencies_ms": lats, "velocities_m_s": cvs,
        "distance_m": distance_m,
        "units": {"t": "ms", "cap": "uV", "velocity": "m/s", "distance": "m"},
        "method": "ENG compound action potential from a fibre-velocity population; ENG and two-point conduction-velocity measurement per Rangayyan (2024) Section 1.2.3 (no CAP equation in the book)",
    })


rangayyan_eng = engcap  # pre-policy spelling


# -- rgepidet: Epileptic seizure detection in EEG.
def seizdet(eeg, fs, epoch_s=1.0, ratio_threshold=2.0, baseline_epochs=None):
    """Detect epileptic seizure activity in an EEG channel by band banding.

    Rangayyan (2024) Section 8.17 ("Application: Detection of Epileptic
    Seizures in EEG Signals") is the application; Section 6.4.2 gives the
    method of Binnie et al., who partitioned the EEG spectrum "into not
    only the traditional delta, theta, alpha and beta bands, but also into
    seven other nonuniform bands specified as 1-2, 2-4, 4-6, 6-8, 8-11,
    11-14, and > 14 Hz", together with form-factor features
    (Section 5.6.4), reporting that in 275 patients with suspected
    epilepsy 90 percent of the pathological signals were classified as
    abnormal.  The traditional bands are those of Section 1.2.6:
    delta 0.5 <= f < 4 Hz, theta 4 <= f < 8 Hz, alpha 8 <= f <= 13 Hz,
    beta f > 13 Hz, with gamma 30-80 Hz.  Band fractions are eq. (6.44)
    and the form factor is eq. (5.26).

    WHY: a seizure is a transition to abnormally synchronous firing, and
    synchrony concentrates spectral power into slow, high-amplitude
    rhythms.  So the seizure signature is a RATIO -- low-band power rising
    relative to a preceding baseline of the same channel -- not an
    absolute power, because absolute EEG power varies by an order of
    magnitude between electrodes and subjects.  Comparing each epoch
    against that subject's own baseline is what makes the detector
    portable.

    Parameters
    ----------
    eeg : array-like
        Single-channel EEG, in microvolts (uV).
    fs : float
        Sampling rate in hertz (Hz); positive.  Must be at least 30 Hz
        for the beta band to be meaningful.
    epoch_s : float
        Analysis epoch length in seconds (s); positive.  The book's
        adaptive-segmentation work uses 1 s or longer (Section 5.6.4).
    ratio_threshold : float
        An epoch is flagged when its (delta+theta) power fraction exceeds
        ratio_threshold times the baseline mean of that fraction.
        Dimensionless, must exceed 1.
    baseline_epochs : int or None
        Number of leading epochs treated as the seizure-free baseline.
        ``None`` uses the first quarter of the epochs, at least one.

    Returns
    -------
    RichResult
        ``epochs`` -- one dict per epoch with ``t_start_s``, the seven
        Binnie band fractions, the four traditional band fractions,
        ``slow_fraction`` (delta+theta), ``form_factor`` and
        ``flagged``;
        ``seizure_detected``, ``n_flagged``, ``seizure_intervals_s``
        (merged runs of flagged epochs);
        ``baseline_slow_fraction``; ``threshold_slow_fraction``.
    """
    xs = [float(v) for v in aslist(eeg)]
    fs = float(fs)
    if fs < 30.0:
        raise ValueError("fs must be at least 30 Hz to resolve the beta band")
    epoch_s = float(epoch_s)
    if epoch_s <= 0.0:
        raise ValueError("epoch_s must be positive (s)")
    ratio_threshold = float(ratio_threshold)
    if ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must exceed 1 (it is a ratio to baseline)")
    w = int(round(epoch_s * fs))
    if w < 8:
        raise ValueError("epoch of %g s is only %d samples; use a longer epoch"
                         % (epoch_s, w))
    n_ep = len(xs) // w
    if n_ep < 2:
        raise ValueError("need at least 2 whole epochs")
    binnie = [(1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0),
              (8.0, 11.0), (11.0, 14.0), (14.0, fs / 2.0)]
    trad = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0001), "beta": (13.0001, fs / 2.0),
            "gamma": (30.0, min(80.0, fs / 2.0))}
    rows = []
    for e in range(n_ep):
        seg = xs[e * w:(e + 1) * w]
        freqs, psd = _bsapsd(seg, fs)
        tot = fsum(psd)
        if tot <= 0.0:
            raise ValueError("epoch %d is constant; no spectrum to analyse" % e)
        row = {"t_start_s": e * epoch_s}
        for lo, hi in binnie:
            row["b_%g_%g_hz" % (lo, hi)] = _bsabandpow(freqs, psd, lo, hi) / tot
        for nm, (lo, hi) in trad.items():
            if hi > lo:
                row[nm + "_fraction"] = _bsabandpow(freqs, psd, lo, hi) / tot
        row["slow_fraction"] = row.get("delta_fraction", 0.0) \
            + row.get("theta_fraction", 0.0)
        row["form_factor"] = _bsahjorth(seg)["form_factor"]
        rows.append(row)
    nb = int(baseline_epochs) if baseline_epochs is not None \
        else max(1, n_ep // 4)
    if not 1 <= nb < n_ep:
        raise ValueError("baseline_epochs must be between 1 and n_epochs-1")
    base = fsum(rows[i]["slow_fraction"] for i in range(nb)) / nb
    if base <= 0.0:
        raise ValueError("baseline has no slow-band power; cannot form a ratio")
    thr = ratio_threshold * base
    for row in rows:
        row["flagged"] = row["slow_fraction"] > thr
    runs, start = [], None
    for i, row in enumerate(rows):
        if row["flagged"] and start is None:
            start = i
        elif not row["flagged"] and start is not None:
            runs.append((start * epoch_s, i * epoch_s))
            start = None
    if start is not None:
        runs.append((start * epoch_s, n_ep * epoch_s))
    nflag = sum(1 for row in rows if row["flagged"])
    return RichResult(payload={
        "epochs": rows,
        "seizure_detected": nflag > 0, "n_flagged": nflag,
        "seizure_intervals_s": runs,
        "baseline_slow_fraction": base, "threshold_slow_fraction": thr,
        "baseline_epochs": nb, "n_epochs": n_ep,
        "epoch_s": epoch_s, "fs_hz": fs,
        "binnie_bands_hz": binnie,
        "units": {"eeg": "uV", "time": "s", "fractions": "dimensionless"},
        "method": "Rangayyan (2024) Section 8.17 with the spectral banding of Binnie et al., Section 6.4.2, the EEG bands of Section 1.2.6 and the form factor of eq. (5.26)",
    })


rangayyan_epilepsy_detect = seizdet  # pre-policy spelling


# -- rgerp: Event-related potential (ERP) latency and amplitude features.
def erpfeat(erp, fs, t0=0.0, components=None, baseline_ms=(None, 0.0)):
    """Latency and amplitude features of an event-related potential.

    Rangayyan (2024) Section 1.2.7 ("Event-related potentials (ERPs)")
    defines the ERP as the ENG or EEG response to light, sound,
    electrical or other external stimuli, notes that short-latency ERPs
    depend mainly on the physical stimulus while longer-latency ERPs
    depend on the conditions of presentation, and states for
    somatosensory evoked potentials that "the latency, duration, and
    amplitude of the response are measured".  Those three quantities are
    what this block extracts.  Section 3.12 ("Application: Removal of
    Artifacts in ERP Signals") covers the synchronised averaging that
    must precede this measurement.

    WHY: an ERP is defined by WHEN it happens as much as by how big it
    is.  Conduction delay along the pathway sets the latency, so a
    prolonged latency localises a lesion; amplitude reflects the number of
    synchronously active neurons and is far more variable between
    subjects.  This is why clinical ERP reporting leads with latency, and
    why the measurement window for each component has to be fixed in
    advance rather than chosen after looking at the trace.

    Parameters
    ----------
    erp : array-like
        The averaged ERP waveform, in microvolts (uV).  At least 4
        samples.
    fs : float
        Sampling rate in hertz (Hz); positive.
    t0 : float
        Time of the stimulus, in milliseconds (ms), relative to the first
        sample.  Latencies are reported relative to this.
    components : dict or None
        Mapping of component name to (window_start_ms, window_end_ms,
        polarity), where polarity is +1 for a positive peak and -1 for a
        negative one, and windows are relative to t0.  ``None`` uses the
        conventional long-latency auditory/visual set
        N100 (50-150 ms, -1), P200 (150-250 ms, +1),
        N200 (180-300 ms, -1) and P300 (250-500 ms, +1).
    baseline_ms : (float or None, float)
        Pre-stimulus window in milliseconds (ms) relative to t0 whose mean
        is subtracted before measurement.  A start of ``None`` means the
        beginning of the record.  Amplitudes are baseline-to-peak.

    Returns
    -------
    RichResult
        ``components`` -- one dict per requested component with
        ``latency_ms``, ``amplitude_uV`` (baseline to peak),
        ``found`` (False when the window falls outside the record);
        ``peak_to_peak_uV`` over the whole epoch;
        ``baseline_uV``; ``t_ms`` and ``erp_uV`` -- the baseline-corrected
        waveform.
    """
    xs = [float(v) for v in aslist(erp)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    if len(xs) < 4:
        raise ValueError("need at least 4 ERP samples")
    t0 = float(t0)
    ts = [1000.0 * i / fs - t0 for i in range(len(xs))]
    bstart, bend = baseline_ms
    bend = float(bend)
    idx = [i for i in range(len(ts))
           if ts[i] < bend and (bstart is None or ts[i] >= float(bstart))]
    base = fsum(xs[i] for i in idx) / len(idx) if idx else 0.0
    ys = [v - base for v in xs]
    if components is None:
        components = {"N100": (50.0, 150.0, -1), "P200": (150.0, 250.0, 1),
                      "N200": (180.0, 300.0, -1), "P300": (250.0, 500.0, 1)}
    if not isinstance(components, dict):
        raise ValueError("components must be a dict of name -> (t1, t2, polarity)")
    rows = {}
    for name, spec in components.items():
        t1, t2, pol = float(spec[0]), float(spec[1]), int(spec[2])
        if t2 <= t1:
            raise ValueError("component %s has an empty window" % name)
        if pol not in (1, -1):
            raise ValueError("component %s polarity must be +1 or -1" % name)
        win = [i for i in range(len(ts)) if t1 <= ts[i] <= t2]
        if not win:
            rows[name] = {"latency_ms": None, "amplitude_uV": None,
                          "found": False, "window_ms": (t1, t2)}
            continue
        i = max(win, key=lambda k: pol * ys[k])
        rows[name] = {"latency_ms": ts[i], "amplitude_uV": ys[i],
                      "found": True, "window_ms": (t1, t2), "polarity": pol}
    return RichResult(payload={
        "components": rows,
        "peak_to_peak_uV": max(ys) - min(ys),
        "baseline_uV": base,
        "t_ms": ts, "erp_uV": ys, "fs_hz": fs, "t0_ms": t0,
        "units": {"amplitude": "uV", "latency": "ms"},
        "method": "Rangayyan (2024) Section 1.2.7 (latency, duration and amplitude of the response); averaging per Section 3.12",
    })


rangayyan_erp_features = erpfeat  # pre-policy spelling


# -- rgfeatex: Feature extraction for BCI from EEG (event-related desynchronization/synchronization).
def erders(eeg, fs, ref_window, active_window, band=(8.0, 13.0)):
    """Event-related desynchronisation / synchronisation for a BCI feature.

    Rangayyan (2024) Section 9.12 ("Application: EEG Analysis for
    Brain-Computer Interfaces") and its Section 9.12.2 ("Feature
    extraction") describe non-negative-matrix-factorisation channel
    selection for motor-imagery BCI, with the EEG band definitions of
    Section 1.2.6 (alpha 8 <= f <= 13 Hz, beta f > 13 Hz).  The book does
    NOT give an ERD/ERS formula.  The definition implemented here is the
    primary one:

        Pfurtscheller G, Aranibar A, "Evaluation of event-related
        desynchronization (ERD) preceding and following voluntary
        self-paced movement", Electroencephalography and Clinical
        Neurophysiology 46(2):138-146, 1979;
        Pfurtscheller G, Lopes da Silva FH, "Event-related EEG/MEG
        synchronization and desynchronization: basic principles",
        Clinical Neurophysiology 110(11):1842-1857, 1999.

        ERD/ERS percent = 100 (A - R) / R,

    where R is the band power averaged over a reference (rest) interval
    and A is the band power over the active interval.  A NEGATIVE value is
    desynchronisation (ERD, power drop), a POSITIVE value is
    synchronisation (ERS, power rise); the sign convention is the source's
    and is reported explicitly in the payload because the opposite
    convention also appears in the literature.

    WHY: the motor-imagery BCI control signal is not an evoked potential
    -- it is not time-locked to a stimulus and averaging in the time
    domain destroys it.  It is a change in the POWER of an ongoing rhythm
    over the sensorimotor cortex.  Expressing that change relative to each
    trial's own rest interval cancels the enormous between-subject and
    between-session variation in absolute alpha power, which is the only
    reason the feature transfers at all.

    Parameters
    ----------
    eeg : array-like
        Single-channel EEG for one trial, in microvolts (uV).
    fs : float
        Sampling rate in hertz (Hz); positive.  400 Hz in the dataset the
        book describes.
    ref_window : (float, float)
        Reference (rest) interval in seconds (s) from the start of the
        record; must be non-empty and inside the record.
    active_window : (float, float)
        Active (motor imagery) interval in seconds (s), same conditions.
    band : (float, float)
        Frequency band in hertz (Hz).  Default 8-13 Hz is the alpha /
        mu band of Section 1.2.6, the usual motor-imagery band; 13-30 Hz
        picks the beta band instead.

    Returns
    -------
    RichResult
        ``erd_percent`` -- 100 (A - R) / R, negative for
        desynchronisation; ``ref_power`` and ``active_power`` in uV^2;
        ``power_ratio`` = A / R (dimensionless);
        ``event`` -- "ERD", "ERS" or "none" (within 1 percent);
        ``band_hz``; ``ref_samples`` and ``active_samples``.
    """
    xs = [float(v) for v in aslist(eeg)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 <= lo < hi:
        raise ValueError("band must satisfy 0 <= lo < hi (Hz)")
    if hi > fs / 2.0:
        raise ValueError("band upper edge exceeds the Nyquist frequency")
    dur = len(xs) / fs

    def cut(win, name):
        a, b = float(win[0]), float(win[1])
        if b <= a:
            raise ValueError("%s must have end > start (s)" % name)
        if a < 0.0 or b > dur:
            raise ValueError("%s (%g, %g) s falls outside the %.3f s record"
                             % (name, a, b, dur))
        seg = xs[int(round(a * fs)):int(round(b * fs))]
        if len(seg) < 4:
            raise ValueError("%s is only %d samples; widen it" % (name, len(seg)))
        return seg

    rseg = cut(ref_window, "ref_window")
    aseg = cut(active_window, "active_window")
    rf, rp = _bsapsd(rseg, fs)
    af, ap = _bsapsd(aseg, fs)
    # power per sample so that windows of different length compare
    R = _bsabandpow(rf, rp, lo, hi) / len(rseg)
    A = _bsabandpow(af, ap, lo, hi) / len(aseg)
    if R <= 0.0:
        raise ValueError("reference window has no power in %g-%g Hz" % (lo, hi))
    pct = 100.0 * (A - R) / R
    event = "none" if abs(pct) < 1.0 else ("ERS" if pct > 0.0 else "ERD")
    return RichResult(payload={
        "erd_percent": pct, "ref_power": R, "active_power": A,
        "power_ratio": A / R, "event": event,
        "band_hz": (lo, hi),
        "ref_window_s": (float(ref_window[0]), float(ref_window[1])),
        "active_window_s": (float(active_window[0]), float(active_window[1])),
        "ref_samples": len(rseg), "active_samples": len(aseg), "fs_hz": fs,
        "sign_convention": "negative erd_percent = desynchronisation (power drop)",
        "units": {"power": "uV^2 per sample", "erd_percent": "percent",
                  "frequency": "Hz"},
        "method": "Pfurtscheller & Aranibar (1979) / Pfurtscheller & Lopes da Silva (1999) ERD-ERS; not defined in Rangayyan (2024), whose Section 9.12.2 covers NMF channel selection",
    })


rangayyan_feature_extract_bci = erders  # pre-policy spelling


# -- rgfrqdom: Frequency-domain feature extraction for CAD.
def cadspec(x, fs, bands=None):
    """Frequency-domain PSD features for coronary artery disease detection.

    Rangayyan (2024) Section 6.4.1 ("Moments of PSDs") defines the
    features computed here directly from the PSD S_xx treated as a
    density:

        total power  E_p                                     eq. (6.32)
        mean frequency in Hz                                 eq. (6.34)
        median frequency (splits the PSD in half)            eq. (6.35)
        variance fm2                                         eq. (6.37)
        spectral skewness = fm3 / fm2^(3/2)                  eq. (6.38)
        spectral kurtosis = fm4 / fm2^2                      eq. (6.41)

    and Section 6.4.2 ("Spectral power ratios") defines the fraction of
    signal power in a band f1:f2 as eq. (6.44).  Section 7.10
    ("Application: Coronary Artery Disease") is where these measures are
    applied to diastolic heart sounds.

    WHY: the acoustic signature of a coronary stenosis is turbulence, and
    turbulence is wideband -- it puts power where a normal diastolic
    segment has almost none, above roughly 300 Hz.  A single number like
    RMS cannot see that, but the mean and median frequency move upward and
    the high-band power fraction rises, which is exactly what these
    moments measure.  The corresponding generative model is ``corsound``.

    Parameters
    ----------
    x : array-like
        Signal segment (for CAD work, a diastolic segment of the recorded
        chest-wall sound) in arbitrary amplitude units.  At least 4
        samples.
    fs : float
        Sampling rate in hertz (Hz); positive.
    bands : sequence of (float, float) or None
        Bands in hertz (Hz) over which to report the fractional power of
        eq. (6.44).  ``None`` uses (0, 100), (100, 300), (300, 600) and
        (600, fs/2) Hz, the low/mid/high partition used for diastolic
        heart-sound analysis.

    Returns
    -------
    RichResult
        ``total_power``, ``mean_freq_hz``, ``median_freq_hz``,
        ``spread_hz`` (the square root of fm2, a bandwidth measure),
        ``spectral_skewness``, ``spectral_kurtosis`` (both dimensionless);
        ``band_power_fraction`` -- a list of (lo_hz, hi_hz, fraction);
        ``dominant_freq_hz`` and ``bandwidth_3db_hz`` / ``q_factor`` of
        the dominant peak; ``freq_hz`` and ``psd`` for plotting.
    """
    fs = float(fs)
    freqs, power = _bsapsd(x, fs)
    mom = _bsapsdmom(freqs, power)
    if bands is None:
        bands = [(0.0, 100.0), (100.0, 300.0), (300.0, 600.0), (600.0, fs / 2.0)]
    frac = []
    for lo, hi in bands:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError("each band must have hi > lo (Hz)")
        frac.append((lo, hi, _bsabandpow(freqs, power, lo, hi) / mom["total_power"]))
    pk = _bsapeaks(freqs, power, count=1)
    fdom = pk[0][0] if pk else freqs[power.index(max(power))]
    bw, q = _bsaqfactor(freqs, power, fdom)
    out = dict(mom)
    out.update({
        "band_power_fraction": frac,
        "dominant_freq_hz": fdom,
        "bandwidth_3db_hz": bw, "q_factor": q,
        "freq_hz": freqs, "psd": power, "fs_hz": fs,
        "units": {"frequency": "Hz", "power": "signal units^2",
                  "fractions": "dimensionless"},
        "method": "Rangayyan (2024) eqs. (6.32), (6.34), (6.35), (6.37), (6.38), (6.41), (6.44); Sections 6.4.1, 6.4.2 and 7.10",
    })
    return RichResult(payload=out)


rangayyan_freq_domain_feat = cadspec  # pre-policy spelling


# -- rgghk: Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential.
def ghk(ion_concs, P_K=1.0, P_Na=0.04, P_Cl=0.45, T=310.15):
    """Goldman-Hodgkin-Katz voltage equation for the resting potential.

    Rangayyan (2024) does NOT give the Goldman-Hodgkin-Katz equation; the
    book stops at the single-ion reverse potential of Equation (7.139).
    The expression implemented here is the primary-source one:

        Goldman DE, "Potential, impedance, and rectification in membranes",
        Journal of General Physiology 27(1):37-60, 1943;
        Hodgkin AL, Katz B, "The effect of sodium ions on the electrical
        activity of the giant axon of the squid", Journal of Physiology
        108(1):37-77, 1949.

        V_m = (R T / F) * log( (P_K [K]_o + P_Na [Na]_o + P_Cl [Cl]_i)
                             / (P_K [K]_i + P_Na [Na]_i + P_Cl [Cl]_o) ).

    WHY: a real membrane is permeable to several ions at once, so the
    resting potential is not any one Nernst potential but a
    permeability-weighted compromise between them.  Chloride enters with
    its concentrations swapped because its valence is negative.

    Parameters
    ----------
    ion_concs : dict
        Concentrations in millimolar (mM) with keys ``K_out``, ``K_in``,
        ``Na_out``, ``Na_in``, ``Cl_out``, ``Cl_in``.  All must be present
        and strictly positive.
    P_K, P_Na, P_Cl : float
        Relative membrane permeabilities, dimensionless (only their ratios
        matter).  Defaults 1 : 0.04 : 0.45 are the resting squid-axon
        ratios of Hodgkin and Katz (1949).  Must be non-negative and not
        all zero.
    T : float
        Absolute temperature in kelvin (K); default 310.15 K = 37 degC.

    Returns
    -------
    RichResult
        ``potential_mV`` (mV), ``potential_V`` (V), ``numerator_mM`` and
        ``denominator_mM`` (the weighted concentration sums, in mM), and
        ``slope_mV`` = R T / F in mV.
    """
    if not isinstance(ion_concs, dict):
        raise ValueError("ion_concs must be a dict of concentrations in mM")
    need = ("K_out", "K_in", "Na_out", "Na_in", "Cl_out", "Cl_in")
    missing = [k for k in need if k not in ion_concs]
    if missing:
        raise ValueError("ion_concs is missing keys: " + ", ".join(missing))
    c = {}
    for k in need:
        v = float(ion_concs[k])
        if v <= 0.0:
            raise ValueError("concentration " + k + " must be positive (mM)")
        c[k] = v
    T = float(T)
    if T <= 0.0:
        raise ValueError("T must be a positive absolute temperature in kelvin")
    P_K, P_Na, P_Cl = float(P_K), float(P_Na), float(P_Cl)
    if min(P_K, P_Na, P_Cl) < 0.0:
        raise ValueError("permeabilities must be non-negative")
    if P_K + P_Na + P_Cl <= 0.0:
        raise ValueError("at least one permeability must be positive")
    num = P_K * c["K_out"] + P_Na * c["Na_out"] + P_Cl * c["Cl_in"]
    den = P_K * c["K_in"] + P_Na * c["Na_in"] + P_Cl * c["Cl_out"]
    if num <= 0.0 or den <= 0.0:
        raise ValueError("weighted concentration sums must be positive")
    slope = _BSA_R_GAS * T / _BSA_FARADAY
    volts = slope * log(num / den)
    return RichResult(payload={
        "potential_mV": volts * 1000.0,
        "potential_V": volts,
        "numerator_mM": num,
        "denominator_mM": den,
        "slope_mV": slope * 1000.0,
        "permeabilities": {"K": P_K, "Na": P_Na, "Cl": P_Cl},
        "T_K": T,
        "units": {"potential": "mV", "concentration": "mM", "T": "K"},
        "method": "Goldman (1943) / Hodgkin & Katz (1949) GHK voltage equation; not given in Rangayyan (2024)",
    })


rangayyan_goldman_eqn = ghk  # pre-policy spelling


# -- rghgate: Hodgkin-Huxley gating variable ODEs (m, h, n).
def hhgate(V, dt=0.01, m=None, h=None, n=None, steps=1):
    """Hodgkin-Huxley activation/inactivation gating variables m, h and n.

    Rangayyan (2024) describes the Hodgkin-Huxley model in Section 7.8.1
    and gives the channel-current form, Equation (7.137),
    I_i = g_i N (V_m - E_i) P(o), but the book does NOT print the gating
    rate constants.  Those are taken from the primary source, which the
    book itself cites as reference [54]:

        Hodgkin AL, Huxley AF, "A quantitative description of membrane
        current and its application to conduction and excitation in
        nerve", Journal of Physiology 117(4):500-544, 1952.

    Each gate obeys a first-order kinetic equation

        dx/dt = alpha_x(V) (1 - x) - beta_x(V) x,     x in {m, h, n},

    whose steady state is x_inf = alpha / (alpha + beta) and whose time
    constant is tau_x = 1 / (alpha + beta).

    WHY: m, h and n are the fractions of open sub-gates whose products
    m^3 h and n^4 scale the sodium and potassium conductances.  The fast
    m gate produces the upstroke of the action potential, the slow h gate
    terminates it, and the slow n gate repolarises the membrane; the
    ordering of their time constants is what makes a spike a spike rather
    than a monotone charging curve.

    Parameters
    ----------
    V : float
        Membrane potential in millivolts (mV), absolute (rest = -65 mV).
    dt : float
        Integration step in milliseconds (ms).  Must be positive.
    m, h, n : float or None
        Initial gate values, dimensionless in [0, 1].  ``None`` starts the
        gate at its steady-state value for the given V.
    steps : int
        Number of exponential-Euler steps of length ``dt`` to take at the
        clamped potential V.  Must be >= 1.

    Returns
    -------
    RichResult
        ``m``, ``h``, ``n`` after integration (dimensionless);
        ``m_inf``, ``h_inf``, ``n_inf`` steady states (dimensionless);
        ``tau_m_ms``, ``tau_h_ms``, ``tau_n_ms`` time constants in ms;
        ``alpha_per_ms`` and ``beta_per_ms`` dicts of rate constants in 1/ms.

    Notes
    -----
    The step is the exponential (Rush-Larsen) update
    x <- x_inf + (x - x_inf) exp(-dt / tau), which is the exact solution
    at a clamped potential and is therefore stable for any dt.
    """
    V = float(V)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive (ms)")
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    am, bm, ah, bh, an, bn = _bsahhrates(V)
    out = {}
    for name, a, b, x0 in (("m", am, bm, m), ("h", ah, bh, h), ("n", an, bn, n)):
        tot = a + b
        if tot <= 0.0:
            raise ValueError("degenerate rate constants for gate " + name)
        xinf = a / tot
        tau = 1.0 / tot
        if x0 is None:
            x = xinf
        else:
            x = float(x0)
            if not 0.0 <= x <= 1.0:
                raise ValueError("gate " + name + " must start in [0, 1]")
            x = xinf + (x - xinf) * exp(-steps * dt / tau)
        out[name] = x
        out[name + "_inf"] = xinf
        out["tau_" + name + "_ms"] = tau
    return RichResult(payload={
        "V_mV": V,
        "dt_ms": dt,
        "steps": steps,
        "m": out["m"], "h": out["h"], "n": out["n"],
        "m_inf": out["m_inf"], "h_inf": out["h_inf"], "n_inf": out["n_inf"],
        "tau_m_ms": out["tau_m_ms"], "tau_h_ms": out["tau_h_ms"],
        "tau_n_ms": out["tau_n_ms"],
        "alpha_per_ms": {"m": am, "h": ah, "n": an},
        "beta_per_ms": {"m": bm, "h": bh, "n": bn},
        "units": {"V": "mV", "time": "ms", "rates": "1/ms", "gates": "dimensionless"},
        "method": "Hodgkin & Huxley (1952) J Physiol 117(4):500-544 gating kinetics; rates not printed in Rangayyan (2024)",
    })


rangayyan_hh_gating = hhgate  # pre-policy spelling


# -- rghhmm: Hodgkin-Huxley membrane model for action potential.
def hhmodel(duration=30.0, dt=0.01, I_ext=10.0, stim_start=5.0, stim_stop=6.0,
            C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
            E_Na=50.0, E_K=-77.0, E_L=-54.387, V0=-65.0):
    """Four-variable Hodgkin-Huxley membrane model, integrated with RK4.

    Rangayyan (2024) Section 7.8.1 presents the model qualitatively with
    the circuit of Figure 7.29 and the membrane current-balance equation

        dV/dt = -(I_ion + I_stim) / C_m,                        eq. (7.138)

    (note the sign convention of the book, in which I_stim adds to the
    outward ionic current).  The book does NOT print the conductance
    expressions or the gating rate constants; those come from the primary
    source that the book cites as its reference [54]:

        Hodgkin AL, Huxley AF, "A quantitative description of membrane
        current and its application to conduction and excitation in
        nerve", Journal of Physiology 117(4):500-544, 1952.

    The full system integrated here is

        C_m dV/dt = I_stim - g_Na m^3 h (V - E_Na)
                            - g_K n^4 (V - E_K) - g_L (V - E_L),
        dx/dt = alpha_x(V)(1 - x) - beta_x(V) x   for x in {m, h, n},

    with I_stim taken as a depolarising inward current (positive I_stim
    drives V upward), the sign convention used in essentially all modern
    restatements of the model.

    WHY: this is the model that explains the action potential rather than
    merely describing its shape.  The regenerative sodium current gives
    the all-or-none upstroke and the threshold; sodium inactivation plus
    delayed potassium activation give repolarisation, the undershoot and
    the refractory period.  A curve-fitted spike template reproduces none
    of that.

    Parameters
    ----------
    duration : float
        Simulated time in milliseconds (ms).
    dt : float
        Fixed RK4 step in milliseconds (ms).  0.01 ms is the default and
        is comfortably below the stability limit of the fast m gate.
    I_ext : float
        Stimulus current density in microamperes per square centimetre
        (uA/cm^2), applied for stim_start <= t < stim_stop.
    stim_start, stim_stop : float
        Stimulus window in milliseconds (ms).
    C_m : float
        Membrane capacitance in microfarads per square centimetre
        (uF/cm^2).  Hodgkin-Huxley value 1.0.
    g_Na, g_K, g_L : float
        Maximal conductances in millisiemens per square centimetre
        (mS/cm^2).  Hodgkin-Huxley values 120, 36, 0.3.
    E_Na, E_K, E_L : float
        Reversal potentials in millivolts (mV).  Values 50, -77, -54.387
        place the resting potential at -65 mV.
    V0 : float
        Initial membrane potential in millivolts (mV); the gates start at
        their steady-state values for V0.

    Returns
    -------
    RichResult
        ``t_ms`` and ``V_mV`` time series; ``m``, ``h``, ``n`` gate series;
        ``I_Na``, ``I_K``, ``I_L`` current-density series in uA/cm^2;
        ``peak_mV`` and ``peak_time_ms``; ``rest_mV`` (the potential just
        before the stimulus); ``min_mV`` (the after-hyperpolarisation);
        ``spiked`` (True if the potential overshot 0 mV);
        ``n_spikes`` (upward crossings of 0 mV).
    """
    duration, dt = float(duration), float(dt)
    if duration <= 0.0:
        raise ValueError("duration must be positive (ms)")
    if not 0.0 < dt <= 1.0:
        raise ValueError("dt must be in (0, 1] ms for a stable RK4 integration")
    if stim_stop < stim_start:
        raise ValueError("stim_stop must not precede stim_start")
    if C_m <= 0.0:
        raise ValueError("C_m must be positive (uF/cm^2)")
    if min(float(g_Na), float(g_K), float(g_L)) < 0.0:
        raise ValueError("conductances must be non-negative (mS/cm^2)")
    C_m = float(C_m)
    g_Na, g_K, g_L = float(g_Na), float(g_K), float(g_L)
    E_Na, E_K, E_L = float(E_Na), float(E_K), float(E_L)
    I_ext = float(I_ext)
    stim_start, stim_stop = float(stim_start), float(stim_stop)

    def stim(t):
        return I_ext if stim_start <= t < stim_stop else 0.0

    def deriv(t, V, m, h, n):
        am, bm, ah, bh, an, bn = _bsahhrates(V)
        iNa = g_Na * m ** 3 * h * (V - E_Na)
        iK = g_K * n ** 4 * (V - E_K)
        iL = g_L * (V - E_L)
        dV = (stim(t) - iNa - iK - iL) / C_m
        return dV, am * (1.0 - m) - bm * m, ah * (1.0 - h) - bh * h, \
            an * (1.0 - n) - bn * n

    V = float(V0)
    am, bm, ah, bh, an, bn = _bsahhrates(V)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    nsteps = int(round(duration / dt))
    ts, Vs, ms, hs, ns = [0.0], [V], [m], [h], [n]
    for i in range(nsteps):
        t = i * dt
        y = (V, m, h, n)
        k1 = deriv(t, *y)
        k2 = deriv(t + dt / 2, *[y[j] + dt / 2 * k1[j] for j in range(4)])
        k3 = deriv(t + dt / 2, *[y[j] + dt / 2 * k2[j] for j in range(4)])
        k4 = deriv(t + dt, *[y[j] + dt * k3[j] for j in range(4)])
        V, m, h, n = [y[j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                      for j in range(4)]
        m = min(1.0, max(0.0, m))
        h = min(1.0, max(0.0, h))
        n = min(1.0, max(0.0, n))
        ts.append((i + 1) * dt)
        Vs.append(V)
        ms.append(m)
        hs.append(h)
        ns.append(n)
    iNa = [g_Na * ms[k] ** 3 * hs[k] * (Vs[k] - E_Na) for k in range(len(ts))]
    iK = [g_K * ns[k] ** 4 * (Vs[k] - E_K) for k in range(len(ts))]
    iL = [g_L * (Vs[k] - E_L) for k in range(len(ts))]
    peak = max(Vs)
    pk = Vs.index(peak)
    rest_idx = max(0, int(stim_start / dt) - 1)
    crossings = sum(1 for k in range(1, len(Vs))
                    if Vs[k - 1] <= 0.0 < Vs[k])
    return RichResult(payload={
        "t_ms": ts, "V_mV": Vs, "m": ms, "h": hs, "n": ns,
        "I_Na_uA_cm2": iNa, "I_K_uA_cm2": iK, "I_L_uA_cm2": iL,
        "peak_mV": peak, "peak_time_ms": ts[pk],
        "rest_mV": Vs[rest_idx], "min_mV": min(Vs),
        "spiked": peak > 0.0, "n_spikes": crossings,
        "dt_ms": dt, "I_ext_uA_cm2": I_ext,
        "units": {"V": "mV", "t": "ms", "I": "uA/cm^2",
                  "g": "mS/cm^2", "C_m": "uF/cm^2"},
        "method": "Hodgkin & Huxley (1952) J Physiol 117(4):500-544, four-variable model, RK4; Rangayyan (2024) eq. (7.138), Section 7.8.1",
    })


rangayyan_hodgkin_huxley = hhmodel  # pre-policy spelling


# -- rghmm: FitzHugh-Nagumo simplified neuron model.
def fhn(duration=200.0, dt=0.01, I_ext=0.5, a=0.7, b=0.8, eps=0.08,
        v0=-1.2, w0=-0.6, stim_start=0.0, stim_stop=None):
    """FitzHugh-Nagumo two-variable excitable-medium neuron model.

    Rangayyan (2024) mentions the FitzHugh-Nagumo equations only in
    passing, in Section 7.8.3, where the Rogers-McCulloch cardiac model is
    described as being "based on the FitzHugh-Nagumo equations ... for
    excitable media".  The book does NOT print the equations.  They are
    taken from the primary sources:

        FitzHugh R, "Impulses and physiological states in theoretical
        models of nerve membrane", Biophysical Journal 1(6):445-466, 1961;
        Nagumo J, Arimoto S, Yoshizawa S, "An active pulse transmission
        line simulating nerve axon", Proceedings of the IRE
        50(10):2061-2070, 1962.

        dv/dt = v - v^3/3 - w + I_ext,
        dw/dt = eps (v + a - b w).

    WHY: FitzHugh-Nagumo is the minimal caricature of Hodgkin-Huxley.  It
    keeps the two features that matter -- a fast cubic regenerative
    variable and a slow linear recovery variable -- and discards the four
    dimensions and the empirical rate curves.  That makes threshold,
    refractoriness and repetitive firing analysable in a phase plane,
    which is why it is the model of choice for tissue-scale excitable
    media simulations.

    Parameters
    ----------
    duration : float
        Simulated time in dimensionless model time units (the FitzHugh-
        Nagumo system carries no physical units; one time unit is
        conventionally read as a few milliseconds of real membrane time).
    dt : float
        RK4 step, same dimensionless time units.
    I_ext : float
        Applied current, dimensionless.  With the classical parameters
        below, values around 0.32 to 1.4 give sustained oscillation.
    a, b, eps : float
        Classical FitzHugh (1961) parameters 0.7, 0.8, 0.08.  ``eps`` sets
        the separation of the fast and slow time scales and must be
        positive; ``b`` must be positive for the recovery variable to be
        restoring.
    v0, w0 : float
        Initial fast and recovery variables, dimensionless.
    stim_start, stim_stop : float
        Window over which I_ext is applied; ``stim_stop=None`` means the
        current stays on to the end of the simulation.

    Returns
    -------
    RichResult
        ``t`` , ``v`` , ``w`` series (all dimensionless);
        ``peak`` , ``min`` , ``n_spikes`` (upward crossings of v = 1.0,
        the conventional spike marker for this model);
        ``period`` -- mean interval between successive spikes, or None if
        fewer than two spikes occurred.
    """
    duration, dt = float(duration), float(dt)
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if not 0.0 < dt <= 1.0:
        raise ValueError("dt must be in (0, 1] for a stable RK4 integration")
    a, b, eps = float(a), float(b), float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive (fast/slow time-scale ratio)")
    if b <= 0.0:
        raise ValueError("b must be positive for a restoring recovery variable")
    I_ext = float(I_ext)
    stim_start = float(stim_start)
    stop = duration if stim_stop is None else float(stim_stop)
    if stop < stim_start:
        raise ValueError("stim_stop must not precede stim_start")

    def deriv(t, v, w):
        cur = I_ext if stim_start <= t < stop else 0.0
        return v - v ** 3 / 3.0 - w + cur, eps * (v + a - b * w)

    v, w = float(v0), float(w0)
    ts, vs, ws = [0.0], [v], [w]
    for i in range(int(round(duration / dt))):
        t = i * dt
        k1 = deriv(t, v, w)
        k2 = deriv(t + dt / 2, v + dt / 2 * k1[0], w + dt / 2 * k1[1])
        k3 = deriv(t + dt / 2, v + dt / 2 * k2[0], w + dt / 2 * k2[1])
        k4 = deriv(t + dt, v + dt * k3[0], w + dt * k3[1])
        v += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ts.append((i + 1) * dt)
        vs.append(v)
        ws.append(w)
    spikes = [ts[k] for k in range(1, len(vs)) if vs[k - 1] <= 1.0 < vs[k]]
    period = None
    if len(spikes) > 1:
        period = (spikes[-1] - spikes[0]) / (len(spikes) - 1)
    return RichResult(payload={
        "t": ts, "v": vs, "w": ws,
        "peak": max(vs), "min": min(vs),
        "n_spikes": len(spikes), "spike_times": spikes, "period": period,
        "a": a, "b": b, "eps": eps, "I_ext": I_ext,
        "units": {"v": "dimensionless", "w": "dimensionless",
                  "t": "dimensionless model time units"},
        "method": "FitzHugh (1961) Biophys J 1(6):445-466 / Nagumo et al. (1962) Proc IRE 50(10):2061-2070; equations not printed in Rangayyan (2024), named in Section 7.8.3",
    })


rangayyan_fitzhugh_nagumo = fhn  # pre-policy spelling


# -- rgmemb: Membrane potential dynamics (RC circuit model).
def rcmemb(t, I_inj=0.0, C_m=0.2, R_m=100.0, V_rest=-65.0):
    """Passive RC (leaky-integrator) membrane potential.

    This is the sub-threshold limit of the membrane current balance that
    Rangayyan (2024) gives as Equation (7.138) in Section 7.8.1,
    dV/dt = -(I_ion + I_stim) / C_m, with the ionic current reduced to a
    single ohmic leak, I_leak = (V - V_rest) / R_m:

        C_m dV/dt = I_inj - (V - V_rest) / R_m,
        tau = R_m C_m,
        V(t) = V_rest + I_inj R_m (1 - exp(-t / tau))   for a step input.

    WHY: below threshold no voltage-gated conductance is engaged, so the
    membrane is just a capacitor in parallel with a resistor.  The single
    time constant tau = R_m C_m is what sets the temporal summation window
    of a neuron, i.e. how far apart two inputs can be and still add.  The
    integration used here is the exponential (exact for piecewise-constant
    input) update, so the returned trace carries no step-size error.

    Parameters
    ----------
    t : array-like or float
        Time points in milliseconds (ms).  A scalar n is read as n evenly
        spaced samples is NOT supported: pass the actual time vector.
        Must be non-decreasing.
    I_inj : float or array-like
        Injected current in nanoamperes (nA).  A scalar is held constant;
        a sequence must match ``t`` in length and is treated as constant
        over each interval.
    C_m : float
        Membrane capacitance in nanofarads (nF).  Must be positive.
    R_m : float
        Membrane resistance in megohms (MOhm).  Must be positive.
        With these units tau = R_m * C_m comes out directly in ms, and
        I_inj * R_m comes out directly in mV.
    V_rest : float
        Resting potential in millivolts (mV).

    Returns
    -------
    RichResult
        ``t_ms``, ``V_mV`` traces; ``tau_ms`` = R_m C_m;
        ``V_steady_mV`` = V_rest + I_inj R_m for the final current;
        ``input_resistance_MOhm``; ``peak_mV``, ``final_mV``.
    """
    ts = [float(v) for v in aslist(t)]
    if len(ts) < 1:
        raise ValueError("t must contain at least one time point (ms)")
    if any(ts[i + 1] < ts[i] for i in range(len(ts) - 1)):
        raise ValueError("t must be non-decreasing (ms)")
    C_m, R_m, V_rest = float(C_m), float(R_m), float(V_rest)
    if C_m <= 0.0:
        raise ValueError("C_m must be positive (nF)")
    if R_m <= 0.0:
        raise ValueError("R_m must be positive (MOhm)")
    try:
        cur = [float(I_inj)] * len(ts)
    except TypeError:
        cur = [float(v) for v in aslist(I_inj)]
        if len(cur) != len(ts):
            raise ValueError("I_inj must be scalar or the same length as t")
    tau = R_m * C_m
    V = V_rest
    Vs = []
    for i, ti in enumerate(ts):
        if i > 0:
            step = ts[i] - ts[i - 1]
            vinf = V_rest + cur[i - 1] * R_m
            V = vinf + (V - vinf) * exp(-step / tau)
        Vs.append(V)
    return RichResult(payload={
        "t_ms": ts, "V_mV": Vs,
        "tau_ms": tau,
        "V_steady_mV": V_rest + cur[-1] * R_m,
        "input_resistance_MOhm": R_m,
        "peak_mV": max(Vs), "final_mV": Vs[-1], "V_rest_mV": V_rest,
        "units": {"V": "mV", "t": "ms", "I": "nA", "R": "MOhm",
                  "C": "nF", "tau": "ms"},
        "method": "Passive RC membrane, leak-only reduction of Rangayyan (2024) eq. (7.138), Section 7.8.1",
    })


rangayyan_membrane_potential = rcmemb  # pre-policy spelling


# -- rgmscart: Muscle contraction artifact removal from VAG signals.
def vagclean(vag, emg_ref, fs, n_taps=8, mu=0.05, alpha=0.02, adaptive_mu=True):
    """Cancel muscle-contraction artifact from a VAG signal by adaptive LMS.

    Rangayyan (2024) Section 3.3.6 identifies muscle-contraction
    interference in VAG signals as the problem, and Section 3.15
    ("Application: Muscle-contraction Interference") together with
    Sections 3.10.1 (the adaptive noise canceller) and 3.10.2 (the LMS
    filter) gives the solution.  The book reports that Zhang et al. used a
    two-stage adaptive LMS filter to cancel muscle-contraction
    interference from VAG signals, with the muscle signal recorded at the
    distal rectus femoris used as the reference input.  The update is the
    Widrow-Hoff LMS algorithm,

        w(n+1) = w(n) + 2 mu e(n) r(n),                       eq. (3.203)

    and the Zhang et al. variant makes the step size time varying,

        w(n+1) = w(n) + 2 mu(n) e(n) r(n),                    eq. (3.204)
        mu(n) = mu / ( (M+1) xbar2(n) ),                      eq. (3.205)
        xbar2(n) = alpha r^2(n) + (1 - alpha) xbar2(n-1),

    with 0 < mu < 1 and a forgetting factor 0 <= alpha << 1 introduced to
    cope with the nonstationarity of the signals.  M+1 is the number of
    filter taps.

    The canceller output is the estimation ERROR e(n) = x(n) - w'(n) r(n):
    the filter learns the part of the primary input that the reference can
    predict, and what is left over is the cleaned VAG.

    WHY: the muscle signal and the VAG signal overlap in frequency, so no
    fixed filter can separate them.  What distinguishes them is that the
    interference -- and only the interference -- is correlated with a
    reference recorded over the muscle.  An adaptive canceller exploits
    exactly that correlation, and it keeps working when the coupling
    between muscle and accelerometer changes during the swing cycle, which
    a fixed filter cannot.

    Parameters
    ----------
    vag : array-like
        Primary input: the VAG recording containing the artifact, in
        arbitrary amplitude units (mV at the accelerometer).
    emg_ref : array-like
        Reference input: the muscle-contraction signal, same length and
        sampling rate.  It must be correlated with the interference and
        UNCORRELATED with the VAG signal of interest, or the canceller
        will remove signal along with artifact.
    fs : float
        Sampling rate in hertz (Hz); positive.
    n_taps : int
        Number of filter taps, M+1 in eq. (3.205); >= 1.
    mu : float
        Step size, dimensionless.  With ``adaptive_mu`` it is the mu of
        eq. (3.205) and must lie in (0, 1).  Without it, it is the fixed
        mu of eq. (3.203) and must be positive and small enough for
        stability: the book states convergence requires mu below the
        reciprocal of the largest eigenvalue of the reference
        autocorrelation matrix, and this block checks the sufficient
        condition mu < 1 / (n_taps * reference power).
    alpha : float
        Forgetting factor of eq. (3.205), 0 <= alpha << 1.
    adaptive_mu : bool
        True selects eqs. (3.204)-(3.205); False the fixed-step
        eq. (3.203).

    Returns
    -------
    RichResult
        ``cleaned`` -- the estimation error e(n), the artifact-free VAG,
        same units as the input; ``artifact_estimate`` -- w'(n) r(n);
        ``weights`` -- the final tap-weight vector;
        ``rms_before`` and ``rms_after``;
        ``artifact_reduction_db`` -- 20 log10(rms_before / rms_after),
        positive when the artifact has been reduced;
        ``mu_trace`` -- the step size actually used at each sample.
    """
    xs = [float(v) for v in aslist(vag)]
    rs = [float(v) for v in aslist(emg_ref)]
    if len(xs) != len(rs):
        raise ValueError("vag and emg_ref must have the same length")
    if len(xs) < 8:
        raise ValueError("need at least 8 samples")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    M1 = int(n_taps)
    if M1 < 1:
        raise ValueError("n_taps must be at least 1")
    mu = float(mu)
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1 (forgetting factor)")
    rpow = fsum(v * v for v in rs) / len(rs)
    if rpow <= 0.0:
        raise ValueError("emg_ref is identically zero; nothing to cancel with")
    if adaptive_mu:
        if not 0.0 < mu < 1.0:
            raise ValueError("with adaptive_mu, mu must lie in (0, 1) per eq. (3.205)")
    else:
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        lim = 1.0 / (M1 * rpow)
        if mu >= lim:
            raise ValueError("mu=%g exceeds the stability limit %g for a "
                             "reference of power %g; the LMS filter would diverge"
                             % (mu, lim, rpow))
    w = [0.0] * M1
    xbar2 = rpow
    out, art, mus = [], [], []
    for n in range(len(xs)):
        r = [rs[n - k] if n - k >= 0 else 0.0 for k in range(M1)]
        y = fsum(w[k] * r[k] for k in range(M1))
        e = xs[n] - y
        if adaptive_mu:
            xbar2 = alpha * r[0] * r[0] + (1.0 - alpha) * xbar2
            step = mu / (M1 * xbar2) if xbar2 > 0.0 else 0.0
        else:
            step = mu
        for k in range(M1):
            w[k] += 2.0 * step * e * r[k]
        out.append(e)
        art.append(y)
        mus.append(step)
    rb, ra = _bsarms(xs), _bsarms(out)
    return RichResult(payload={
        "cleaned": out, "artifact_estimate": art, "weights": w,
        "rms_before": rb, "rms_after": ra,
        "artifact_reduction_db": (20.0 * log(rb / ra, 10.0)
                                  if ra > 0.0 and rb > 0.0 else None),
        "mu_trace": mus, "n_taps": M1, "alpha": alpha,
        "adaptive_mu": bool(adaptive_mu), "fs_hz": fs,
        "units": {"signals": "input amplitude units (mV)",
                  "reduction": "dB", "mu": "dimensionless"},
        "method": "Rangayyan (2024) eqs. (3.203), (3.204), (3.205), Sections 3.10.1, 3.10.2, 3.3.6 and 3.15, after Zhang et al.",
    })


rangayyan_muscle_artifact = vagclean  # pre-policy spelling


# -- rgmuap: Motor unit action potential (MUAP) model.
def muapmodel(t, n_fibers=25, conduction_vel=4.0, spread_mm=3.0,
              amp_uV=8.0, width_ms=1.0, phases=3):
    """Motor unit action potential as the superposition of fibre potentials.

    Rangayyan (2024) Section 1.2.4 ("The electromyogram (EMG)") defines
    the single-motor-unit action potential (SMUAP, or MUAP) as "the
    summation of the action potentials of all of its constituent cells",
    and states that normal SMUAPs are biphasic or triphasic, 3 to 15 ms
    in duration, 100 to 300 microvolts in amplitude, and fire at 6 to
    30 per second.  The book gives no closed-form SMUAP equation, so the
    per-fibre waveform used here is stated explicitly rather than cited:
    each single-fibre potential is a Gaussian derivative,

        biphasic  (phases=2):  s(u) = -u exp(-u^2 / 2),
        triphasic (phases=3):  s(u) = (u^2 - 1) exp(-u^2 / 2),

    with u = (t - delay) / sigma, which are the standard shapes of the
    first and second derivative of a Gaussian and reproduce the biphasic
    and triphasic morphologies the book describes.

    WHY: the recorded MUAP is not a fibre potential, it is the sum of the
    potentials of every fibre in the unit, each arriving at the electrode
    at a slightly different time because the fibres differ in end-plate
    position and in the distance from the electrode.  That temporal
    dispersion is the whole reason MUAP duration (3-15 ms) is an order of
    magnitude longer than a single fibre potential, and it is why
    increased dispersion -- from reinnervation, for example -- shows up as
    a polyphasic, prolonged MUAP.

    Parameters
    ----------
    t : array-like
        Time points in milliseconds (ms).
    n_fibers : int
        Number of muscle fibres in the motor unit, dimensionless.  Must
        be >= 1.  The book quotes innervation ratios from about 2-3
        (laryngeal muscles) to about 1,900 (medial gastrocnemius).
    conduction_vel : float
        Muscle-fibre conduction velocity in metres per second (m/s);
        must be positive.  3-5 m/s is the usual range.
    spread_mm : float
        Total spread of the fibre end-plate positions along the fibre
        direction, in millimetres (mm).  Fibre delays are spread evenly
        over spread_mm / conduction_vel milliseconds, symmetric about
        zero.  Note 1 mm / (1 m/s) = 1 ms, so the delay spread in ms is
        spread_mm / conduction_vel directly.
    amp_uV : float
        Peak-to-peak amplitude contributed by one fibre, in microvolts
        (uV).
    width_ms : float
        Gaussian width parameter sigma of one fibre potential, in
        milliseconds (ms); must be positive.
    phases : int
        2 for a biphasic single-fibre potential, 3 for triphasic.

    Returns
    -------
    RichResult
        ``t_ms``, ``muap_uV`` waveform; ``peak_to_peak_uV``;
        ``duration_ms`` -- the span over which |MUAP| exceeds 5 percent of
        its peak, the conventional MUAP duration measure;
        ``n_phases_observed`` -- baseline crossings plus one, so that a
        value above 4 marks a polyphasic (abnormal) MUAP;
        ``delays_ms`` -- the per-fibre arrival delays;
        ``in_normal_duration_band`` -- True when duration_ms lies in the
        3-15 ms band the book quotes for normal SMUAPs.
    """
    ts = [float(v) for v in aslist(t)]
    if len(ts) < 3:
        raise ValueError("t must contain at least 3 time points (ms)")
    n_fibers = int(n_fibers)
    if n_fibers < 1:
        raise ValueError("n_fibers must be at least 1")
    conduction_vel = float(conduction_vel)
    if conduction_vel <= 0.0:
        raise ValueError("conduction_vel must be positive (m/s)")
    width_ms = float(width_ms)
    if width_ms <= 0.0:
        raise ValueError("width_ms must be positive (ms)")
    spread_mm = float(spread_mm)
    if spread_mm < 0.0:
        raise ValueError("spread_mm must be non-negative (mm)")
    phases = int(phases)
    if phases not in (2, 3):
        raise ValueError("phases must be 2 (biphasic) or 3 (triphasic)")
    amp_uV = float(amp_uV)
    span_ms = spread_mm / conduction_vel
    if n_fibers == 1:
        delays = [0.0]
    else:
        delays = [-span_ms / 2.0 + span_ms * i / (n_fibers - 1)
                  for i in range(n_fibers)]
    tmid = 0.5 * (ts[0] + ts[-1])
    wave = []
    for ti in ts:
        acc = 0.0
        for dly in delays:
            u = (ti - tmid - dly) / width_ms
            if abs(u) > 8.0:
                continue
            e = exp(-0.5 * u * u)
            acc += (-u * e) if phases == 2 else ((u * u - 1.0) * e)
        wave.append(amp_uV * acc)
    pk = max(abs(v) for v in wave)
    if pk <= 0.0:
        raise ValueError("MUAP is identically zero; widen t to cover the waveform")
    thr = 0.05 * pk
    on = [ts[i] for i in range(len(ts)) if abs(wave[i]) >= thr]
    dur = (on[-1] - on[0]) if len(on) > 1 else 0.0
    cross = sum(1 for i in range(1, len(wave))
                if (wave[i - 1] < 0.0) != (wave[i] < 0.0))
    return RichResult(payload={
        "t_ms": ts, "muap_uV": wave,
        "peak_to_peak_uV": max(wave) - min(wave),
        "peak_uV": pk,
        "duration_ms": dur,
        "n_phases_observed": cross + 1,
        "delays_ms": delays,
        "n_fibers": n_fibers,
        "conduction_vel_m_s": conduction_vel,
        "in_normal_duration_band": 3.0 <= dur <= 15.0,
        "units": {"t": "ms", "muap": "uV", "conduction velocity": "m/s"},
        "method": "MUAP as summed single-fibre potentials; morphology and normal ranges from Rangayyan (2024) Section 1.2.4 (no waveform equation given in the book)",
    })


rangayyan_muap = muapmodel  # pre-policy spelling


# -- rgmurm: Heart murmur frequency analysis for valvular defect diagnosis.
def murmspec(pcg, fs, f1=25.0, f2=75.0, f3=150.0):
    """Murmur frequency analysis for the diagnosis of valvular defects.

    Rangayyan (2024) Section 6.2.2 ("Frequency analysis of murmurs to
    diagnose valvular defects") and Section 6.4.2 give the measure of
    Johnson et al., who compared the integral of the MAGNITUDE spectrum of
    the systolic murmur of aortic stenosis over a high band to that over a
    low band, eq. (6.45):

        PA / CA = integral_{f2}^{f3} |X(f)| df
                  / integral_{f1}^{f2} |X(f)| df,

    with f1 = 25 Hz, f2 = 75 Hz and f3 = 150 Hz.  The band f2:f3 is the
    "predictive area" (PA) related to aortic stenosis; the band f1:f2 is
    the "constant area" (CA) common to all systolic PCG segments, which
    serves as the reference.  Rangayyan reports that Johnson et al. showed
    the ratio correlates well with the severity of aortic stenosis.

    NOTE: eq. (6.45) integrates the magnitude spectrum |X(f)|, NOT the
    power spectrum.  This block follows the book and integrates
    magnitudes; squaring them would change the ratio and is a different
    measure.

    WHY: a systolic murmur is turbulence downstream of a narrowed valve,
    and the tighter the stenosis the higher the velocity and therefore the
    higher the frequency of the resulting sound.  Normalising by a
    lower-frequency band that every systolic segment contains removes the
    dependence on how hard the microphone was pressed to the chest, which
    is what makes the ratio comparable between recordings and patients.

    Parameters
    ----------
    pcg : array-like
        Systolic segment of the PCG, in arbitrary amplitude units.  At
        least 4 samples.
    fs : float
        Sampling rate in hertz (Hz); positive.  f3 must be below fs/2.
    f1, f2, f3 : float
        Band edges in hertz (Hz), strictly increasing.  Defaults 25, 75
        and 150 Hz are the values in eq. (6.45).

    Returns
    -------
    RichResult
        ``pa_over_ca`` -- the ratio of eq. (6.45), dimensionless;
        ``predictive_area`` and ``constant_area`` -- the two magnitude
        integrals; ``mean_freq_hz``, ``median_freq_hz``, ``spread_hz``,
        ``spectral_skewness``, ``spectral_kurtosis`` of the power
        spectrum; ``dominant_freq_hz``; ``freq_hz``, ``magnitude`` and
        ``psd``.
    """
    fs = float(fs)
    f1, f2, f3 = float(f1), float(f2), float(f3)
    if not f1 < f2 < f3:
        raise ValueError("band edges must satisfy f1 < f2 < f3 (Hz)")
    if f3 > fs / 2.0:
        raise ValueError("f3 exceeds the Nyquist frequency")
    freqs, psd = _bsapsd(pcg, fs)
    mag = [sqrt(v) for v in psd]
    ca = fsum(m for f, m in zip(freqs, mag) if f1 <= f < f2)
    pa = fsum(m for f, m in zip(freqs, mag) if f2 <= f < f3)
    if ca <= 0.0:
        raise ValueError("constant-area band %g-%g Hz carries no energy" % (f1, f2))
    mom = _bsapsdmom(freqs, psd)
    pk = _bsapeaks(freqs, psd, count=3)
    out = dict(mom)
    out.update({
        "pa_over_ca": pa / ca,
        "predictive_area": pa, "constant_area": ca,
        "bands_hz": {"CA": (f1, f2), "PA": (f2, f3)},
        "dominant_freq_hz": pk[0][0] if pk else None,
        "peaks_hz": pk,
        "freq_hz": freqs, "magnitude": mag, "psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "pa_over_ca": "dimensionless"},
        "method": "Rangayyan (2024) eq. (6.45), Sections 6.2.2 and 6.4.2, after Johnson et al.",
    })
    return RichResult(payload=out)


rangayyan_murmur_analysis = murmspec  # pre-policy spelling


# -- rgnrnst: Nernst equilibrium potential for ionic species.
def nernst(T=310.15, z=1, conc_out=5.0, conc_in=140.0, ion="K+"):
    """Nernst equilibrium (reverse) potential of a single ionic species.

    Rangayyan (2024) gives this as Equation (7.139), in Section 7.8.1
    ("Electrophysiological modeling at the cellular level"):

        E_x = (R T) / (z F) * log(X_out / X_in),

    where R is the gas constant, T the absolute temperature, z the ionic
    charge, F Faraday's constant, and X_out/X_in the ratio of the ionic
    concentration outside the cell to that inside.  The logarithm is the
    natural logarithm.

    WHY: the reverse potential is the membrane potential at which the
    electrical force on the ion exactly cancels its concentration gradient,
    so no net current of that species flows.  It sets the driving force
    (V_m - E_x) that every conductance in a Hodgkin-Huxley-type model is
    multiplied by, which is why it must be computed before any membrane
    model can be integrated.

    Parameters
    ----------
    T : float
        Absolute temperature in kelvin (K).  Default 310.15 K = 37 degC.
    z : int or float
        Valence of the ion, dimensionless and signed (+1 for K+ and Na+,
        +2 for Ca2+, -1 for Cl-).  Must be non-zero.
    conc_out, conc_in : float
        Extracellular and intracellular concentrations, in millimolar (mM).
        Any consistent unit works since only their ratio enters, but the
        payload reports them as mM.
    ion : str
        Label carried through to the payload; does not affect the result.

    Returns
    -------
    RichResult
        ``potential_mV`` -- equilibrium potential in millivolts (mV);
        ``potential_V`` -- the same in volts (V);
        ``slope_mV`` -- the factor R T / (z F) in mV, i.e. the potential
        change per e-fold change in the concentration ratio;
        ``ratio`` -- conc_out / conc_in (dimensionless).

    Examples
    --------
    Potassium at 37 degC with 5 mM outside and 140 mM inside gives about
    -89 mV, the textbook resting potassium equilibrium potential.
    """
    T = float(T)
    z = float(z)
    conc_out = float(conc_out)
    conc_in = float(conc_in)
    if T <= 0.0:
        raise ValueError("T must be a positive absolute temperature in kelvin")
    if z == 0.0:
        raise ValueError("z (ionic valence) must be non-zero")
    if conc_out <= 0.0 or conc_in <= 0.0:
        raise ValueError("ion concentrations must be positive (mM)")
    slope = _BSA_R_GAS * T / (z * _BSA_FARADAY)
    volts = slope * log(conc_out / conc_in)
    return RichResult(payload={
        "ion": str(ion),
        "potential_mV": volts * 1000.0,
        "potential_V": volts,
        "slope_mV": slope * 1000.0,
        "ratio": conc_out / conc_in,
        "T_K": T,
        "z": z,
        "conc_out_mM": conc_out,
        "conc_in_mM": conc_in,
        "units": {"potential": "mV", "T": "K", "concentration": "mM"},
        "method": "Nernst equilibrium potential, Rangayyan (2024) eq. (7.139), Section 7.8.1",
    })


rangayyan_nernst_potential = nernst  # pre-policy spelling


# -- rgoae: Otoacoustic emission (OAE) signal analysis.
def oaefeat(oae, fs, noise_floor=None, bands=None):
    """Otoacoustic emission analysis: emission bands, level and reproducibility.

    Rangayyan (2024) Section 1.2.16 ("Otoacoustic emission (OAE)
    signals") states that the OAE signal "represents the acoustic energy
    emitted by the cochlea either spontaneously or in response to an
    acoustic stimulus", that its existence shows the cochlea "not only
    receives sound but also produces acoustic energy", and that it "may
    also assist in screening of hearing function and in the diagnosis of
    hearing impairment".  The book gives no OAE equation, so the measures
    returned here are the standard spectral ones drawn from
    Section 6.4.1 (PSD moments, eqs. 6.32, 6.34, 6.35) and the
    band-power fraction of eq. (6.44), plus a per-band signal-to-noise
    ratio against a caller-supplied noise-floor recording.

    WHY: an OAE is tiny -- tens of decibels below the stimulus -- and
    buried in the ear-canal noise floor.  So the question a screening
    device must answer is not "how big is it" but "is there energy at
    this frequency that is not noise", which is a per-band SNR question.
    Half-octave bands are used because cochlear frequency resolution is
    roughly constant in relative, not absolute, terms.

    Parameters
    ----------
    oae : array-like
        Ear-canal recording containing the emission, in arbitrary
        amplitude units (pascals at the probe microphone in practice).
        At least 4 samples.
    fs : float
        Sampling rate in hertz (Hz); positive.  OAE work needs at least
        10 kHz.
    noise_floor : array-like or None
        A recording of the same length made with no emission present (or
        the difference of two interleaved averages), in the same units.
        When given, per-band SNR in decibels is returned.  ``None``
        suppresses the SNR fields.
    bands : sequence of (float, float) or None
        Analysis bands in hertz (Hz).  ``None`` uses the half-octave
        bands centred at 1000, 1414, 2000, 2828 and 4000 Hz, the usual
        transient-OAE screening bands, clipped to fs/2.

    Returns
    -------
    RichResult
        ``band_analysis`` -- one dict per band with ``lo_hz``, ``hi_hz``,
        ``power``, ``fraction`` (dimensionless) and, when a noise floor is
        supplied, ``snr_db``; ``dominant_freq_hz``;
        ``emission_detected`` -- True when any band has SNR >= 6 dB, the
        conventional screening criterion (only present with a noise
        floor); ``rms``; ``mean_freq_hz``, ``median_freq_hz``,
        ``total_power``; ``freq_hz`` and ``psd``.
    """
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    xs = [float(v) for v in aslist(oae)]
    freqs, psd = _bsapsd(xs, fs)
    mom = _bsapsdmom(freqs, psd)
    npsd = None
    if noise_floor is not None:
        ns = [float(v) for v in aslist(noise_floor)]
        if len(ns) != len(xs):
            raise ValueError("noise_floor must have the same length as oae")
        _, npsd = _bsapsd(ns, fs)
    if bands is None:
        centres = [1000.0, 1414.0, 2000.0, 2828.0, 4000.0]
        bands = [(c / 2.0 ** 0.25, c * 2.0 ** 0.25) for c in centres
                 if c * 2.0 ** 0.25 < fs / 2.0]
        if not bands:
            raise ValueError("fs is too low for any default OAE band; "
                             "supply bands explicitly")
    rows, detected = [], False
    for lo, hi in bands:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError("each band must have hi > lo (Hz)")
        if hi > fs / 2.0:
            raise ValueError("band upper edge exceeds the Nyquist frequency")
        p = _bsabandpow(freqs, psd, lo, hi)
        row = {"lo_hz": lo, "hi_hz": hi, "power": p,
               "fraction": p / mom["total_power"]}
        if npsd is not None:
            npow = _bsabandpow(freqs, npsd, lo, hi)
            if npow <= 0.0:
                raise ValueError("noise floor has no power in band %g-%g Hz"
                                 % (lo, hi))
            row["snr_db"] = 10.0 * log(p / npow, 10.0)
            detected = detected or row["snr_db"] >= 6.0
        rows.append(row)
    pk = _bsapeaks(freqs, psd, count=3, minsep=fs / 200.0)
    out = dict(mom)
    out.update({
        "band_analysis": rows,
        "dominant_freq_hz": pk[0][0] if pk else None,
        "peaks_hz": pk, "rms": _bsarms(xs),
        "freq_hz": freqs, "psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "snr": "dB", "fraction": "dimensionless"},
        "method": "Rangayyan (2024) Section 1.2.16 with the PSD measures of Section 6.4.1 and the band fraction of eq. (6.44) (no OAE equation given in the book)",
    })
    if npsd is not None:
        out["emission_detected"] = detected
        out["snr_criterion_db"] = 6.0
    return RichResult(payload=out)


rangayyan_oae = oaefeat  # pre-policy spelling


# -- rgpark: Parkinson's disease monitoring via multimodal signal analysis.
def pdmonitor(eeg, emg, gait, fs, tremor_band=(3.0, 7.0)):
    """Parkinson's disease monitoring from multimodal signals.

    Rangayyan (2024) Section 10.14 ("Application: Monitoring Parkinson's
    Disease Using Multimodal Signals") is the application; the book
    explains that in Parkinson's disease diseased or damaged neurons
    "produce less dopamine, resulting in mobility problems associated with
    the condition".  The EEG bands are those of Section 1.2.6 and the
    band-power fractions are eq. (6.44), Section 6.4.2; the EMG
    characterisation follows Section 5.6.4 (form factor, eq. 5.26) and
    the turns count of Section 5.6.3.  The book does NOT give a
    quantitative tremor index, so the three per-modality indices returned
    here are stated rather than cited, and are band-power fractions in the
    3-7 Hz parkinsonian tremor band -- the frequency range is the
    standard clinical one for rest tremor, not a value taken from the
    book.

    WHY: Parkinson's disease is a movement disorder, so the informative
    measurement is periodicity in the 3-7 Hz band appearing where it does
    not belong -- in resting EMG, in gait -- rather than any change in
    overall signal level.  Monitoring is multimodal because the three
    cardinal signs (tremor, rigidity, bradykinesia) show up in different
    channels: tremor in EMG, gait disturbance in the accelerometer, and
    cortical changes in the EEG beta band.

    Parameters
    ----------
    eeg : array-like
        EEG channel, in microvolts (uV).
    emg : array-like
        Surface EMG from a limb muscle, in microvolts (uV).
    gait : array-like
        Gait signal -- accelerometer or force-plate trace -- in arbitrary
        amplitude units.  All three may have different lengths; each is
        analysed on its own.
    fs : float
        Common sampling rate in hertz (Hz); must be at least 60 Hz.
    tremor_band : (float, float)
        Tremor band in hertz (Hz).  Default 3-7 Hz.

    Returns
    -------
    RichResult
        ``eeg_bands`` -- delta/theta/alpha/beta power fractions;
        ``eeg_beta_fraction`` -- flagged in the literature as elevated in
        Parkinson's disease;
        ``emg_tremor_fraction``, ``gait_tremor_fraction`` -- band-power
        fractions in ``tremor_band``, dimensionless;
        ``emg_tremor_freq_hz``, ``gait_tremor_freq_hz`` -- the dominant
        frequency inside the tremor band;
        ``emg_form_factor``, ``emg_turns_per_second``;
        ``gait_rate_hz`` -- dominant gait frequency (step rate);
        ``gait_regularity`` -- the normalised autocorrelation peak at the
        gait period, 0 to 1, low for the irregular gait of Parkinson's
        disease;
        ``tremor_present`` -- True when the EMG tremor fraction exceeds
        0.2 AND a dominant peak sits inside the tremor band.
    """
    fs = float(fs)
    if fs < 60.0:
        raise ValueError("fs must be at least 60 Hz")
    tlo, thi = float(tremor_band[0]), float(tremor_band[1])
    if not 0.0 < tlo < thi:
        raise ValueError("tremor_band must satisfy 0 < lo < hi (Hz)")
    if thi > fs / 2.0:
        raise ValueError("tremor_band upper edge exceeds the Nyquist frequency")

    def tremor(sig, name):
        xs = [float(v) for v in aslist(sig)]
        if len(xs) < 4:
            raise ValueError(name + " needs at least 4 samples")
        fr, ps = _bsapsd(xs, fs)
        tot = fsum(ps)
        if tot <= 0.0:
            raise ValueError(name + " is constant; no spectrum to analyse")
        inb = [(f, p) for f, p in zip(fr, ps) if tlo <= f <= thi]
        fpk = max(inb, key=lambda t: t[1])[0] if inb else None
        return xs, fr, ps, tot, _bsabandpow(fr, ps, tlo, thi) / tot, fpk

    ea, efr, eps, etot, _etr, _ef = tremor(eeg, "eeg")
    bands = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
             "alpha": (8.0, 13.0001), "beta": (13.0001, min(30.0, fs / 2.0))}
    eeg_bands = {k: _bsabandpow(efr, eps, lo, hi) / etot
                 for k, (lo, hi) in bands.items() if hi > lo}
    ma, _mfr, _mps, _mt, mtr, mf = tremor(emg, "emg")
    ga, gfr, gps, _gt, gtr, gf = tremor(gait, "gait")
    hj = _bsahjorth(ma)
    turns = sum(1 for i in range(1, len(ma) - 1)
                if (ma[i] - ma[i - 1]) * (ma[i + 1] - ma[i]) < 0.0)
    gpk = _bsapeaks(gfr, gps, count=1)
    grate = gpk[0][0] if gpk else None
    greg = None
    if grate and grate > 0.0:
        lag = int(round(fs / grate))
        if 1 <= lag < len(ga) - 1:
            acf = _bsaacf(ga, lag)
            if acf[0] > 0.0:
                greg = max(0.0, min(1.0, acf[lag] / acf[0]))
    return RichResult(payload={
        "eeg_bands": eeg_bands,
        "eeg_beta_fraction": eeg_bands.get("beta"),
        "emg_tremor_fraction": mtr, "emg_tremor_freq_hz": mf,
        "gait_tremor_fraction": gtr, "gait_tremor_freq_hz": gf,
        "emg_form_factor": hj["form_factor"],
        "emg_turns_per_second": turns * fs / len(ma),
        "gait_rate_hz": grate, "gait_regularity": greg,
        "tremor_present": mtr > 0.2 and mf is not None,
        "tremor_band_hz": (tlo, thi), "fs_hz": fs,
        "units": {"eeg": "uV", "emg": "uV", "frequency": "Hz",
                  "fractions": "dimensionless", "turns": "1/s"},
        "method": "Rangayyan (2024) Section 10.14 with the EEG bands of Section 1.2.6, band fractions of eq. (6.44) and the form factor of eq. (5.26); the 3-7 Hz tremor band is the standard clinical range, not a value given in the book",
    })


rangayyan_parkinson_multimodal = pdmonitor  # pre-policy spelling


# -- rgpcgeeg: PCG-EEG coupling analysis for auditory evoked response.
def pcgeeg(pcg, eeg, fs, n_segments=8, band=(1.0, 100.0)):
    """PCG-EEG coupling by the magnitude-squared coherence spectrum.

    Rangayyan (2024) eq. (4.32), Section 4.5, gives the normalised
    magnitude of the coherence spectrum of two signals as

        Gamma_xy(f) = [ |S_xy(f)|^2 / (S_xx(f) S_yy(f)) ]^(1/2),

    and warns explicitly that "if this expression is computed for two
    individual signals directly, the magnitude of the result will be equal
    to unity for all f, which is incorrect": each of S_xy, S_xx and S_yy
    must be estimated by AVERAGING over several observations.  This block
    therefore splits both records into ``n_segments`` non-overlapping
    Hann-windowed segments and averages the three spectra before forming
    the ratio.  The phase of the coherence spectrum, psi_xy(f) = angle
    S_xy(f), is also returned; the book states that it "represents the
    average phase difference (related to the time delay) between frequency
    components in the two signals".  Rangayyan cites Dobie and Wilson's
    analysis of auditory evoked potentials by magnitude-squared coherence
    as an application.

    WHY: correlation in the time domain answers "do these two signals move
    together", which for a heart sound and an EEG is almost always no.
    Coherence answers the useful question instead: is there a SPECIFIC
    FREQUENCY at which they are consistently phase-locked.  That is what
    an auditory-evoked response to one's own heart sound would look like,
    and it is invisible to a correlation coefficient.

    Parameters
    ----------
    pcg, eeg : array-like
        The two simultaneously recorded signals, same length, in their own
        arbitrary amplitude units.  Length must be at least
        8 * n_segments.
    fs : float
        Common sampling rate in hertz (Hz); positive.
    n_segments : int
        Number of segments averaged; must be at least 2, or the
        coherence is identically 1 as the book warns.  More segments give
        a lower-variance but coarser-resolution estimate.
    band : (float, float)
        Band in hertz (Hz) over which the summary statistics are taken.

    Returns
    -------
    RichResult
        ``freq_hz``, ``coherence`` (Gamma_xy, dimensionless in [0, 1]),
        ``coherence_sq`` (magnitude-squared coherence) and ``phase_rad``
        (psi_xy, radians);
        ``peak_coherence`` and ``peak_freq_hz`` within ``band``;
        ``mean_coherence`` over ``band``;
        ``delay_ms_at_peak`` -- phase divided by angular frequency at the
        peak, the group-delay-like time offset in milliseconds;
        ``significance_level`` -- the 95 percent level 1 - 0.05^(1/(L-1))
        for L independent segments, above which a coherence value is not
        explicable by chance alone.
    """
    xs = [float(v) for v in aslist(pcg)]
    ys = [float(v) for v in aslist(eeg)]
    if len(xs) != len(ys):
        raise ValueError("pcg and eeg must have the same length")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    L = int(n_segments)
    if L < 2:
        raise ValueError("n_segments must be at least 2: with one segment the "
                         "coherence is identically unity (Rangayyan eq. 4.32)")
    if len(xs) < 8 * L:
        raise ValueError("need at least 8*n_segments samples")
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 <= lo < hi:
        raise ValueError("band must satisfy 0 <= lo < hi (Hz)")
    w = len(xs) // L
    nfft = 1
    while nfft < w:
        nfft <<= 1
    m = nfft // 2 + 1
    Sxx = [0.0] * m
    Syy = [0.0] * m
    Sxyr = [0.0] * m
    Sxyi = [0.0] * m
    han = [0.5 - 0.5 * cos(2.0 * pi * i / (w - 1)) for i in range(w)]
    for s in range(L):
        a = xs[s * w:(s + 1) * w]
        b = ys[s * w:(s + 1) * w]
        ma, mb = fsum(a) / w, fsum(b) / w
        ar = [(a[i] - ma) * han[i] for i in range(w)] + [0.0] * (nfft - w)
        ai = [0.0] * nfft
        br = [(b[i] - mb) * han[i] for i in range(w)] + [0.0] * (nfft - w)
        bi = [0.0] * nfft
        _bsafft(ar, ai)
        _bsafft(br, bi)
        for k in range(m):
            Sxx[k] += ar[k] * ar[k] + ai[k] * ai[k]
            Syy[k] += br[k] * br[k] + bi[k] * bi[k]
            # X conj(Y)
            Sxyr[k] += ar[k] * br[k] + ai[k] * bi[k]
            Sxyi[k] += ai[k] * br[k] - ar[k] * bi[k]
    freqs = [k * fs / nfft for k in range(m)]
    coh, coh2, ph = [], [], []
    for k in range(m):
        den = Sxx[k] * Syy[k]
        c2 = ((Sxyr[k] ** 2 + Sxyi[k] ** 2) / den) if den > 0.0 else 0.0
        c2 = min(1.0, max(0.0, c2))
        coh2.append(c2)
        coh.append(sqrt(c2))
        ph.append(atan2(Sxyi[k], Sxyr[k]))
    inb = [k for k in range(m) if lo <= freqs[k] <= hi]
    if not inb:
        raise ValueError("no spectral bins in %g-%g Hz" % (lo, hi))
    kpk = max(inb, key=lambda k: coh[k])
    delay = None
    if freqs[kpk] > 0.0:
        delay = 1000.0 * ph[kpk] / (2.0 * pi * freqs[kpk])
    return RichResult(payload={
        "freq_hz": freqs, "coherence": coh, "coherence_sq": coh2,
        "phase_rad": ph,
        "peak_coherence": coh[kpk], "peak_freq_hz": freqs[kpk],
        "mean_coherence": fsum(coh[k] for k in inb) / len(inb),
        "delay_ms_at_peak": delay,
        "significance_level": 1.0 - 0.05 ** (1.0 / (L - 1)),
        "n_segments": L, "segment_samples": w, "band_hz": (lo, hi),
        "fs_hz": fs,
        "units": {"frequency": "Hz", "coherence": "dimensionless [0,1]",
                  "phase": "radians", "delay": "ms"},
        "method": "Rangayyan (2024) eq. (4.32), Section 4.5, magnitude coherence with segment averaging as the book requires",
    })


rangayyan_pcg_eeg_coupling = pcgeeg  # pre-policy spelling


# -- rgpcgmrm: Murmur presence detection in PCG via spectral analysis.
def murmdet(pcg, fs, threshold=0.15, hf_band=(150.0, 600.0)):
    """Detect the presence of a murmur in a PCG from its spectral content.

    Rangayyan (2024) Section 10.2.4 poses the question "Is a murmur
    present?" as a pattern-classification problem, and Section 6.4.2
    supplies the measure: "in the case of PCG analysis for the detection
    of murmurs, we could specifically investigate the presence of signal
    power in the frequency range beyond that of S1 and/or S2", the
    fraction of signal power in a band being eq. (6.44),

        E_{f1:f2} = (2 / (N E_x)) sum_{k=k1}^{k2} |X(k)|^2.

    WHY: S1 and S2 are transient, largely low-frequency events -- valve
    closure sounds.  A murmur is turbulent flow, and turbulence is
    wideband, so its distinguishing feature is not amplitude (a loud
    normal heart sound is still normal) but the presence of energy ABOVE
    the band that S1 and S2 occupy.  A power FRACTION rather than an
    absolute power is used so that the decision does not depend on the
    recording gain.

    Parameters
    ----------
    pcg : array-like
        PCG segment, in arbitrary amplitude units; at least 4 samples.
        For best discrimination use a systolic or diastolic segment
        excluding S1 and S2 themselves.
    fs : float
        Sampling rate in hertz (Hz); positive.
    threshold : float
        Decision threshold on the high-band power fraction,
        dimensionless, in (0, 1).  The default 0.15 is a starting point
        only: the book treats this as a trained classifier decision, so
        the threshold must be set on labelled data for the recording
        setup in use.
    hf_band : (float, float)
        Band in hertz (Hz) taken to lie beyond S1 and S2.  Upper edge
        must not exceed fs/2.

    Returns
    -------
    RichResult
        ``murmur_present`` -- the boolean decision;
        ``hf_power_fraction`` -- eq. (6.44) over hf_band, dimensionless;
        ``threshold``; ``margin`` = fraction - threshold;
        ``mean_freq_hz``, ``median_freq_hz``, ``spread_hz``,
        ``spectral_skewness``, ``spectral_kurtosis``;
        ``dominant_freq_hz``; ``freq_hz`` and ``psd``.
    """
    fs = float(fs)
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be a power fraction in (0, 1)")
    lo, hi = float(hf_band[0]), float(hf_band[1])
    if hi <= lo:
        raise ValueError("hf_band must have hi > lo (Hz)")
    if hi > fs / 2.0:
        raise ValueError("hf_band upper edge exceeds the Nyquist frequency")
    freqs, psd = _bsapsd(pcg, fs)
    mom = _bsapsdmom(freqs, psd)
    frac = _bsabandpow(freqs, psd, lo, hi) / mom["total_power"]
    pk = _bsapeaks(freqs, psd, count=3)
    out = dict(mom)
    out.update({
        "murmur_present": frac >= threshold,
        "hf_power_fraction": frac,
        "threshold": threshold, "margin": frac - threshold,
        "hf_band_hz": (lo, hi),
        "dominant_freq_hz": pk[0][0] if pk else None,
        "freq_hz": freqs, "psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "fraction": "dimensionless"},
        "method": "Rangayyan (2024) Section 10.2.4 with the band power fraction of eq. (6.44), Section 6.4.2",
    })
    return RichResult(payload=out)


rangayyan_pcg_murmur_detect = murmdet  # pre-policy spelling


# -- rgpolysg: Polysomnography signal fusion for sleep staging.
def psgstage(eeg, eog, emg, fs, epoch_len=30.0):
    """Polysomnographic signal fusion for sleep staging.

    Rangayyan (2024) Section 2.4.1 ("Monitoring of sleep apnea by
    polysomnography") states that polysomnography, "which involves
    multichannel recording of several biomedical signals and parameters,
    is the current standard for the evaluation of sleep-related problems".
    The EEG rhythm bands used here are those defined in Section 1.2.6:
    delta 0.5 <= f < 4 Hz, theta 4 <= f < 8 Hz, alpha 8 <= f <= 13 Hz,
    beta f > 13 Hz; band-power fractions are eq. (6.44), Section 6.4.2.
    The book does NOT give scoring rules for sleep stages, so the stage
    assignment below is the conventional band-dominance rule and is
    labelled as a heuristic in the payload -- it is not a clinical scoring
    algorithm.

    WHY: no single channel identifies a sleep stage.  EEG slowing alone
    cannot separate REM sleep from wakefulness, because REM EEG is
    fast and wake-like; what distinguishes REM is fast EEG TOGETHER WITH
    eye movements and TOGETHER WITH the loss of chin muscle tone.  That is
    precisely why the recording is polysomnographic -- the discriminating
    information is in the combination of channels, not in any one of them.

    Parameters
    ----------
    eeg, eog, emg : array-like
        Central EEG, electro-oculogram and chin EMG, all the same length
        and sampling rate, in microvolts (uV).
    fs : float
        Common sampling rate in hertz (Hz); must be at least 60 Hz so
        that the beta band and the EMG band are resolved.
    epoch_len : float
        Scoring epoch in seconds (s); positive.  30 s is the standard.

    Returns
    -------
    RichResult
        ``epochs`` -- one dict per epoch with ``t_start_s``, the four EEG
        band fractions, ``eeg_slow_fraction`` (delta+theta),
        ``eog_activity`` (RMS of the EOG in uV, band-limited to
        0.3-8 Hz where eye movements live), ``emg_tone`` (RMS of the EMG
        above 10 Hz, in uV), and ``stage``;
        ``stage_sequence``; ``stage_minutes`` -- minutes spent in each
        stage; ``total_sleep_time_min``; ``sleep_efficiency`` -- sleep
        time over recording time; ``n_epochs``; ``heuristic`` (always
        True).
    """
    a = [float(v) for v in aslist(eeg)]
    b = [float(v) for v in aslist(eog)]
    c = [float(v) for v in aslist(emg)]
    if not len(a) == len(b) == len(c):
        raise ValueError("eeg, eog and emg must have the same length")
    fs = float(fs)
    if fs < 60.0:
        raise ValueError("fs must be at least 60 Hz for polysomnographic bands")
    epoch_len = float(epoch_len)
    if epoch_len <= 0.0:
        raise ValueError("epoch_len must be positive (s)")
    w = int(round(epoch_len * fs))
    n_ep = len(a) // w
    if n_ep < 1:
        raise ValueError("recording is shorter than one epoch")
    rows = []
    for e in range(n_ep):
        sl = slice(e * w, (e + 1) * w)
        fr, ps = _bsapsd(a[sl], fs)
        tot = fsum(ps)
        if tot <= 0.0:
            raise ValueError("EEG epoch %d is constant" % e)
        d = _bsabandpow(fr, ps, 0.5, 4.0) / tot
        th = _bsabandpow(fr, ps, 4.0, 8.0) / tot
        al = _bsabandpow(fr, ps, 8.0, 13.0001) / tot
        be = _bsabandpow(fr, ps, 13.0001, fs / 2.0) / tot
        fro, pso = _bsapsd(b[sl], fs)
        eog_act = sqrt(_bsabandpow(fro, pso, 0.3, 8.0) / len(b[sl]))
        frm, psm = _bsapsd(c[sl], fs)
        emg_tone = sqrt(_bsabandpow(frm, psm, 10.0, fs / 2.0) / len(c[sl]))
        rows.append({"t_start_s": e * epoch_len, "delta_fraction": d,
                     "theta_fraction": th, "alpha_fraction": al,
                     "beta_fraction": be, "eeg_slow_fraction": d + th,
                     "eog_activity": eog_act, "emg_tone": emg_tone})
    # Heuristic staging: reference levels are the medians of this recording,
    # so the rule is self-calibrating rather than absolute.
    def med(key):
        v = sorted(r[key] for r in rows)
        m = len(v)
        return v[m // 2] if m % 2 else 0.5 * (v[m // 2 - 1] + v[m // 2])

    emg_ref = med("emg_tone")
    eog_ref = med("eog_activity")
    for r in rows:
        if r["delta_fraction"] > 0.5:
            r["stage"] = "N3"
        elif r["emg_tone"] < 0.5 * emg_ref and r["eog_activity"] > eog_ref \
                and r["beta_fraction"] + r["theta_fraction"] > 0.4:
            r["stage"] = "REM"
        elif r["alpha_fraction"] > 0.3 and r["emg_tone"] >= emg_ref:
            r["stage"] = "Wake"
        elif r["theta_fraction"] > r["alpha_fraction"]:
            r["stage"] = "N2"
        else:
            r["stage"] = "N1"
    seq = [r["stage"] for r in rows]
    mins = {}
    for s in seq:
        mins[s] = mins.get(s, 0.0) + epoch_len / 60.0
    tst = sum(v for k, v in mins.items() if k != "Wake")
    total = n_ep * epoch_len / 60.0
    return RichResult(payload={
        "epochs": rows, "stage_sequence": seq, "stage_minutes": mins,
        "total_sleep_time_min": tst, "recording_time_min": total,
        "sleep_efficiency": tst / total if total > 0.0 else 0.0,
        "n_epochs": n_ep, "epoch_len_s": epoch_len, "fs_hz": fs,
        "heuristic": True,
        "units": {"signals": "uV", "time": "s", "stage_minutes": "minutes",
                  "fractions": "dimensionless"},
        "method": "Rangayyan (2024) Section 2.4.1 (polysomnography) with the EEG bands of Section 1.2.6 and band fractions of eq. (6.44); staging rule is a self-calibrating heuristic, not a clinical scoring algorithm and not given in the book",
    })


rangayyan_polysomnography = psgstage  # pre-policy spelling


# -- rgppt: Point process model for inter-event interval (IEI) statistics.
def ieistats(event_times, T=None, n_bins=20):
    """Inter-event interval statistics of a physiological point process.

    Rangayyan (2024) Section 7.3 ("Point Processes") characterises a
    point process by its inter-pulse-interval (IPI) statistics: the mean
    repetition rate mu_r in pulses per second (pps) and the coefficient
    of variation

        CV_r = sigma_r / mu_r,

    where sigma_r is the standard deviation of the repetition rate.  The
    book reports, for the patellofemoral pulse trains of Zhang et al.,
    mu_r = 25.2 pps with CV_r = 0.07 for one normal subject and
    mu_r = 16.1 pps with CV_r = 0.25 for another, and simulations at
    mu_r = 21 pps with CV_r = 0.1 and 0.05.  It states the consequences
    directly: the PSD of the pulse train shows "the most-dominant peak at
    the mean repetition rate of the point process, followed by smaller
    peaks at its harmonics", the higher harmonics being better defined the
    smaller CV_r; in the limit CV_r = 0 the PSD is a periodic impulse
    train of equal-strength impulses.  The same model applies to SMUAP
    trains and to voiced speech (Section 7.2.3).

    NOTE ON UNITS: the book reports the mean repetition RATE mu_r in pps
    and its coefficient of variation.  Because the rate is the reciprocal
    of the interval, CV computed on intervals and CV computed on rates are
    NOT equal in general; both are returned, labelled, so that a
    comparison with the book's figures uses the right one.

    WHY: a point process carries its information in WHEN events occur,
    not in their amplitude, and CV_r is the single number that says how
    regular that timing is.  A CV near zero means a near-periodic
    generator with sharp spectral harmonics; a large CV means the
    repetition structure is smeared out and, as the book notes, the effect
    of the repetition is limited to low frequencies.  Everything about the
    spectrum of the resulting signal follows from these two numbers.

    Parameters
    ----------
    event_times : array-like
        Times of the events, in seconds (s), strictly increasing.  At
        least 3 events, so that at least 2 intervals exist.
    T : float or None
        Total observation duration in seconds (s), used for the overall
        event rate.  ``None`` uses the span from the first to the last
        event.
    n_bins : int
        Number of bins of the returned IPI histogram; >= 2.  The book
        shows such histograms in its Figure 7.9.

    Returns
    -------
    RichResult
        ``mean_ipi_s``, ``sd_ipi_s``, ``cv_ipi`` -- interval statistics;
        ``mean_rate_pps`` (mu_r), ``sd_rate_pps`` (sigma_r) and
        ``cv_rate`` (CV_r as the book defines it);
        ``event_rate_pps`` -- n_events / T, the overall rate;
        ``min_ipi_s``, ``max_ipi_s``, ``median_ipi_s``;
        ``fano_factor`` -- variance of counts over mean of counts,
        1 for a Poisson process;
        ``ipi_histogram`` -- list of (bin_centre_s, count);
        ``regularity`` -- "near-periodic" for CV_r < 0.1, "regular" below
        0.25, "irregular" otherwise, using the book's own reported
        values as the anchors.
    """
    ts = [float(v) for v in aslist(event_times)]
    if len(ts) < 3:
        raise ValueError("need at least 3 event times to form 2 intervals")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("event_times must be strictly increasing (s)")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    ipi = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    n = len(ipi)
    mi = fsum(ipi) / n
    si = sqrt(fsum((v - mi) ** 2 for v in ipi) / (n - 1)) if n > 1 else 0.0
    rates = [1.0 / v for v in ipi]
    mr = fsum(rates) / n
    sr = sqrt(fsum((v - mr) ** 2 for v in rates) / (n - 1)) if n > 1 else 0.0
    span = (ts[-1] - ts[0]) if T is None else float(T)
    if span <= 0.0:
        raise ValueError("observation duration must be positive (s)")
    srt = sorted(ipi)
    med = (srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2]))
    lo, hi = min(ipi), max(ipi)
    hist = []
    if hi > lo:
        width = (hi - lo) / n_bins
        counts = [0] * n_bins
        for v in ipi:
            counts[min(int((v - lo) / width), n_bins - 1)] += 1
        hist = [(lo + (b + 0.5) * width, counts[b]) for b in range(n_bins)]
    else:
        hist = [(lo, n)]
    # Fano factor over windows of ten mean intervals
    win = 10.0 * mi
    nw = max(2, int(span / win))
    counts = [0] * nw
    for t in ts:
        b = int((t - ts[0]) / span * nw)
        counts[min(max(b, 0), nw - 1)] += 1
    mc = fsum(counts) / nw
    fano = (fsum((c - mc) ** 2 for c in counts) / nw / mc) if mc > 0.0 else None
    cvr = sr / mr if mr > 0.0 else 0.0
    reg = ("near-periodic" if cvr < 0.1
           else ("regular" if cvr < 0.25 else "irregular"))
    return RichResult(payload={
        "mean_ipi_s": mi, "sd_ipi_s": si, "cv_ipi": si / mi if mi > 0 else 0.0,
        "mean_rate_pps": mr, "sd_rate_pps": sr, "cv_rate": cvr,
        "event_rate_pps": len(ts) / span,
        "min_ipi_s": lo, "max_ipi_s": hi, "median_ipi_s": med,
        "fano_factor": fano,
        "ipi_histogram": hist, "n_events": len(ts), "n_intervals": n,
        "duration_s": span, "regularity": reg,
        "units": {"interval": "s", "rate": "pps (pulses per second)",
                  "cv": "dimensionless"},
        "method": "Rangayyan (2024) Section 7.3, IPI statistics mu_r and CV_r = sigma_r / mu_r, after Zhang et al.",
    })


rangayyan_point_process = ieistats  # pre-policy spelling


# -- rgpros: Prosthetic heart valve evaluation via PCG spectral analysis.
def valvepcg(pcg, fs, n_peaks=3, order=None):
    """Prosthetic heart valve evaluation from the PCG spectrum.

    Rangayyan (2024) Section 6.5 ("Application: Evaluation of Prosthetic
    Heart Valves") reports the work of Durand et al., who characterised
    the PSDs of prosthetic-valve sounds by treating each resonance peak
    as a bandpass filter and computing, per Section 6.4.2, "the -3 dB
    bandwidth of the peak and, furthermore, ... its quality factor as the
    ratio of the peak frequency to the bandwidth", "for not only the
    dominant peak but also several peaks at progressively lower levels of
    signal power".  That is exactly what this block returns, computed on
    an autoregressive spectrum (Section 7.5) so that the peaks are
    resolved rather than smeared by the periodogram variance.

    WHY: a mechanical valve is a rigid structure that rings when it
    closes, so its sound has sharp resonances at frequencies fixed by its
    geometry and mass.  Degeneration -- thrombus, pannus, a cracked
    occluder -- changes that mechanical resonator, and the change shows up
    as a shifted peak frequency or a lowered Q (more damping) long before
    it produces a clinical sign.  Tracking (f_peak, Q) over follow-up
    visits is the measurement; the absolute values alone are not
    diagnostic.

    Parameters
    ----------
    pcg : array-like
        PCG segment containing the valve closure sound, in arbitrary
        amplitude units.  At least 32 samples.
    fs : float
        Sampling rate in hertz (Hz); positive.  Prosthetic-valve sounds
        extend well above 1 kHz, so fs should be at least 4 kHz.
    n_peaks : int
        Number of resonance peaks to characterise, strongest first; >= 1.
    order : int or None
        AR model order.  ``None`` uses 2 * n_peaks + 4, enough poles for
        the requested resonances plus a spectral floor.

    Returns
    -------
    RichResult
        ``peaks`` -- a list of dicts, one per resonance, with
        ``freq_hz``, ``power``, ``bandwidth_3db_hz`` and ``q_factor``
        (dimensionless);
        ``dominant_freq_hz``, ``dominant_q``;
        ``mean_freq_hz``, ``median_freq_hz``, ``spread_hz`` of the
        spectrum; ``freq_hz`` and ``ar_psd``.
    """
    xs = [float(v) for v in aslist(pcg)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    if len(xs) < 32:
        raise ValueError("need at least 32 PCG samples")
    n_peaks = int(n_peaks)
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")
    p = int(order) if order is not None else 2 * n_peaks + 4
    if p < 2:
        raise ValueError("order must be at least 2")
    if len(xs) < 4 * p:
        raise ValueError("need at least 4*order samples to fit the AR model")
    a, err = _bsalpc(xs, p)
    freqs, psd = _bsalpcspec(a, fs, npts=2048)
    psd = [err * v for v in psd]
    found = _bsapeaks(freqs, psd, count=n_peaks, minsep=fs / 200.0)
    peaks = []
    for f, pw in found:
        bw, q = _bsaqfactor(freqs, psd, f)
        peaks.append({"freq_hz": f, "power": pw,
                      "bandwidth_3db_hz": bw, "q_factor": q})
    mom = _bsapsdmom(freqs, psd)
    out = dict(mom)
    out.update({
        "peaks": peaks,
        "dominant_freq_hz": peaks[0]["freq_hz"] if peaks else None,
        "dominant_q": peaks[0]["q_factor"] if peaks else None,
        "order": p, "prediction_error": err, "ar_coeffs": a,
        "freq_hz": freqs, "ar_psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "bandwidth": "Hz",
                  "q_factor": "dimensionless"},
        "method": "Rangayyan (2024) Section 6.5 after Durand et al., with the -3 dB bandwidth and quality-factor measures of Section 6.4.2",
    })
    return RichResult(payload=out)


rangayyan_prosthetic_valve = valvepcg  # pre-policy spelling


# -- rgrespf: Respiratory signal analysis: rate, depth, I:E ratio.
def respfeat(resp, fs, signal_type="flow", min_breath_s=1.0):
    """Respiratory rate, depth and inspiratory-to-expiratory ratio.

    Rangayyan (2024) Section 2.4.1 ("Monitoring of sleep apnea by
    polysomnography") lists respiratory effort and airflow among the
    channels recorded, and Section 5.10 ("Application: Analysis of
    Respiration") relates EMG activity of the parasternal intercostal and
    crural diaphragm muscles to airflow via measures such as the
    zero-crossing rate of the EMG.  The book gives no equations for
    respiratory rate, tidal depth or the I:E ratio; the definitions used
    here are the standard respiratory-mechanics ones and are stated
    rather than cited:

        rate = 60 / mean breath period            breaths per minute
        depth = mean peak-to-trough excursion of the volume signal, or of
                the integrated flow signal
        I:E = mean inspiratory duration / mean expiratory duration,

    inspiration being the positive-flow (or rising-volume) phase.

    WHY: a respiratory signal is analysed cycle by cycle, not in
    aggregate, because the clinically abnormal patterns are changes in the
    RELATIONSHIP between the phases -- a prolonged expiratory phase (a low
    I:E ratio) in obstructive disease, an irregular period in
    Cheyne-Stokes breathing -- and averaging over the whole record hides
    exactly those.  Per-breath values are therefore returned alongside the
    summaries.

    Parameters
    ----------
    resp : array-like
        Respiratory signal, in arbitrary amplitude units (litres per
        second for flow, litres for volume).
    fs : float
        Sampling rate in hertz (Hz); positive.
    signal_type : str
        ``"flow"`` -- inspiration is the positive part of the signal, and
        depth is obtained by integrating flow over each inspiration;
        ``"volume"`` -- inspiration is the rising part, and depth is the
        trough-to-peak excursion.
    min_breath_s : float
        Shortest accepted breath period in seconds (s); shorter zero
        crossings or turning points are treated as noise and merged.
        Positive.

    Returns
    -------
    RichResult
        ``rate_breaths_per_min``; ``mean_period_s`` and ``sd_period_s``;
        ``depth`` -- mean tidal excursion in the input amplitude units
        (litres when flow is in L/s or volume in L);
        ``ie_ratio`` -- mean inspiratory over mean expiratory duration
        (dimensionless); ``mean_ti_s``, ``mean_te_s``;
        ``breaths`` -- one dict per detected breath with ``t_start_s``,
        ``period_s``, ``ti_s``, ``te_s``, ``depth``;
        ``n_breaths``; ``regularity_cv`` -- SD over mean of the period.
    """
    xs = [float(v) for v in aslist(resp)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    if len(xs) < 8:
        raise ValueError("need at least 8 samples")
    signal_type = str(signal_type).lower()
    if signal_type not in ("flow", "volume"):
        raise ValueError("signal_type must be 'flow' or 'volume'")
    min_breath_s = float(min_breath_s)
    if min_breath_s <= 0.0:
        raise ValueError("min_breath_s must be positive (s)")
    mu = fsum(xs) / len(xs)
    ys = [v - mu for v in xs]
    if signal_type == "volume":
        drive = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)] + [0.0]
    else:
        drive = ys
    # inspiration onsets = upward zero crossings of the drive signal
    ons = [i for i in range(1, len(drive))
           if drive[i - 1] <= 0.0 < drive[i]]
    kept = []
    for i in ons:
        if not kept or (i - kept[-1]) / fs >= min_breath_s:
            kept.append(i)
    if len(kept) < 2:
        raise ValueError("fewer than 2 breaths detected; check signal_type, "
                         "fs and min_breath_s")
    breaths = []
    for b in range(len(kept) - 1):
        i0, i1 = kept[b], kept[b + 1]
        # end of inspiration = downward zero crossing of the drive signal
        iend = i1
        for i in range(i0 + 1, i1):
            if drive[i - 1] > 0.0 >= drive[i]:
                iend = i
                break
        ti = (iend - i0) / fs
        te = (i1 - iend) / fs
        if signal_type == "flow":
            depth = fsum(ys[i0:iend]) / fs
        else:
            seg = ys[i0:i1]
            depth = (max(seg) - min(seg)) if seg else 0.0
        breaths.append({"t_start_s": i0 / fs, "period_s": (i1 - i0) / fs,
                        "ti_s": ti, "te_s": te, "depth": depth})
    per = [b["period_s"] for b in breaths]
    n = len(per)
    mp = fsum(per) / n
    sp = sqrt(fsum((v - mp) ** 2 for v in per) / (n - 1)) if n > 1 else 0.0
    mti = fsum(b["ti_s"] for b in breaths) / n
    mte = fsum(b["te_s"] for b in breaths) / n
    return RichResult(payload={
        "rate_breaths_per_min": 60.0 / mp,
        "mean_period_s": mp, "sd_period_s": sp,
        "regularity_cv": sp / mp if mp > 0.0 else 0.0,
        "depth": fsum(b["depth"] for b in breaths) / n,
        "mean_ti_s": mti, "mean_te_s": mte,
        "ie_ratio": mti / mte if mte > 0.0 else None,
        "breaths": breaths, "n_breaths": n,
        "signal_type": signal_type, "fs_hz": fs,
        "units": {"rate": "breaths/min", "time": "s",
                  "depth": "litres if flow is L/s or volume is L",
                  "ie_ratio": "dimensionless"},
        "method": "Standard per-breath respiratory measures; Rangayyan (2024) Sections 2.4.1 and 5.10 give the context but no equations for rate, depth or I:E",
    })


rangayyan_respiration_features = respfeat  # pre-policy spelling


# -- rgrespsnd: Respiratory sound generation model (bronchial turbulence).
def respsound(length_m=0.1, radius_m=0.009, freqs=None, rho=1.2, c=343.0,
              mu=1.8e-5, P0=101325.0, eta=1.4, lam=0.026, cp=1005.0):
    """Acoustic tube-segment model of respiratory sound transmission.

    Rangayyan (2024) Section 7.7.1 ("Modeling of respiratory sounds")
    derives the electrical analogue of a segment of the airway from the
    acoustics of a rigid pipe.  Equating pressure with voltage and volume
    velocity with current gives, from the book:

        acoustic inductance (inertance)   L_a = rho l / A       eq. (7.122)
        acoustic capacitance (compliance) C_a = V_a / (P eta)   eq. (7.127)
        acoustic resistance               R_a = (l S / A^2) sqrt(omega rho mu / 2)
                                                                eq. (7.128)
        acoustic conductance              G_a = (S l / (rho c^2)) (eta - 1)
                                                * sqrt(lambda omega / (2 c_p rho))
                                                                eq. (7.129)

    where l is segment length, A its cross-sectional area, S its
    circumference, V_a the enclosed gas volume, P the pressure, eta the
    adiabatic constant, mu the viscosity coefficient, lambda the
    coefficient of heat conduction and c_p the specific heat of air at
    constant pressure.  The book then says these segments "may be used to
    model segments of tubes or pipes that can be combined to form more
    elaborate models of parts of the respiratory system", the effect of a
    segment being the filtering of an input signal as it passes through.
    That series-L_a-R_a, shunt-C_a-G_a two-port is what is evaluated here.

    NOTE ON THE SOURCE TEXT: the flattened text of eq. (7.129) shows the
    numerator as "eta - l"; it is eta - 1, since the term must vanish for
    an isothermal gas (eta = 1) and the character is a digit misread as
    the letter ell, exactly as elsewhere in the same equation's "S l".

    WHY: normal breath sounds are turbulence generated in the larger
    airways and then FILTERED by the airway tree and the chest wall on the
    way to the stethoscope.  Disease changes the filter -- narrowing,
    stiffening, loss of smoothness -- so the resonance of the segment
    model moves and its damping changes.  Modelling the transmission path
    separately from the source is what lets a change in the recorded sound
    be attributed to one or the other.

    Parameters
    ----------
    length_m : float
        Segment length l in metres (m); positive.
    radius_m : float
        Segment radius in metres (m); positive.  A = pi r^2 and
        S = 2 pi r follow.  9 mm is roughly an adult trachea.
    freqs : array-like or None
        Frequencies in hertz (Hz).  ``None`` gives 10 Hz to 2000 Hz in
        10 Hz steps, the band of interest for respiratory sounds.
    rho : float
        Gas density in kilograms per cubic metre (kg/m^3); air 1.2.
    c : float
        Speed of sound in metres per second (m/s); air 343.
    mu : float
        Viscosity coefficient in pascal seconds (Pa s); air 1.8e-5.
    P0 : float
        Static pressure in pascals (Pa); one atmosphere 101325.
    eta : float
        Adiabatic constant, dimensionless; air 1.4.  Must exceed 1.
    lam : float
        Coefficient of heat conduction in watts per metre per kelvin
        (W/(m K)); air 0.026.
    cp : float
        Specific heat at constant pressure in joules per kilogram per
        kelvin (J/(kg K)); air 1005.

    Returns
    -------
    RichResult
        ``freq_hz``; ``transfer_mag`` and ``transfer_db`` -- the pressure
        transfer magnitude of the segment; ``La_kg_per_m4``,
        ``Ca_m3_per_Pa``, ``Ra_Pa_s_per_m3`` and ``Ga_m3_per_Pa_s``
        (the last two are frequency dependent and returned as series);
        ``resonance_hz`` = 1 / (2 pi sqrt(L_a C_a));
        ``area_m2``, ``circumference_m``, ``volume_m3``.
    """
    l = float(length_m)
    r = float(radius_m)
    if l <= 0.0:
        raise ValueError("length_m must be positive (m)")
    if r <= 0.0:
        raise ValueError("radius_m must be positive (m)")
    rho, c, mu = float(rho), float(c), float(mu)
    P0, eta, lam, cp = float(P0), float(eta), float(lam), float(cp)
    if min(rho, c, mu, P0, lam, cp) <= 0.0:
        raise ValueError("rho, c, mu, P0, lam and cp must all be positive")
    if eta <= 1.0:
        raise ValueError("eta (adiabatic constant) must exceed 1")
    A = pi * r * r
    S = 2.0 * pi * r
    Va = A * l
    La = rho * l / A                       # eq. (7.122), kg/m^4
    Ca = Va / (P0 * eta)                   # eq. (7.127), m^3/Pa
    if freqs is None:
        fs_hz = [10.0 * k for k in range(1, 201)]
    else:
        fs_hz = [float(v) for v in aslist(freqs)]
        if any(v <= 0.0 for v in fs_hz):
            raise ValueError("frequencies must be positive (Hz)")
    Ra, Ga, mag = [], [], []
    for f in fs_hz:
        w = 2.0 * pi * f
        ra = (l * S / (A * A)) * sqrt(w * rho * mu / 2.0)        # eq. (7.128)
        ga = (S * l / (rho * c * c)) * (eta - 1.0) \
            * sqrt(lam * w / (2.0 * cp * rho))                   # eq. (7.129)
        Ra.append(ra)
        Ga.append(ga)
        # series impedance Ra + j w La, shunt admittance Ga + j w Ca
        zr, zi = ra, w * La
        yr, yi = ga, w * Ca
        # H = 1 / (1 + Z Y)
        dr = 1.0 + (zr * yr - zi * yi)
        di = zr * yi + zi * yr
        mag.append(1.0 / hypot(dr, di))
    f0 = 1.0 / (2.0 * pi * sqrt(La * Ca))
    return RichResult(payload={
        "freq_hz": fs_hz,
        "transfer_mag": mag,
        "transfer_db": [20.0 * log(v, 10.0) if v > 0 else -300.0 for v in mag],
        "La_kg_per_m4": La, "Ca_m3_per_Pa": Ca,
        "Ra_Pa_s_per_m3": Ra, "Ga_m3_per_Pa_s": Ga,
        "resonance_hz": f0,
        "area_m2": A, "circumference_m": S, "volume_m3": Va,
        "units": {"freq": "Hz", "La": "kg/m^4", "Ca": "m^3/Pa",
                  "Ra": "Pa s/m^3", "Ga": "m^3/(Pa s)", "length": "m"},
        "method": "Rangayyan (2024) eqs. (7.122), (7.127), (7.128), (7.129), Section 7.7.1, after Flanagan and Moussavi",
    })


rangayyan_respiratory_sound = respsound  # pre-policy spelling


# -- rgsapdet: Sleep apnea detection using multimodal biomedical signals.
def apneadet(ecg, spo2, snore, fs, epoch_s=60.0, desat_pct=4.0):
    """Sleep apnea detection from multimodal biomedical signals.

    Rangayyan (2024) Section 10.2.5 ("Detection of sleep apnea using
    multimodal biomedical signals") and Section 10.13 ("Application:
    Detection of Sleep Apnea") pose this as a multimodal pattern
    classification problem; Section 2.4 ("Application: Diagnosis and
    Monitoring of Sleep Apnea"), with Sections 2.4.1 (polysomnography),
    2.4.2 (home monitoring) and 2.4.3 (multivariate and multiorgan
    analysis), gives the clinical context.  The book reports that "one of
    the features giving good results in the detection of OSA was the
    coherence" between the signals analysed, coherence being eq. (4.32).
    HRV analysis follows Section 7.9.

    The three evidence streams combined here are:
      * OXYGEN DESATURATION -- a drop of ``desat_pct`` percentage points
        or more below the local baseline SpO2, the standard event marker;
      * SNORE ENERGY -- the RMS of the snore channel, whose cessation
        followed by a loud resumption marks an obstructive event;
      * CARDIAC MODULATION -- the low-frequency (0.01-0.04 Hz) power
        fraction of the instantaneous heart rate derived from the ECG,
        which is where cyclical apnea-related variation appears.

    The book gives published accuracies but no coefficient vector, so the
    combination rule here is an explicitly documented heuristic score, not
    a trained classifier; ``epoch_flagged`` is the transparent per-epoch
    evidence and should be the input to a classifier trained on labelled
    data.

    WHY: an apnea is not visible in any one channel with confidence.  A
    desaturation can come from a bad probe, snoring can stop because the
    subject rolled over, heart-rate variation has a dozen causes.  It is
    the CO-OCCURRENCE of the three within the same epoch that identifies
    an event, which is why the recording is multimodal in the first place.

    Parameters
    ----------
    ecg : array-like
        Single-lead ECG, in millivolts (mV).
    spo2 : array-like
        Pulse oximetry saturation, in percent (0-100), same length and
        sampling rate as ecg.
    snore : array-like
        Snore microphone channel, in arbitrary amplitude units, same
        length.
    fs : float
        Common sampling rate in hertz (Hz); must be at least 100 Hz for
        usable R-wave timing.
    epoch_s : float
        Analysis epoch in seconds (s); positive.  60 s is conventional
        for apnea scoring per-minute.
    desat_pct : float
        Desaturation depth in percentage points that marks an event;
        positive.  4 is the usual criterion.

    Returns
    -------
    RichResult
        ``epochs`` -- per-epoch dicts with ``t_start_s``, ``min_spo2``,
        ``desat_depth_pct``, ``snore_rms``, ``mean_hr_bpm``,
        ``hr_lf_fraction``, ``score`` (0-3, the number of criteria met)
        and ``epoch_flagged`` (score >= 2);
        ``n_flagged``, ``apnea_suspected``;
        ``events_per_hour`` -- flagged epochs per hour, the analogue of
        the apnea-hypopnea index;
        ``severity`` -- "none" (<5/h), "mild" (5-15), "moderate"
        (15-30), "severe" (>30), the conventional AHI bands;
        ``heuristic`` (always True).
    """
    e = [float(v) for v in aslist(ecg)]
    s = [float(v) for v in aslist(spo2)]
    q = [float(v) for v in aslist(snore)]
    if not len(e) == len(s) == len(q):
        raise ValueError("ecg, spo2 and snore must have the same length")
    fs = float(fs)
    if fs < 100.0:
        raise ValueError("fs must be at least 100 Hz for R-wave timing")
    if any(v < 0.0 or v > 100.0 for v in s):
        raise ValueError("spo2 values must lie in 0-100 percent")
    epoch_s = float(epoch_s)
    if epoch_s <= 0.0:
        raise ValueError("epoch_s must be positive (s)")
    desat_pct = float(desat_pct)
    if desat_pct <= 0.0:
        raise ValueError("desat_pct must be positive (percentage points)")
    w = int(round(epoch_s * fs))
    n_ep = len(e) // w
    if n_ep < 1:
        raise ValueError("recording is shorter than one epoch")
    # R peaks: threshold the squared first difference at 60 percent of its
    # epoch maximum, with a 200 ms refractory period
    d = [e[i + 1] - e[i] for i in range(len(e) - 1)] + [0.0]
    sq = [v * v for v in d]
    refr = int(0.2 * fs)
    rows, snores = [], []
    for k in range(n_ep):
        sl = slice(k * w, (k + 1) * w)
        seg = sq[sl]
        thr = 0.6 * max(seg) if max(seg) > 0.0 else 1.0
        peaks, last = [], -refr
        for i in range(len(seg)):
            if seg[i] >= thr and i - last >= refr:
                peaks.append(i)
                last = i
        rr = [(peaks[i + 1] - peaks[i]) / fs for i in range(len(peaks) - 1)]
        hr = 60.0 / (fsum(rr) / len(rr)) if rr else None
        lf = None
        if len(rr) >= 8:
            hrs = [60.0 / v for v in rr]
            frr = len(rr) / epoch_s          # mean sampling rate of the HR series
            fr, ps = _bsapsd(hrs, frr)
            tot = fsum(ps)
            if tot > 0.0:
                lf = _bsabandpow(fr, ps, 0.01, 0.04) / tot
        base = max(s[sl])
        lo = min(s[sl])
        srms = _bsarms(q[sl])
        snores.append(srms)
        rows.append({"t_start_s": k * epoch_s, "min_spo2": lo,
                     "baseline_spo2": base, "desat_depth_pct": base - lo,
                     "snore_rms": srms, "mean_hr_bpm": hr,
                     "hr_lf_fraction": lf})
    smed = sorted(snores)[len(snores) // 2]
    for row in rows:
        score = 0
        if row["desat_depth_pct"] >= desat_pct:
            score += 1
        if smed > 0.0 and row["snore_rms"] >= 1.5 * smed:
            score += 1
        if row["hr_lf_fraction"] is not None and row["hr_lf_fraction"] >= 0.3:
            score += 1
        row["score"] = score
        row["epoch_flagged"] = score >= 2
    nflag = sum(1 for r in rows if r["epoch_flagged"])
    hours = n_ep * epoch_s / 3600.0
    idx = nflag / hours if hours > 0.0 else 0.0
    sev = ("none" if idx < 5.0 else "mild" if idx < 15.0
           else "moderate" if idx < 30.0 else "severe")
    return RichResult(payload={
        "epochs": rows, "n_flagged": nflag, "apnea_suspected": idx >= 5.0,
        "events_per_hour": idx, "severity": sev,
        "n_epochs": n_ep, "epoch_s": epoch_s, "fs_hz": fs,
        "desat_criterion_pct": desat_pct, "heuristic": True,
        "units": {"spo2": "percent", "ecg": "mV", "hr": "bpm",
                  "events_per_hour": "1/h", "time": "s"},
        "method": "Rangayyan (2024) Sections 10.2.5, 10.13 and 2.4 (multimodal apnea detection); the combination rule is a documented heuristic, not a trained classifier, and is not given in the book",
    })


rangayyan_sleep_apnea_detect = apneadet  # pre-policy spelling


# -- rgspeech: Speech signal formant and pitch extraction.
def speechfeat(speech, fs, order=None, n_formants=4, f0_range=(60.0, 400.0)):
    """Formant frequencies and pitch of a short segment of speech.

    Rangayyan (2024) Section 7.2.3 ("Formants and pitch in speech")
    models speech as the convolution of the (time-variant) vocal-tract
    impulse response with the glottal input, with a POINT PROCESS input
    for voiced speech and a RANDOM input for unvoiced speech.  The book
    states: "In the case of voiced speech, the IPI statistics of the
    point-process input, in particular its mean, are related to the pitch.
    Furthermore, the frequency response of the filter H(omega)
    representing the vocal tract determines the spectral content of the
    speech signal: The dominant frequencies or peaks are known as formants
    in the case of voiced speech."  The all-pole model of the vocal tract
    is that of Section 7.5 ("Autoregressive or All-pole Modeling"), fitted
    here by the Levinson-Durbin recursion; the pitch is taken from the
    first autocorrelation peak, the discrete equivalent of the mean
    inter-pulse interval the book refers to.

    WHY: formants and pitch come from two physically separate parts of the
    same production system -- the vocal tract shape and the glottal
    pulsing rate -- and they must be estimated separately.  Reading peaks
    off a raw periodogram confuses them, because the harmonics of the
    glottal pulse train dominate the spectrum and are NOT formants.  The
    all-pole model estimates the tract alone; the autocorrelation
    estimates the excitation period alone.

    Parameters
    ----------
    speech : array-like
        Speech segment, in arbitrary amplitude units.  Should be short
        enough (20-30 ms) that the vocal-tract shape can be treated as
        fixed, as the book requires, and long enough to contain at least
        two glottal periods.
    fs : float
        Sampling rate in hertz (Hz); positive.
    order : int or None
        All-pole model order.  ``None`` uses the standard rule
        2 + fs/1000 rounded down, which allots roughly one pole pair per
        kilohertz for the formants plus two poles for the spectral tilt.
    n_formants : int
        Number of formants to report, lowest frequency first; >= 1.
    f0_range : (float, float)
        Plausible fundamental-frequency range in hertz (Hz) for the
        autocorrelation pitch search.  Default 60-400 Hz spans adult male
        to child voices.

    Returns
    -------
    RichResult
        ``formants_hz`` -- formant frequencies in ascending order, with
        ``formant_bandwidths_hz`` and ``formant_powers``;
        ``f0_hz`` -- fundamental frequency, or None if unvoiced;
        ``pitch_period_ms``; ``voiced`` -- True when the normalised
        autocorrelation peak exceeds ``voicing_threshold``;
        ``voicing_strength`` (dimensionless, 0-1);
        ``zero_crossing_rate`` per second -- high for unvoiced speech;
        ``order``, ``ar_coeffs``, ``prediction_error``;
        ``freq_hz`` and ``lpc_psd``.
    """
    xs = [float(v) for v in aslist(speech)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    p = int(order) if order is not None else int(2 + fs / 1000.0)
    if p < 4:
        raise ValueError("order must be at least 4 to resolve a formant")
    if len(xs) < 4 * p:
        raise ValueError("need at least 4*order samples for the all-pole fit")
    n_formants = int(n_formants)
    if n_formants < 1:
        raise ValueError("n_formants must be at least 1")
    flo, fhi = float(f0_range[0]), float(f0_range[1])
    if not 0.0 < flo < fhi:
        raise ValueError("f0_range must satisfy 0 < lo < hi (Hz)")
    if fhi > fs / 2.0:
        raise ValueError("f0_range upper edge exceeds the Nyquist frequency")
    a, err = _bsalpc(xs, p)
    freqs, psd = _bsalpcspec(a, fs, npts=2048)
    psd = [err * v for v in psd]
    # formants: the strongest peaks, then re-sorted by frequency
    pk = _bsapeaks(freqs, psd, count=n_formants, minsep=90.0)
    pk.sort()
    fmt = [f for f, _ in pk]
    bws, pws = [], []
    for f, pw in pk:
        bw, _q = _bsaqfactor(freqs, psd, f)
        bws.append(bw)
        pws.append(pw)
    lag_lo = max(1, int(fs / fhi))
    lag_hi = min(len(xs) - 1, int(fs / flo))
    if lag_hi <= lag_lo:
        raise ValueError("segment is too short for the requested f0_range")
    acf = _bsaacf(xs, lag_hi)
    if acf[0] <= 0.0:
        raise ValueError("segment is constant; no pitch to estimate")
    best = max(range(lag_lo, lag_hi + 1), key=lambda k: acf[k])
    strength = max(0.0, acf[best] / acf[0])
    voiced = strength >= 0.3
    zc = sum(1 for i in range(1, len(xs)) if (xs[i - 1] < 0.0) != (xs[i] < 0.0))
    return RichResult(payload={
        "formants_hz": fmt, "formant_bandwidths_hz": bws,
        "formant_powers": pws,
        "f0_hz": (fs / best) if voiced else None,
        "pitch_period_ms": (1000.0 * best / fs) if voiced else None,
        "voiced": voiced, "voicing_strength": strength,
        "voicing_threshold": 0.3,
        "zero_crossing_rate": zc * fs / len(xs),
        "order": p, "ar_coeffs": a, "prediction_error": err,
        "freq_hz": freqs, "lpc_psd": psd, "fs_hz": fs,
        "units": {"frequency": "Hz", "period": "ms",
                  "zero_crossing_rate": "1/s"},
        "method": "Rangayyan (2024) Section 7.2.3 with all-pole vocal-tract modelling of Section 7.5; pitch from the autocorrelation peak (mean inter-pulse interval of the point-process excitation)",
    })


rangayyan_speech_features = speechfeat  # pre-policy spelling


# -- rgvag: Vibroarthrogram (VAG) signal characterization.
def vagfeat(vag, fs, n_segments=8):
    """Statistical characterisation of a vibroarthrogram.

    Rangayyan (2024) Section 5.12.3 ("Screening of VAG signals using
    statistical parameters") states that VAG signals related to knee-joint
    pathology "have been observed to possess a larger extent of
    variability over the duration of a swing cycle of the leg than normal
    VAG signals", and lists the parameters used: the form factor FF of
    Section 5.6.4, skewness, kurtosis, entropy, an adaptive turns count,
    and the variance of the mean-squared value over segments (Moussavi et
    al. used the variance of the segment means).  The book gives

        mobility  M_x = sigma_{x'} / sigma_x                  eq. (5.25)
        form factor FF = (sigma_{x''}/sigma_{x'}) / M_x       eq. (5.26)
        k-th central moment m_k = sum_l (x_l - mu)^k p_x(x_l) eq. (5.31)

    and notes (Section 3.2.1 discussion) that the Gaussian kurtosis is 3,
    so K' = K - 3 is the kurtosis excess.  The form factor of a sinusoid
    is unity and rises with waveform complexity.

    WHY: a healthy knee joint has smooth cartilage and its VAG signal over
    a swing cycle is close to stationary low-level noise.  Roughened or
    eroded cartilage produces intermittent bursts of vibration, so the
    signal becomes non-stationary and heavy-tailed.  That is why the
    discriminating features are measures of VARIABILITY and of tail weight
    -- form factor, kurtosis, the variance of the segment mean-square --
    rather than any overall amplitude.

    Parameters
    ----------
    vag : array-like
        VAG signal over one swing cycle, in arbitrary amplitude units
        (millivolts at the accelerometer output, in the usual setup).
        At least 4 * n_segments samples.
    fs : float
        Sampling rate in hertz (Hz); positive.  VAG signals are commonly
        sampled at 2 kHz.
    n_segments : int
        Number of equal, non-overlapping segments over which the
        segment-wise statistics are computed; >= 2.

    Returns
    -------
    RichResult
        ``mean``, ``variance``, ``skewness``, ``kurtosis`` (raw fourth
        standardised moment, 3 for a Gaussian) and ``kurtosis_excess``;
        ``form_factor``, ``mobility``, ``activity`` -- eqs. (5.25),
        (5.26); ``turns_count`` and ``turns_per_second``;
        ``var_of_segment_means`` and ``var_of_segment_ms`` -- the
        variability measures of Section 5.12.3;
        ``entropy_bits`` -- Shannon entropy of the amplitude histogram;
        ``rms``; ``duration_s``.
    """
    xs = [float(v) for v in aslist(vag)]
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive (Hz)")
    ns = int(n_segments)
    if ns < 2:
        raise ValueError("n_segments must be at least 2")
    if len(xs) < 4 * ns:
        raise ValueError("need at least 4*n_segments samples")
    mu, var, sk, ku = _bsamoments(xs)
    hj = _bsahjorth(xs)
    seg = len(xs) // ns
    means = [fsum(xs[i * seg:(i + 1) * seg]) / seg for i in range(ns)]
    mss = [fsum(v * v for v in xs[i * seg:(i + 1) * seg]) / seg for i in range(ns)]

    def varof(v):
        m = fsum(v) / len(v)
        return fsum((a - m) ** 2 for a in v) / len(v)

    turns = 0
    for i in range(1, len(xs) - 1):
        if (xs[i] - xs[i - 1]) * (xs[i + 1] - xs[i]) < 0.0:
            turns += 1
    # Shannon entropy of a 64-bin amplitude histogram, eq. (5.31) style
    lo, hi = min(xs), max(xs)
    if hi > lo:
        nb = 64
        counts = [0] * nb
        for v in xs:
            b = int((v - lo) / (hi - lo) * nb)
            counts[min(b, nb - 1)] += 1
        n = len(xs)
        ent = -fsum((c / n) * log(c / n, 2.0) for c in counts if c > 0)
    else:
        ent = 0.0
    return RichResult(payload={
        "mean": mu, "variance": var, "skewness": sk,
        "kurtosis": ku, "kurtosis_excess": ku - 3.0,
        "form_factor": hj["form_factor"], "mobility": hj["mobility"],
        "activity": hj["activity"],
        "turns_count": turns, "turns_per_second": turns * fs / len(xs),
        "var_of_segment_means": varof(means),
        "var_of_segment_ms": varof(mss),
        "segment_means": means, "segment_ms": mss,
        "entropy_bits": ent, "rms": _bsarms(xs),
        "duration_s": len(xs) / fs, "fs_hz": fs, "n_segments": ns,
        "units": {"amplitude": "signal units (mV at the accelerometer)",
                  "entropy": "bits", "turns_per_second": "1/s",
                  "form_factor": "dimensionless"},
        "method": "Rangayyan (2024) Section 5.12.3 with eqs. (5.25), (5.26) and (5.31), Sections 5.6.4 and 3.2.1",
    })


rangayyan_vag_analysis = vagfeat  # pre-policy spelling


# -- rgvagkn: VAG-based knee-joint cartilage pathology detection.
def vagknee(vag, fs, weights=None, bias=None, n_segments=8):
    """Screen a vibroarthrogram for knee-joint cartilage pathology.

    Rangayyan (2024) Section 10.12 ("Application: Detection of Knee-joint
    Cartilage Pathology") treats the screening of VAG signals as a
    supervised two-class problem, and Section 5.12.3 supplies the feature
    set -- form factor (eq. 5.26), skewness, kurtosis, entropy, turns
    count and the variance of the segment mean-squared value.  Section
    10.4.1 gives the linear discriminant function; the decision here is
    the sign of

        d(x) = w0 + sum_i w_i x_i,

    with the features standardised as described below.  Section 10.9.1
    covers the ROC evaluation that any such threshold requires.

    THIS BLOCK DOES NOT INVENT A TRAINED CLASSIFIER.  The book gives
    published accuracies but not a coefficient vector, so with
    ``weights=None`` the returned decision uses an explicitly documented
    heuristic discriminant that scores the DIRECTION each feature is
    reported to move in with pathology (higher form factor, higher
    kurtosis excess, higher segment-to-segment variability, higher turns
    rate).  Pass ``weights`` from your own trained model for a real
    classification; the heuristic exists so the function returns a usable
    number rather than nothing, and it is flagged as such in the payload.

    WHY: cartilage pathology roughens the articular surfaces, so the
    vibration recorded during a swing cycle stops being stationary
    low-level noise and becomes a sequence of intermittent bursts.  Every
    feature above is a different way of measuring that same
    non-stationarity, which is why they are used together rather than
    singly.

    Parameters
    ----------
    vag : array-like
        VAG signal over one swing cycle, arbitrary amplitude units.
    fs : float
        Sampling rate in hertz (Hz); positive.
    weights : sequence of float or None
        Five discriminant coefficients, in the order (form_factor,
        kurtosis_excess, log var_of_segment_ms, turns_per_second,
        entropy_bits).  ``None`` selects the documented heuristic.
    bias : float or None
        The w0 term.  Required when ``weights`` is given.
    n_segments : int
        Segments used for the variability features; >= 2.

    Returns
    -------
    RichResult
        ``pathology_suspected`` -- boolean, d(x) > 0;
        ``discriminant`` -- d(x), dimensionless;
        ``features`` -- the five feature values actually used;
        ``weights_used``, ``bias_used``, ``trained`` (False for the
        heuristic); plus the full feature payload of ``vagfeat``.
    """
    base = vagfeat(vag, fs, n_segments=n_segments)
    varms = base["var_of_segment_ms"]
    feats = [base["form_factor"], base["kurtosis_excess"],
             log(varms) if varms > 0.0 else -30.0,
             base["turns_per_second"], base["entropy_bits"]]
    if weights is None:
        if bias is not None:
            raise ValueError("bias may only be given together with weights")
        # Heuristic: each feature is scored by how far it exceeds a
        # nominal normal-VAG level, with signs set by the direction the
        # book reports for pathology.  NOT a trained classifier.
        w = [1.0, 0.5, 0.2, 0.01, 0.5]
        b = -(1.0 * 1.2 + 0.5 * 0.0 + 0.2 * log(1e-4) + 0.01 * 200.0
              + 0.5 * 5.0)
        trained = False
    else:
        w = [float(v) for v in aslist(weights)]
        if len(w) != 5:
            raise ValueError("weights must have 5 coefficients")
        if bias is None:
            raise ValueError("bias is required when weights are given")
        b = float(bias)
        trained = True
    d = b + fsum(wi * fi for wi, fi in zip(w, feats))
    out = dict(base)
    out.update({
        "pathology_suspected": d > 0.0,
        "discriminant": d,
        "features": {"form_factor": feats[0], "kurtosis_excess": feats[1],
                     "log_var_segment_ms": feats[2],
                     "turns_per_second": feats[3],
                     "entropy_bits": feats[4]},
        "weights_used": w, "bias_used": b, "trained": trained,
        "method": "Rangayyan (2024) Section 10.12 with the VAG feature set of Section 5.12.3 and the linear discriminant of Section 10.4.1; coefficients are the caller's unless the documented untrained heuristic is used",
    })
    return RichResult(payload=out)


rangayyan_vag_knee_cartilage = vagknee  # pre-policy spelling


# -- rng029: Decomposition of a signal into weighted deltas (Rangayyan eq. 3.29).
def deltadecomp(x, t=None):
    """Resolve a signal into a weighted combination of shifted deltas.

    Rangayyan (2024) eq. (3.29):
        x(t) = integral x(alpha) delta(t - alpha) d alpha.

    The book reads this as resolving x into mutually orthogonal delta
    functions.  Discretely, the weight carried by the delta at alpha_i is
    x(alpha_i) times the grid spacing, so that summing the weights
    reproduces the integral of x rather than the sum of its samples;
    reconstructing from the weights returns the original samples exactly,
    which is the check returned in ``reconstruction_error``.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    ts = [float(i) for i in range(n)] if t is None else aslist(t)
    if len(ts) != n:
        raise ValueError("t and x must have the same length")
    if n == 1:
        dt = [1.0]
    else:
        # trapezoidal weights: half a spacing at each end, so that the
        # weights sum to the integral of x rather than overcounting the
        # two endpoints by half a panel each.
        dt = []
        for i in range(n):
            lo = ts[i] - ts[i - 1] if i > 0 else 0.0
            hi = ts[i + 1] - ts[i] if i < n - 1 else 0.0
            dt.append(0.5 * (lo + hi))
    weights = [v * d for v, d in zip(xs, dt)]
    recon = [w / d for w, d in zip(weights, dt)]
    err = max(abs(a - b) for a, b in zip(recon, xs))
    return RichResult(payload={
        "locations": ts, "weights": weights, "amplitudes": xs,
        "total_weight": sum(weights),
        "integral": gridint(xs, ts) if n > 1 else 0.0,
        "reconstruction_error": err,
        "method": "Rangayyan (2024) eq. (3.29)"})


rangayyan_ch3_signal_as_delta_decomposition = deltadecomp  # pre-policy spelling


# -- rng235: Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum..
def clogprod(X, H, omega=None):
    """Complex logarithm turning the product Y = X H into a sum of logs.

    Rangayyan (2024) eq. (4.63), Section 4.8 (homomorphic filtering).  For
    y(t) = x(t) * h(t) the convolution theorem gives eq. (4.62),
    Y(omega) = X(omega) H(omega), and taking the complex logarithm gives

        log[Y(omega)] = log[X(omega)] + log[H(omega)];
        X(omega) != 0, H(omega) != 0 for all omega,

    where the book's accompanying note defines the complex logarithm as

        log_e[X(omega)] = log_e[|X(omega)|] + j angle X(omega),

    i.e. the log magnitude and the phase spectrum are the real and
    imaginary parts.

    WHY: this is the whole point of the homomorphic (cepstral) filter.
    Convolution of an excitation with a system response is not separable
    by any linear filter, but the Fourier transform turns it into a
    product and the complex log turns the product into a sum -- and sums
    ARE separable by a linear filter, provided the two components occupy
    different regions of the transformed domain.  Everything downstream
    (glottal excitation vs vocal tract, S1 vs the aortic component of S2)
    depends on this one identity.

    Parameters
    ----------
    X, H : array-like of complex
        The spectra X(omega) and H(omega), sampled on the same frequency
        grid, in arbitrary but identical amplitude units.  Real sequences
        are accepted and read as complex with zero imaginary part.  No
        element of either may be zero, per the condition stated in the
        equation.
    omega : array-like or None
        The frequency grid, in radians per sample (or Hz -- the values
        are only carried through to the payload).  Must match X and H in
        length when given.

    Returns
    -------
    RichResult
        ``log_Y_real`` / ``log_Y_imag`` -- log|Y| and unwrapped phase of Y;
        ``log_X_real`` / ``log_X_imag`` and ``log_H_real`` / ``log_H_imag``
        -- the same for the two factors;
        ``Y_real`` / ``Y_imag`` -- the product itself;
        ``max_abs_error`` -- the largest deviation of
        log Y - (log X + log H) over the grid, which is zero to machine
        precision when the identity holds and is returned as the check
        that it does.

    Notes
    -----
    Phases are unwrapped along the grid before the sum is checked; the
    principal-value phase of a product jumps by 2 pi and would otherwise
    make an exactly correct identity look violated.
    """
    xs = [complex(v) for v in aslistc(X)]
    hs = [complex(v) for v in aslistc(H)]
    if len(xs) != len(hs):
        raise ValueError("X and H must have the same length")
    if not xs:
        raise ValueError("X and H must be non-empty")
    if omega is not None:
        om = [float(v) for v in aslistc(omega)]
        if len(om) != len(xs):
            raise ValueError("omega must have the same length as X and H")
    else:
        om = list(range(len(xs)))
    if any(v == 0 for v in xs):
        raise ValueError("X(omega) must be non-zero for all omega (eq. 4.63)")
    if any(v == 0 for v in hs):
        raise ValueError("H(omega) must be non-zero for all omega (eq. 4.63)")

    def unwrap(ph):
        out = [ph[0]]
        for v in ph[1:]:
            d = v - out[-1]
            while d > pi:
                d -= 2.0 * pi
            while d < -pi:
                d += 2.0 * pi
            out.append(out[-1] + d)
        return out

    ys = [a * b for a, b in zip(xs, hs)]
    lxr = [log(abs(v)) for v in xs]
    lhr = [log(abs(v)) for v in hs]
    lyr = [log(abs(v)) for v in ys]
    lxi = unwrap([atan2(v.imag, v.real) for v in xs])
    lhi = unwrap([atan2(v.imag, v.real) for v in hs])
    lyi = unwrap([atan2(v.imag, v.real) for v in ys])
    err = 0.0
    for k in range(len(xs)):
        er = abs(lyr[k] - (lxr[k] + lhr[k]))
        # the unwrapped product phase may differ from the sum of the
        # unwrapped factor phases by a constant multiple of 2 pi
        ei = (lyi[k] - (lxi[k] + lhi[k])) / (2.0 * pi)
        ei = abs(ei - round(ei)) * 2.0 * pi
        err = max(err, er, ei)
    return RichResult(payload={
        "omega": om,
        "Y_real": [v.real for v in ys], "Y_imag": [v.imag for v in ys],
        "log_Y_real": lyr, "log_Y_imag": lyi,
        "log_X_real": lxr, "log_X_imag": lxi,
        "log_H_real": lhr, "log_H_imag": lhi,
        "max_abs_error": err,
        "units": {"log magnitude": "nepers", "phase": "radians (unwrapped)"},
        "method": "Rangayyan (2024) eq. (4.63), complex log of a product, Section 4.8 homomorphic filtering",
    })


rangayyan_ch4_complex_log_of_product = clogprod  # pre-policy spelling


# -- rng240: Complex log of X(z) expanded as a sum of log terms over poles and zeros..
def clogpz(z, A=1.0, r=0, a_k=(), b_k=(), c_k=(), d_k=(),
           M_I=None, M_O=None, N_I=None, N_O=None):
    """Complex cepstrum of a rational X(z) as a sum over its poles and zeros.

    Rangayyan (2024) eq. (4.68), Section 4.8.  Starting from the
    factored rational form of eq. (4.67), whose zeros and poles satisfy
    |a_k|, |b_k|, |c_k|, |d_k| < 1, the complex logarithm expands into

        X_hat(z) = log[X(z)]
                 = log[A] + log[z^r]
                   + sum_{k=1..M_I} log(1 - a_k z^-1)
                   + sum_{k=1..M_O} log(1 - b_k z)
                   - sum_{k=1..N_I} log(1 - c_k z^-1)
                   - sum_{k=1..N_O} log(1 - d_k z),

    where (eq. 4.67) X(z) has M_I zeros inside the unit circle at a_k,
    M_O zeros outside at 1/b_k, N_I poles inside at c_k, and N_O poles
    outside at 1/d_k.

    WHY: the log of a rational function is not rational, but it IS a sum
    of elementary logs, each of which has a known, closed-form inverse
    z-transform.  That is what makes the complex cepstrum computable in
    the first place, and it is why poles and zeros inside the unit circle
    contribute only to positive quefrency while those outside contribute
    only to negative quefrency -- the property the homomorphic filter
    exploits to separate minimum-phase from maximum-phase components.

    Parameters
    ----------
    z : complex or array-like of complex
        Point(s) in the z-plane at which to evaluate X_hat(z).
        Dimensionless.  None may be zero when r != 0 or when any a_k or
        c_k is non-zero.
    A : complex
        Overall gain of eq. (4.67), in the amplitude units of x.  Must be
        non-zero.
    r : int
        The exponent of the z^r linear-phase (pure-delay) factor,
        dimensionless.
    a_k, b_k, c_k, d_k : sequences of complex
        Zeros inside, zeros outside (as reciprocals), poles inside and
        poles outside (as reciprocals) respectively.  All must have
        modulus strictly less than 1, as eq. (4.67) requires.
    M_I, M_O, N_I, N_O : int or None
        Optional expected counts.  When given they must equal the lengths
        of a_k, b_k, c_k and d_k; they exist only so that the caller can
        assert the model order matches the factor lists.

    Returns
    -------
    RichResult
        ``z_real`` / ``z_imag`` -- the evaluation points;
        ``xhat_real`` / ``xhat_imag`` -- log magnitude (nepers) and phase
        (radians) of X_hat(z) at each point;
        ``X_real`` / ``X_imag`` -- X(z) itself, rebuilt from eq. (4.67);
        ``max_abs_error`` -- the largest |exp(X_hat(z)) - X(z)|, which
        checks that the log expansion and the product form agree;
        ``terms`` -- the four partial sums (gain+delay, zeros in, zeros
        out, poles in, poles out) at the first evaluation point.
    """
    zs = [complex(v) for v in aslistc(z)]
    if not zs:
        raise ValueError("z must contain at least one point")
    A = complex(A)
    if A == 0:
        raise ValueError("A must be non-zero (eq. 4.67 gain)")
    r = int(r)
    sets = {"a_k": [complex(v) for v in aslistc(a_k)],
            "b_k": [complex(v) for v in aslistc(b_k)],
            "c_k": [complex(v) for v in aslistc(c_k)],
            "d_k": [complex(v) for v in aslistc(d_k)]}
    for nm, vals in sets.items():
        for v in vals:
            if abs(v) >= 1.0:
                raise ValueError(nm + " entries must have modulus < 1 (eq. 4.67)")
    for cnt, nm in ((M_I, "a_k"), (M_O, "b_k"), (N_I, "c_k"), (N_O, "d_k")):
        if cnt is not None and int(cnt) != len(sets[nm]):
            raise ValueError("declared count does not match the length of " + nm)
    ak, bk, ck, dk = sets["a_k"], sets["b_k"], sets["c_k"], sets["d_k"]
    if r != 0 or ak or ck:
        if any(v == 0 for v in zs):
            raise ValueError("z = 0 is a singularity of this expansion")

    def clog(w):
        if w == 0:
            raise ValueError("log of zero: a factor vanishes at this z")
        return complex(log(abs(w)), atan2(w.imag, w.real))

    xhat, xval, first = [], [], None
    for zi in zs:
        gain = clog(A) + (r * clog(zi) if r else 0j)
        s_ai = sum((clog(1.0 - v / zi) for v in ak), 0j)
        s_bo = sum((clog(1.0 - v * zi) for v in bk), 0j)
        s_ci = sum((clog(1.0 - v / zi) for v in ck), 0j)
        s_do = sum((clog(1.0 - v * zi) for v in dk), 0j)
        tot = gain + s_ai + s_bo - s_ci - s_do
        xhat.append(tot)
        num = A * (zi ** r if r else 1.0)
        for v in ak:
            num *= (1.0 - v / zi)
        for v in bk:
            num *= (1.0 - v * zi)
        den = 1.0 + 0j
        for v in ck:
            den *= (1.0 - v / zi)
        for v in dk:
            den *= (1.0 - v * zi)
        if den == 0:
            raise ValueError("X(z) has a pole at one of the evaluation points")
        xval.append(num / den)
        if first is None:
            first = {"gain_and_delay": [gain.real, gain.imag],
                     "zeros_inside": [s_ai.real, s_ai.imag],
                     "zeros_outside": [s_bo.real, s_bo.imag],
                     "poles_inside": [s_ci.real, s_ci.imag],
                     "poles_outside": [s_do.real, s_do.imag]}
    err = 0.0
    for k in range(len(zs)):
        try:
            err = max(err, abs(complex(exp(xhat[k].real) * cos(xhat[k].imag),
                                       exp(xhat[k].real) * sin(xhat[k].imag))
                               - xval[k]))
        except OverflowError:
            err = float("inf")
    return RichResult(payload={
        "z_real": [v.real for v in zs], "z_imag": [v.imag for v in zs],
        "xhat_real": [v.real for v in xhat], "xhat_imag": [v.imag for v in xhat],
        "X_real": [v.real for v in xval], "X_imag": [v.imag for v in xval],
        "max_abs_error": err,
        "terms": first,
        "counts": {"M_I": len(ak), "M_O": len(bk), "N_I": len(ck), "N_O": len(dk)},
        "units": {"xhat_real": "nepers (log magnitude)",
                  "xhat_imag": "radians (phase)", "z": "dimensionless"},
        "method": "Rangayyan (2024) eq. (4.68) with eq. (4.67), complex log of a rational X(z) over poles and zeros",
    })


rangayyan_ch4_complex_log_x_z = clogpz  # pre-policy spelling


_CHEATSHEET = [
    'emdsg: Empirical Mode Decomposition (standalone).',
    'Idealised action-potential template: linear upstroke, exponential repolarisation (mV vs ms)',
    '1-D cardiac monodomain propagation plus bidomain extracellular potential (Rangayyan eqs. 7.143-7.149)',
    'AR-spectrum high-frequency power ratio of a diastolic segment for CAD detection',
    'Turbulent coronary-flow sound spectrum from stenosis geometry (Rangayyan eq. 7.136)',
    'Infant cry F0 track (cry melody) coded falling/flat/rising, plus formants',
    'EGG dominant gastric frequency (cpm) and brady/normo/tachygastria power fractions',
    'rgelast: Heart-sound spectral stiffness index.',
    'ENG compound action potential and conduction velocity from a fibre-velocity population',
    "Seizure detection by per-epoch EEG band fractions against the subject's own baseline",
    'ERP component latencies (ms) and baseline-to-peak amplitudes (uV)',
    'Event-related desynchronisation/synchronisation percent of an EEG band for BCI',
    'PSD moments and band-power fractions of a signal segment (Rangayyan eqs. 6.32-6.44)',
    'Goldman-Hodgkin-Katz resting membrane potential from permeability-weighted ion concentrations (mV)',
    'Hodgkin-Huxley m, h, n gate kinetics: steady states, time constants (ms) and rate constants',
    'Four-variable Hodgkin-Huxley action-potential simulation by RK4 (mV vs ms)',
    'FitzHugh-Nagumo two-variable excitable neuron model integrated by RK4',
    'Passive RC membrane relaxation with time constant tau = R_m C_m (mV vs ms)',
    'LMS adaptive cancellation of muscle-contraction artifact from a VAG signal (Rangayyan eqs. 3.203-3.205)',
    'Motor unit action potential built from dispersed single-fibre potentials (uV vs ms)',
    'PA/CA magnitude-spectrum ratio of a systolic murmur (Rangayyan eq. 6.45)',
    'Nernst equilibrium potential of an ion from its concentration ratio (mV)',
    'OAE half-octave band powers, per-band SNR and emission-present decision',
    'Multimodal Parkinson monitoring: EEG bands, EMG tremor fraction and gait regularity',
    'PCG-EEG magnitude-squared coherence spectrum with segment averaging (Rangayyan eq. 4.32)',
    'Murmur presence decision from the PCG power fraction above the S1/S2 band',
    'Sleep staging from fused EEG band fractions, EOG activity and chin EMG tone',
    'Inter-event interval statistics of a point process: mean rate (pps) and CV_r',
    'Resonance frequencies, -3 dB bandwidths and Q factors of prosthetic valve sounds',
    'Per-breath respiratory rate, tidal depth and I:E ratio from a flow or volume signal',
    'Acoustic RLC tube-segment model of an airway: inertance, compliance, losses and resonance',
    'Multimodal sleep apnea screening from SpO2 desaturation, snore energy and HR modulation',
    'Speech formant frequencies from an all-pole vocal-tract model plus autocorrelation pitch',
    'Statistical VAG characterisation: form factor, kurtosis, turns, segment variability',
    'Linear-discriminant screening of a VAG signal for knee cartilage pathology',
    'rng029: Decomposition of a signal into weighted deltas (Rangayyan eq. 3.29).',
    'Complex log turning Y=X*H into log X + log H (Rangayyan eq. 4.63), with the identity checked',
    'Complex log of a rational X(z) as a sum of pole/zero log terms (Rangayyan eq. 4.68)',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)

# Pre-policy run-together spellings.  These were in the lazy
# map but not in the module, so morie.fn.<name> raised
# AttributeError.  Restored rather than dropped, because the
# map is the public flat namespace.
rangayyanegg = rangayyan_egg  # pre-policy spelling, kept live
rangayyaneng = rangayyan_eng  # pre-policy spelling, kept live
rangayyanmuap = rangayyan_muap  # pre-policy spelling, kept live
rangayyanoae = rangayyan_oae  # pre-policy spelling, kept live
