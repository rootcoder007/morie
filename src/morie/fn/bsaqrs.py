# morie.fn -- bsaqrs (rootcoder007/morie)
"""ECG event detection and rate analysis: QRS detectors, P and T waves, heart rate, HRV.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 46
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from math import fsum, sqrt, cos, sin, pi, atan2, hypot
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult

__all__ = [
    'blwander',
    'rangayyan_baseline_wander',
    'cpulsefeat',
    'rangayyan_carotid_pulse',
    'qrsderiv',
    'rangayyan_deriv_qrs',
    'dicnotch',
    'rangayyan_dicrotic_notch',
    'ecgemgcpl',
    'rangayyan_ecg_emg_coupling',
    'ecgfeat',
    'rangayyan_ecg_features',
    'ecgwaveshp',
    'rangayyan_ecg_waveshape',
    'exerecgst',
    'rangayyan_exercise_ecg',
    'hrvfreq',
    'rangayyan_hrv_freq_domain',
    'hrvtime',
    'rangayyan_hrv_time_domain',
    'hsoundid',
    'rangayyan_heart_sound_id',
    'mecgfilt',
    'rangayyan_maternal_ecg_filter',
    'motionart',
    'rangayyan_motion_artifact',
    'qrsdetect',
    'rangayyan_pan_tompkins',
    'pcgparts',
    'rangayyan_pcg_segments',
    'plinenotch',
    'rangayyan_powerline_removal',
    'ppgfeat',
    'rangayyan_ppg_features',
    'pwavedet',
    'rangayyan_p_wave_detect',
    'edrsignal',
    'rangayyan_resp_signal',
    'apneaedr',
    'rangayyan_sleep_apnea',
    'lfhfratio',
    'rangayyan_spectral_power_ratio',
    'twaspectr',
    'rangayyan_twave_alternans',
    'rangayyan_twa_spectral_mx',
    'twavedet',
    'rangayyan_t_wave_detect',
    'vfdetect',
    'rangayyan_vf_detect',
    'qrsderiv1',
    'rangayyan_ch4_qrs_first_derivative_balda',
    'qrsderiv2',
    'rangayyan_ch4_qrs_second_derivative_balda',
    'qrsderivmx',
    'rangayyan_ch4_qrs_combined_balda',
    'qrswsqdrv',
    'rangayyan_ch4_filtered_derivative_murthy',
    'qrsdrvsmth',
    'rangayyan_ch4_qrs_smoothing_ma_filter',
    'qrslpasstf',
    'rangayyan_ch4_pan_tompkins_lowpass_transfer',
    'qrslpassdf',
    'rangayyan_ch4_pan_tompkins_lowpass_difference_eq',
    'qrshplptf',
    'rangayyan_ch4_pan_tompkins_highpass_lp_component',
    'qrshplpdf',
    'rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq',
    'qrshpasstf',
    'rangayyan_ch4_pan_tompkins_highpass_transfer',
    'qrshpassdf',
    'rangayyan_ch4_pan_tompkins_highpass_difference_eq',
    'qrshpassio',
    'rangayyan_ch4_pan_tompkins_highpass_combined',
    'qrsderivop',
    'rangayyan_ch4_pan_tompkins_derivative_operator',
    'qrsmwint',
    'rangayyan_ch4_pan_tompkins_moving_window_integrator',
    'qrsthresh',
    'rangayyan_ch4_pan_tompkins_thresholds',
    'qrsspkiupd',
    'rangayyan_ch4_pan_tompkins_searchback_update',
    'hrfromcnt',
    'rangayyan_ch4_heart_rate_from_count',
    'rangayyan_ch4_heart_rate_from_rr',
    'lengthxfm',
    'rangayyan_ch4_length_transformation',
    'rangayyan_ch4_dicrotic_notch_second_derivative',
    'dnotchsmth',
    'rangayyan_ch4_dicrotic_notch_smoothed_squared',
]

def _rgpad(seq, k):
    """Left-pad a per-sample operator output with k zeros to keep the length."""
    return [0.0] * k + list(seq)


def _rgmavg(x, m):
    """Causal m-point moving average; the first m-1 outputs use a short window."""
    if m < 1:
        raise ValueError("moving-average length must be >= 1")
    out = []
    run = 0.0
    for i, v in enumerate(x):
        run += v
        if i >= m:
            run -= x[i - m]
        out.append(run / float(min(i + 1, m)))
    return out


def _rgdft(x):
    """Plain O(n^2) DFT returning a list of (real, imag) pairs.

    ponytail: O(n^2) is fine for RR tachograms and beat series (hundreds of
    points).  Swap in a radix-2 FFT if this is ever fed whole-night data.
    """
    n = len(x)
    if n < 2:
        raise ValueError("DFT needs at least two samples")
    out = []
    for k in range(n):
        w = -2.0 * pi * k / n
        re = fsum(x[j] * cos(w * j) for j in range(n))
        im = fsum(x[j] * sin(w * j) for j in range(n))
        out.append((re, im))
    return out


def _rgpsd(x, fs):
    """One-sided periodogram of x.  Returns (freqs, power), power in units^2/Hz."""
    n = len(x)
    spec = _rgdft(x)
    nyq = n // 2
    freqs = [fs * k / float(n) for k in range(nyq + 1)]
    power = []
    for k in range(nyq + 1):
        re, im = spec[k]
        p = (re * re + im * im) / (fs * n)
        if 0 < k < n - k:
            p *= 2.0
        power.append(p)
    return freqs, power


def _rgpeaks(g, th, m):
    """Peak set {p} of Rangayyan (2024) Equation 4.6.

    A sample is a peak when it exceeds the threshold th and is strictly
    greater than its m preceding and m succeeding neighbours.
    """
    n = len(g)
    out = []
    for i in range(n):
        if g[i] <= th:
            continue
        lo = max(0, i - m)
        hi = min(n, i + m + 1)
        if all(g[i] > g[j] for j in range(lo, i)) and all(g[i] > g[j] for j in range(i + 1, hi)):
            out.append(i)
    return out


def _rgptbp(x):
    """Pan-Tompkins bandpass: Equation 4.8 lowpass then Equation 4.13 highpass.

    Coefficients are integers tied to fs = 200 Hz (book, Section 4.3.2).
    """
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        a = 2.0 * y[i - 1] if i >= 1 else 0.0
        b = y[i - 2] if i >= 2 else 0.0
        c = x[i] - (2.0 * x[i - 6] if i >= 6 else 0.0) + (x[i - 12] if i >= 12 else 0.0)
        y[i] = a - b + c / 32.0
    # Equation 4.13 acts on the lowpass output y, not on the raw input.
    p = [0.0] * n
    for i in range(n):
        prev = p[i - 1] if i >= 1 else 0.0
        p[i] = (
            prev
            - y[i] / 32.0
            + (y[i - 16] if i >= 16 else 0.0)
            - (y[i - 17] if i >= 17 else 0.0)
            + ((y[i - 32] / 32.0) if i >= 32 else 0.0)
        )
    return p


def _rgptderiv(x):
    """Pan-Tompkins derivative, Equation 4.14."""
    n = len(x)
    out = [0.0] * n
    for i in range(n):
        out[i] = (
            2.0 * x[i]
            + (x[i - 1] if i >= 1 else 0.0)
            - (x[i - 3] if i >= 3 else 0.0)
            - 2.0 * (x[i - 4] if i >= 4 else 0.0)
        ) / 8.0
    return out


def _rgptint(x, w):
    """Moving-window integrator, Equation 4.15."""
    return _rgmavg(x, w)


def _rgchain(x, fs):
    """Full Pan-Tompkins front end: bandpass, derivative, square, integrate.

    The integrator width is the book's N = 30 samples at fs = 200 Hz
    (150 ms), scaled to the supplied fs.
    """
    bp = _rgptbp(x)
    dv = _rgptderiv(bp)
    sq = [v * v for v in dv]
    w = max(1, int(round(0.150 * fs)))
    return bp, dv, sq, _rgptint(sq, w), w


def _rgcorr(a, b):
    """Pearson correlation of two equal-length sequences; 0.0 if either is flat."""
    n = len(a)
    if n != len(b) or n < 2:
        raise ValueError("correlation needs two equal-length series of length >= 2")
    ma = fsum(a) / n
    mb = fsum(b) / n
    sa = fsum((v - ma) ** 2 for v in a)
    sb = fsum((v - mb) ** 2 for v in b)
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return fsum((a[i] - ma) * (b[i] - mb) for i in range(n)) / sqrt(sa * sb)


def _rgcheck(x, least=1, what="signal"):
    """Coerce to a list of floats and reject anything shorter than `least`."""
    v = aslist(x)
    if len(v) < least:
        raise ValueError("%s needs at least %d samples, got %d" % (what, least, len(v)))
    return v


def _rgfs(fs):
    """Validate a sampling rate."""
    fs = float(fs)
    if not fs > 0.0:
        raise ValueError("fs must be positive")
    return fs


# -- rgblwand: Baseline wander removal from ECG.
def blwander(ecg, fs, pole=0.995):
    """Baseline-wander removal by the modified first-order difference
    filter, eqs. (3.132)-(3.133).

        H(z) = (1/T) (1 - z^-1) / (1 - 0.995 z^-1)

    The plain first-order difference of eq. (3.123) already has the zero
    at z = 1 that kills DC, which is what baseline wander is.  Its
    problem is that its magnitude response stays low far beyond the
    wander band, so it distorts the QRS complex it was supposed to
    preserve.

    The book's fix is to boost the gain back up by placing a POLE on the
    real axis near the zero, at z = 0.995, so the response climbs to
    roughly unity by about 0.5 Hz -- close enough to the zero to leave
    DC rejected, far enough that everything above the wander band comes
    through at its original level.

    The stub this replaces claimed "high-pass > 0.05 Hz Butterworth or
    cubic spline through isoelectric points".  Neither appears in the
    book for this problem: the text says the gain should be near unity
    "after about 0.5 Hz", an order of magnitude away, and derives the
    pole-zero filter above rather than a Butterworth or a spline.

    The 1/T factor makes the difference an estimate of the derivative;
    it is applied here as written, so the output is scaled by fs.  Pass
    normalize=... via `pole` only to move the pole -- a pole closer to 1
    gives a narrower transition and a longer transient.
    """
    xs = aslist(ecg)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    a = float(pole)
    if not 0.0 <= a < 1.0:
        raise ValueError("the pole must lie inside the unit circle, "
                         "0 <= pole < 1; got %g" % a)

    # y(n) = a y(n-1) + (1/T) [ x(n) - x(n-1) ],  T = 1/fs
    y = [0.0] * n
    prev = 0.0
    for i in range(1, n):
        prev = a * prev + fsv * (xs[i] - xs[i - 1])
        y[i] = prev

    # Magnitude response at DC and at the 0.5 Hz design point, so a
    # caller can see the two properties the design is FOR rather than
    # taking them on trust.
    def _gain(f):
        w = 2.0 * pi * f / fsv
        num = complex(1.0 - cos(w), sin(w))          # 1 - e^-jw
        den = complex(1.0 - a * cos(w), a * sin(w))  # 1 - a e^-jw
        if abs(den) == 0.0:
            return float("inf")
        return abs(num / den) * fsv

    g0 = _gain(0.0)
    ghalf = _gain(0.5)
    gnyq = _gain(fsv / 2.0)
    return RichResult(payload={
        "ecg_detrended": y, "n": n, "fs": fsv, "pole": a,
        "gain_dc": g0, "gain_at_half_hz": ghalf, "gain_at_nyquist": gnyq,
        "gain_relative_at_half_hz": (ghalf / gnyq) if gnyq > 0 else None,
        "dc_is_rejected": g0 < 1e-12,
        "zero_at_z_equals_one": True,
        "pole_restores_gain_above_the_wander_band": True,
        "differentiates_by_the_one_over_T_factor": True,
        "method": "Rangayyan (2024) eqs. (3.132)-(3.133), modified "
                  "first-order difference for baseline-wander removal"})


rangayyan_baseline_wander = blwander  # pre-policy spelling


# -- rgcpulse: Carotid pulse waveform feature extraction.
def cpulsefeat(cp, fs, qrs, hr=None):
    """Carotid pulse waveform features and systolic time intervals.

    The waveform landmarks are those of Rangayyan (2024) Section 1.2.10: the
    abrupt upstroke on ejection, the percussion wave (the systolic peak), the
    dicrotic notch marking aortic valve closure, and the dicrotic wave that
    may follow it.

    The intervals are those defined in Section 4.9.  The pre-ejection period
    PEP runs from the onset of the QRS to the onset of the carotid upstroke;
    the ejection time ET runs from the upstroke onset to the dicrotic notch.
    Both are rate-dependent, so the book corrects them:

        PEPC = PEP + 0.4 HR       (normal adults 131 +/- 13 ms)
        ETC  = ET  + 1.6 HR       (males 395 +/- 13 ms, females 415 +/- 11 ms)

    with the periods in ms and HR in bpm.  The corrections are what make the
    values comparable between subjects recorded at different heart rates.

    Parameters
    ----------
    cp : sequence of float
        Carotid pulse samples.
    fs : float
        Sampling rate in Hz.
    qrs : sequence of int
        QRS onset sample positions, from a QRS detector.
    hr : float, optional
        Heart rate in bpm used for the rate corrections.  Defaults to the rate
        implied by the mean interval between the supplied QRS positions.

    Returns
    -------
    RichResult
        Per-beat lists: "upstroke", "percussion", "notch", "dicwave",
        "pep", "et" (ms) and "pepc", "etc" (ms), plus their means.
    """
    fs = _rgfs(fs)
    cp = _rgcheck(cp, 16, "carotid pulse")
    q = [int(v) for v in aslist(qrs)]
    if len(q) < 1:
        raise ValueError("need at least one QRS position")
    if hr is None:
        if len(q) < 2:
            raise ValueError("hr must be given when fewer than two QRS positions are supplied")
        hr = 60.0 * fs / (fsum(q[i] - q[i - 1] for i in range(1, len(q))) / (len(q) - 1))
    hr = float(hr)
    if not hr > 0.0:
        raise ValueError("hr must be positive")

    dn = dicnotch(cp, fs, qrs=q)
    notch = list(dn["notch"])
    ups = list(dn["upstroke"])
    n = len(cp)
    span = int(round(0.500 * fs))

    perc, dicw, pep, et = [], [], [], []
    for k in range(min(len(notch), len(ups))):
        u, d = ups[k], notch[k]
        if d <= u:
            continue
        perc.append(max(range(u, d), key=lambda i: cp[i]))
        hi = min(n, d + int(round(0.200 * fs)))
        dicw.append(max(range(d, hi), key=lambda i: cp[i]) if hi > d + 1 else d)
        # the QRS onset preceding this upstroke
        prior = [v for v in q if v <= u and u - v <= span]
        if prior:
            pep.append(1000.0 * (u - prior[-1]) / fs)
        et.append(1000.0 * (d - u) / fs)

    pepc = [v + 0.4 * hr for v in pep]
    etc = [v + 1.6 * hr for v in et]
    mean = lambda v: (fsum(v) / len(v)) if v else None
    return RichResult(payload={
        "upstroke": ups,
        "percussion": perc,
        "notch": notch,
        "dicwave": dicw,
        "pep": pep,
        "et": et,
        "pepc": pepc,
        "etc": etc,
        "peppmean": mean(pep),
        "etmean": mean(et),
        "pepcmean": mean(pepc),
        "etcmean": mean(etc),
        "hr": hr,
        "fs": fs,
        "normpepc": (131.0, 13.0),
        "normetcmale": (395.0, 13.0),
        "normetcfemale": (415.0, 11.0),
        "method": "carotid pulse features and systolic time intervals, Rangayyan (2024) Sections 1.2.10 and 4.9",
    })


rangayyan_carotid_pulse = cpulsefeat  # pre-policy spelling


# -- rgderqrs: Derivative-based QRS detection (first and second differences).
def qrsderiv(x, fs, thresh=1.0):
    """Derivative-based QRS detection from the first and second differences.

    Rangayyan (2024) Section 4.3.1 (Balda et al.): form y0 by Eq 4.1 and y1 by
    Eq 4.2, combine them as y2 = 1.3 y0 + 1.1 y1 (Eq 4.3), then scan y2 with a
    threshold of 1.0 on a maximum-normalised ECG.  Wherever the threshold is
    crossed the next eight samples are tested against the same threshold, and
    the eight-sample segment is accepted as part of a QRS when at least six of
    them pass.  The six-of-eight rule is what makes the decision robust to the
    single-sample spikes that a derivative operator produces from noise.

    The book also smooths y2 with the eight-point moving-average filter of its
    Equation 3.108 to obtain y3, a single pulse per QRS; that smoothed signal
    is returned as well and its peaks give the beat positions.

    The signal is normalised by its maximum absolute value before the
    threshold is applied, matching the book's illustration.

    Parameters
    ----------
    x : sequence of float
        ECG samples, ideally lowpass filtered and free of baseline drift.
    fs : float
        Sampling rate in Hz.
    thresh : float
        Threshold applied to y2.  The book uses 1.0.

    Returns
    -------
    RichResult
        payload["qrs"] beat positions, payload["y0"], ["y1"], ["y2"], ["y3"],
        and payload["mask"], the accepted eight-sample QRS segments.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 16, "ECG")
    thresh = float(thresh)
    if not thresh > 0.0:
        raise ValueError("thresh must be positive")
    peak = max(abs(v) for v in x)
    if peak <= 0.0:
        raise ValueError("ECG is identically zero")
    xn = [v / peak for v in x]
    n = len(xn)
    y0 = _rgpad([abs(xn[i] - xn[i - 2]) for i in range(2, n)], 2)
    y1 = _rgpad([abs(xn[i] - 2.0 * xn[i - 2] + xn[i - 4]) for i in range(4, n)], 4)
    y2 = [1.3 * a + 1.1 * b for a, b in zip(y0, y1)]
    y3 = _rgmavg(y2, 8)

    mask = [False] * n
    i = 0
    while i < n:
        if y2[i] > thresh:
            seg = y2[i:i + 8]
            if sum(1 for v in seg if v > thresh) >= 6:
                for j in range(i, min(n, i + 8)):
                    mask[j] = True
                i += 8
                continue
        i += 1

    # one beat position per contiguous accepted run: the maximum of y3 in it
    qrs = []
    i = 0
    delay = 4 + 4  # two-sample derivative span plus half the 8-point MA
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            best = max(range(i, j), key=lambda k: y3[k])
            qrs.append(max(0, best - delay))
            i = j
        else:
            i += 1
    return RichResult(payload={
        "qrs": qrs,
        "y0": y0,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "mask": mask,
        "thresh": thresh,
        "fs": fs,
        "hr": 60.0 * len(qrs) / (n / fs) if qrs else 0.0,
        "method": "derivative-based QRS detection, Rangayyan (2024) Section 4.3.1, Eqs 4.1-4.3 with the 8-point MA of Eq 3.108 (Balda et al.)",
    })


rangayyan_deriv_qrs = qrsderiv  # pre-policy spelling


# -- rgdnot: Dicrotic notch detection in carotid pulse waveform.
def dicnotch(cp, fs, qrs=None, mwin=16):
    """Dicrotic notch detection in the carotid pulse.

    The Lehner and Rangayyan method of Rangayyan (2024) Section 4.3.5: form
    the noncausal least-squares second derivative

        p(n) = 2 y(n-2) - y(n-1) - 2 y(n) - y(n+1) + 2 y(n+2)      (Eq 4.22)

    then square and smooth it with the linear weights of Eq 4.23.  The second
    derivative is used because the notch is a short wave riding on the falling
    limb of the pulse: a first derivative would return a near-constant value
    over that limb, whereas the second derivative cancels the slope and leaves
    the notch itself.

    s(n) has two peaks per cycle.  The first is the carotid upstroke onset;
    the second is the notch.  The notch is then placed at the local minimum of
    the carotid pulse within +/- 20 ms of that second peak.  When QRS
    positions are supplied, the search runs in the 500 ms interval after each
    QRS, as the book recommends.

    Parameters
    ----------
    cp : sequence of float
        Carotid pulse samples.
    fs : float
        Sampling rate in Hz.  The book used M = 16 at 256 Hz.
    qrs : sequence of int, optional
        QRS sample positions.  When given, one notch is reported per beat.
    mwin : int
        Smoothing window M of Eq 4.23.

    Returns
    -------
    RichResult
        payload["notch"] sample indices, payload["upstroke"] the first peak of
        each cycle, payload["s"] the detection function.
    """
    fs = _rgfs(fs)
    cp = _rgcheck(cp, 9, "carotid pulse")
    mwin = int(mwin)
    if mwin < 1:
        raise ValueError("mwin must be >= 1")
    n = len(cp)
    p = [0.0] * n
    for i in range(2, n - 2):
        p[i] = (2.0 * cp[i - 2] - cp[i - 1] - 2.0 * cp[i]
                - cp[i + 1] + 2.0 * cp[i + 2])
    s = [0.0] * n
    for i in range(n):
        s[i] = fsum(p[i - k + 1] ** 2 * (mwin - k + 1)
                    for k in range(1, mwin + 1) if 0 <= i - k + 1)

    tol = max(1, int(round(0.020 * fs)))
    guard = max(1, int(round(0.050 * fs)))

    def _localmin(centre):
        lo = max(0, centre - tol)
        hi = min(n, centre + tol + 1)
        return min(range(lo, hi), key=lambda k: cp[k])

    notch, upstroke = [], []
    if qrs is not None:
        span = int(round(0.500 * fs))
        for q in qrs:
            q = int(q)
            lo, hi = max(0, q), min(n, q + span)
            if hi - lo < 3 * guard:
                continue
            pk = _rgpeaks(s[lo:hi], 0.25 * max(s[lo:hi] or [0.0]), guard)
            if len(pk) < 2:
                continue
            upstroke.append(lo + pk[0])
            notch.append(_localmin(lo + pk[1]))
    else:
        thr = 0.25 * max(s)
        pk = _rgpeaks(s, thr, guard)
        for j, idx in enumerate(pk):
            if j % 2 == 0:
                upstroke.append(idx)
            else:
                notch.append(_localmin(idx))
    return RichResult(payload={
        "notch": notch,
        "upstroke": upstroke,
        "s": s,
        "p": p,
        "mwin": mwin,
        "fs": fs,
        "tolerancems": 20.0,
        "method": "dicrotic notch detection, Rangayyan (2024) Section 4.3.5, Eqs 4.22 and 4.23 (Lehner and Rangayyan)",
    })


rangayyan_dicrotic_notch = dicnotch  # pre-policy spelling


# -- rgecgemu: ECG-EMG coupling during physical effort (VMG correlation).
def ecgemgcpl(ecg, emg, qrs, fs):
    """Coupling between cardiac rate and muscle activity during physical effort.

    Two facts from Rangayyan (2024) are combined.  Section 2.2.6 states that
    the RMS and mean frequency of the EMG both increase with the level of
    muscle contraction until fatigue sets in, at which point both decrease,
    and that the VMG parameters parallel the EMG ones.  Section 2.2.5 states
    that heart rate is under autonomic control and rises with effort.  So the
    beat-by-beat instantaneous heart rate and the EMG activity measured over
    the same beat interval should move together during effort, and the
    correlation between them quantifies the coupling.

    RMS and mean frequency are computed per cardiac cycle so that the two
    series are sampled on the same beat grid and can be correlated directly.
    Mean frequency is the power-weighted centroid of the EMG spectrum, which
    is the definition the book's statement refers to.

    Parameters
    ----------
    ecg : sequence of float
        ECG samples (retained for the record; the rate comes from qrs).
    emg : sequence of float
        Simultaneously recorded EMG, same length and rate.
    qrs : sequence of int
        QRS sample positions.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    RichResult
        payload["hr"] instantaneous rate per beat, payload["rms"] and
        payload["meanfreq"] per beat, payload["rrms"] and payload["rmnf"] the
        correlations of each with heart rate.
    """
    fs = _rgfs(fs)
    ecg = _rgcheck(ecg, 16, "ECG")
    emg = _rgcheck(emg, 16, "EMG")
    if len(ecg) != len(emg):
        raise ValueError("ECG and EMG must have the same length and rate")
    q = [int(v) for v in aslist(qrs)]
    if len(q) < 3:
        raise ValueError("need at least three QRS positions")

    hr, rms, mnf = [], [], []
    for k in range(1, len(q)):
        a, b = q[k - 1], q[k]
        if b - a < 8:
            continue
        seg = emg[a:b]
        mu = fsum(seg) / len(seg)
        seg = [v - mu for v in seg]
        hr.append(60.0 * fs / (b - a))
        rms.append(sqrt(fsum(v * v for v in seg) / len(seg)))
        freqs, power = _rgpsd(seg, fs)
        tot = fsum(power)
        mnf.append((fsum(freqs[i] * power[i] for i in range(len(freqs))) / tot)
                   if tot > 0.0 else 0.0)
    if len(hr) < 2:
        raise ValueError("too few usable cardiac cycles for a coupling estimate")
    return RichResult(payload={
        "hr": hr,
        "rms": rms,
        "meanfreq": mnf,
        "rrms": _rgcorr(hr, rms),
        "rmnf": _rgcorr(hr, mnf),
        "nbeats": len(hr),
        "fs": fs,
        "method": "per-cycle EMG RMS and mean frequency correlated with instantaneous heart rate, Rangayyan (2024) Sections 2.2.5 and 2.2.6",
    })


rangayyan_ecg_emg_coupling = ecgemgcpl  # pre-policy spelling


# -- rgecgf: ECG waveform feature extraction (P, QRS, T amplitudes and durations).
def ecgfeat(x, qrs, fs):
    """P, QRS and T amplitudes and durations from a single-lead ECG.

    The waves and intervals are those summarised in Rangayyan (2024) Section
    1.2.4: the P wave of atrial depolarisation, the QRS of ventricular
    depolarisation, the ST segment which is normally isoelectric and in line
    with the PQ segment, and the T wave of ventricular repolarisation.

    Amplitudes are measured against the PQ segment rather than against zero,
    because the PQ segment is the isoelectric reference the book names and it
    absorbs any residual baseline offset.  Durations are taken between the
    points where the signal returns to within a small fraction of that
    reference, which is the operational definition a computer can apply.

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    qrs : sequence of int
        QRS (R peak) sample positions.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    RichResult
        Per-beat lists "pamp", "ramp", "qamp", "samp", "tamp" (amplitude
        relative to the PQ baseline) and "qrsdur", "pdur", "tdur", "prdur",
        "qtdur" in seconds, plus their means.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 16, "ECG")
    q = [int(v) for v in aslist(qrs)]
    if not q:
        raise ValueError("need at least one QRS position")
    n = len(x)
    pqa = max(1, int(round(0.080 * fs)))
    pqb = max(1, int(round(0.040 * fs)))
    half = max(1, int(round(0.060 * fs)))
    pwin = max(2, int(round(0.200 * fs)))
    twinlo = max(1, int(round(0.100 * fs)))
    twinhi = max(2, int(round(0.450 * fs)))

    pamp, ramp, qamp, samp, tamp = [], [], [], [], []
    qrsdur, pdur, tdur, prdur, qtdur = [], [], [], [], []

    for pos in q:
        a = max(0, pos - pqa)
        b = max(a + 1, pos - pqb)
        ref = fsum(x[a:b]) / (b - a)

        lo = max(0, pos - half)
        hi = min(n, pos + half)
        if hi - lo < 3:
            continue
        r = max(range(lo, hi), key=lambda i: x[i] - ref)
        qi = min(range(lo, r + 1), key=lambda i: x[i] - ref) if r > lo else r
        si = min(range(r, hi), key=lambda i: x[i] - ref) if hi > r + 1 else r
        ramp.append(x[r] - ref)
        qamp.append(x[qi] - ref)
        samp.append(x[si] - ref)

        amp = max(abs(x[r] - ref), 1e-12)
        tol = 0.05 * amp
        qs = qi
        while qs > lo and abs(x[qs] - ref) > tol:
            qs -= 1
        qe = si
        while qe < hi - 1 and abs(x[qe] - ref) > tol:
            qe += 1
        qrsdur.append((qe - qs) / fs)

        pa, pb = max(0, qs - pwin), qs
        if pb - pa >= 2:
            pi = max(range(pa, pb), key=lambda i: abs(x[i] - ref))
            pamp.append(x[pi] - ref)
            pmag = max(abs(x[pi] - ref), 1e-12)
            ps, pe = pi, pi
            while ps > pa and abs(x[ps] - ref) > 0.1 * pmag:
                ps -= 1
            while pe < pb - 1 and abs(x[pe] - ref) > 0.1 * pmag:
                pe += 1
            pdur.append((pe - ps) / fs)
            prdur.append((qs - ps) / fs)

        ta, tb = min(n - 1, pos + twinlo), min(n, pos + twinhi)
        if tb - ta >= 2:
            ti = max(range(ta, tb), key=lambda i: abs(x[i] - ref))
            tamp.append(x[ti] - ref)
            tmag = max(abs(x[ti] - ref), 1e-12)
            ts, te = ti, ti
            while ts > ta and abs(x[ts] - ref) > 0.1 * tmag:
                ts -= 1
            while te < tb - 1 and abs(x[te] - ref) > 0.1 * tmag:
                te += 1
            tdur.append((te - ts) / fs)
            qtdur.append((te - qs) / fs)

    mean = lambda v: (fsum(v) / len(v)) if v else None
    return RichResult(payload={
        "pamp": pamp, "qamp": qamp, "ramp": ramp, "samp": samp, "tamp": tamp,
        "qrsdur": qrsdur, "pdur": pdur, "tdur": tdur,
        "prdur": prdur, "qtdur": qtdur,
        "qrsdurmean": mean(qrsdur), "prdurmean": mean(prdur),
        "qtdurmean": mean(qtdur), "rampmean": mean(ramp),
        "nbeats": len(q),
        "fs": fs,
        "method": "ECG wave amplitudes and durations against the PQ isoelectric reference, Rangayyan (2024) Section 1.2.4",
    })


rangayyan_ecg_features = ecgfeat  # pre-policy spelling


# -- rgecgwvf: ECG waveform analysis for ischemia and bundle branch block.
def ecgwaveshp(qrsdur, stdev, rdur=None, sdur=None, qpresent=None):
    """ECG waveshape assessment for ischemia and bundle-branch block.

    Two findings from Rangayyan (2024) are applied.

    Ischemia and infarction (Section 1.2.4): the ST segment is normally
    isoelectric and in line with the PQ segment; it may be elevated or
    depressed by myocardial ischemia (reduced coronary supply) or by
    myocardial infarction (dead, non-contracting tissue).  ST deviation is
    therefore reported against the PQ reference, not against zero.

    Bundle-branch block (Section 10.2.1): a block delays activation of one
    ventricle relative to the other, so contraction becomes asynchronous and
    the QRS becomes wider than normal, 100 to 120 ms or more, and may be
    jagged or slurred.  The book's decision rules for the incomplete forms
    are applied where the required measurements are supplied:

      incomplete LBBB requires QRS duration >= 105 ms and <= 120 ms, negative
      QRS in V1 and V2, Q or S duration >= 80 ms in V1 and V2, no Q wave in
      any two of I, V5, V6, and R duration > 60 ms in any two of I, aVL, V5,
      V6;

      incomplete RBBB requires QRS duration >= 91 ms and <= 120 ms and
      S duration >= 40 ms in any two of I, aVL, V4, V5, V6, together with the
      R or R-prime conditions in V1 or V2.

    This function evaluates the duration-based parts of those rules, which are
    the parts computable from a single measured QRS duration; the lead-specific
    polarity and Q/R/S conditions are returned as required checks the caller
    must supply from a 12-lead measurement, not silently assumed.

    Parameters
    ----------
    qrsdur : float
        QRS duration in seconds.
    stdev : float
        ST-segment deviation in mV relative to the PQ segment; positive is
        elevation, negative is depression.
    rdur, sdur : float, optional
        R and S wave durations in seconds, for the lead-specific rules.
    qpresent : bool, optional
        Whether a Q wave is present in the lead being assessed.

    Returns
    -------
    RichResult
        payload["qrswide"], payload["lbbbdur"], payload["rbbbdur"],
        payload["stfinding"] and payload["required"], the further 12-lead
        conditions the book's rules still need.
    """
    qrsdur = float(qrsdur)
    stdev = float(stdev)
    if not qrsdur > 0.0:
        raise ValueError("qrsdur must be positive")
    ms = qrsdur * 1000.0
    st = "isoelectric"
    if stdev >= 0.1:
        st = "elevated"
    elif stdev <= -0.1:
        st = "depressed"
    req = []
    if 105.0 <= ms <= 120.0:
        req.append("LBBB also needs: negative QRS in V1,V2; Q or S >= 80 ms in V1,V2; no Q in any two of I,V5,V6; R > 60 ms in any two of I,aVL,V5,V6")
    if 91.0 <= ms <= 120.0:
        req.append("RBBB also needs: S >= 40 ms in any two of I,aVL,V4,V5,V6; and in V1 or V2 either R (or R') > 30 ms with amplitude > 100 uV and no S (or S')")
    return RichResult(payload={
        "qrsdurms": ms,
        "qrswide": ms > 120.0,
        "lbbbdur": 105.0 <= ms <= 120.0,
        "rbbbdur": 91.0 <= ms <= 120.0,
        "sdurok": (None if sdur is None else float(sdur) * 1000.0 >= 40.0),
        "rdurok": (None if rdur is None else float(rdur) * 1000.0 > 60.0),
        "qabsent": (None if qpresent is None else not bool(qpresent)),
        "stdev": stdev,
        "stfinding": st,
        "required": req,
        "method": "ECG waveshape rules for ischemia and bundle-branch block, Rangayyan (2024) Sections 1.2.4 and 10.2.1",
    })


rangayyan_ecg_waveshape = ecgwaveshp  # pre-policy spelling


# -- rgexecg: Exercise ECG analysis: ST deviation, slope, and ischemia detection.
def exerecgst(x, qrs, fs, jofs=0.060, thresh=0.1):
    """Exercise ECG analysis: ST deviation and ST slope per beat.

    Rangayyan (2024) Section 1.2.4 states that the ST segment lies about 100
    to 120 ms after the QRS, is normally isoelectric and in line with the PQ
    segment, and may be elevated or depressed by ischemia or infarction.  That
    is the measurement this function makes: the level at J + jofs relative to
    the PQ isoelectric level, and the slope of the segment that follows.

    The slope matters because upsloping and horizontal or downsloping ST
    depression carry different weight in exercise testing; both are reported
    so the caller can apply whatever criterion is in use.

    Threshold note: Rangayyan (2024) gives no numeric exercise-ECG criterion
    for a positive test.  The default thresh = 0.1 mV is the conventional
    clinical figure and is a parameter, not a book value.  It was not verified
    against a primary source in this session and the payload says so.

    Parameters
    ----------
    x : sequence of float
        ECG samples in mV.
    qrs : sequence of int
        QRS sample positions.
    fs : float
        Sampling rate in Hz.
    jofs : float
        Offset after the J point at which ST level is measured, in seconds.
    thresh : float
        ST deviation magnitude, in mV, above which a beat is flagged.

    Returns
    -------
    RichResult
        payload["stdev"] per-beat deviation in mV, payload["stslope"] in mV/s,
        payload["pattern"] one of "upsloping", "horizontal", "downsloping",
        and payload["flagged"] the count exceeding thresh.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 16, "ECG")
    q = [int(v) for v in aslist(qrs)]
    if not q:
        raise ValueError("need at least one QRS position")
    jofs = float(jofs)
    if not jofs > 0.0:
        raise ValueError("jofs must be positive")
    n = len(x)
    pqa = max(1, int(round(0.080 * fs)))
    pqb = max(1, int(round(0.040 * fs)))
    jpt = max(1, int(round(0.040 * fs)))
    off = max(1, int(round(jofs * fs)))
    span = max(2, int(round(0.040 * fs)))

    dev, slope, pattern = [], [], []
    for pos in q:
        a = max(0, pos - pqa)
        b = max(a + 1, pos - pqb)
        ref = fsum(x[a:b]) / (b - a)
        m = pos + jpt + off
        if m + span >= n:
            continue
        lvl = x[m] - ref
        sl = (x[m + span] - x[m]) * fs / span
        dev.append(lvl)
        slope.append(sl)
        if sl > 0.05:
            pattern.append("upsloping")
        elif sl < -0.05:
            pattern.append("downsloping")
        else:
            pattern.append("horizontal")
    if not dev:
        raise ValueError("no beat had enough samples after the J point for an ST measurement")
    mean = fsum(dev) / len(dev)
    return RichResult(payload={
        "stdev": dev,
        "stslope": slope,
        "pattern": pattern,
        "stdevmean": mean,
        "stslopemean": fsum(slope) / len(slope),
        "flagged": sum(1 for v in dev if abs(v) >= thresh),
        "thresh": thresh,
        "threshnote": "0.1 mV is a conventional clinical figure, not from Rangayyan (2024); no primary source verified here",
        "jofs": jofs,
        "fs": fs,
        "method": "ST level and slope against the PQ isoelectric reference, Rangayyan (2024) Section 1.2.4",
    })


rangayyan_exercise_ecg = exerecgst  # pre-policy spelling


# -- rghrvf: HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio.
def hrvfreq(rr, fsr=4.0, bands="taskforce"):
    """Frequency-domain HRV: VLF, LF and HF power and the LF/HF ratio.

    Rangayyan (2024) Section 8.12 gives two band definitions.  Bianchi et al.,
    whose time-variant analysis the section describes, used VLF 0 to 0.03 Hz
    (humoral and thermoregulatory), LF 0.03 to 0.15 Hz (sympathetic) and HF
    0.18 to 0.4 Hz (respiration and vagal).  The book then notes that the
    Task Force standards give slightly different bands: VLF <= 0.04 Hz,
    LF 0.04 to 0.15 Hz, HF 0.15 to 0.4 Hz.  Both are selectable here; the
    Task Force set is the default because it is the one in general use.

    The RR series is a series of events, not a uniformly sampled signal, so it
    is first resampled onto a uniform grid by linear interpolation of the
    tachogram before the periodogram is taken.  The mean is removed first so
    that the DC term does not leak into VLF.

    Parameters
    ----------
    rr : sequence of float
        RR intervals in seconds.
    fsr : float
        Resampling rate in Hz for the interpolated tachogram.
    bands : str
        "taskforce" or "bianchi".

    Returns
    -------
    RichResult
        payload["vlf"], ["lf"], ["hf"] absolute powers in s^2, their
        percentages of total power, and payload["lfhf"], the ratio.
    """
    rr = _rgcheck(rr, 8, "RR series")
    if any(v <= 0.0 for v in rr):
        raise ValueError("RR intervals must be positive")
    fsr = _rgfs(fsr)
    if bands == "taskforce":
        lim = {"vlf": (0.0, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.4)}
    elif bands == "bianchi":
        lim = {"vlf": (0.0, 0.03), "lf": (0.03, 0.15), "hf": (0.18, 0.4)}
    else:
        raise ValueError("bands must be 'taskforce' or 'bianchi'")

    # event times are the cumulative RR sums; interpolate onto a uniform grid
    t = [0.0]
    for v in rr:
        t.append(t[-1] + v)
    tt = t[1:]
    total = tt[-1]
    m = int(total * fsr)
    if m < 8:
        raise ValueError("RR series too short for spectral analysis at this fsr")
    grid = []
    j = 0
    for k in range(m):
        tk = k / fsr
        while j < len(tt) - 1 and tt[j + 1] < tk:
            j += 1
        if j >= len(tt) - 1:
            grid.append(rr[-1])
            continue
        t0, t1 = tt[j], tt[j + 1]
        w = 0.0 if t1 <= t0 else (tk - t0) / (t1 - t0)
        w = min(1.0, max(0.0, w))
        grid.append(rr[j] * (1.0 - w) + rr[j + 1] * w)
    mu = fsum(grid) / len(grid)
    grid = [v - mu for v in grid]

    freqs, power = _rgpsd(grid, fsr)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

    def _band(lo, hi):
        return fsum(power[k] * df for k in range(len(freqs)) if lo < freqs[k] <= hi)

    vlf = _band(*lim["vlf"])
    lf = _band(*lim["lf"])
    hf = _band(*lim["hf"])
    tot = vlf + lf + hf
    pct = lambda v: (100.0 * v / tot) if tot > 0.0 else 0.0
    return RichResult(payload={
        "vlf": vlf, "lf": lf, "hf": hf,
        "total": tot,
        "vlfpct": pct(vlf), "lfpct": pct(lf), "hfpct": pct(hf),
        "lfhf": (lf / hf) if hf > 0.0 else None,
        "bands": bands,
        "limits": lim,
        "fsr": fsr,
        "n": len(rr),
        "method": "frequency-domain HRV bands, Rangayyan (2024) Section 8.12 (Bianchi et al. bands, and the Task Force bands quoted there)",
    })


rangayyan_hrv_freq_domain = hrvfreq  # pre-policy spelling


# -- rghrvt: HRV time-domain metrics: SDNN, RMSSD, pNN50.
def hrvtime(rr):
    """Time-domain heart rate variability: SDNN, RMSSD and pNN50.

    Rangayyan (2024) Section 2.2.5 establishes why these are measured: beat-to
    beat variability is normal and healthy, reflects the balance between
    sympathetic and parasympathetic activity, and reduced variability after
    myocardial infarction predicts poor outcome.  The book does not, however,
    define SDNN, RMSSD or pNN50.  The definitions used here are those of the
    Task Force of the European Society of Cardiology and the North American
    Society of Pacing Electrophysiology, Heart rate variability: Standards of
    measurement, physiological interpretation, and clinical use, Circulation
    93(5):1043-1065, 1996 -- reference [84] of the book's Chapter 8.

    SDNN is the standard deviation of the NN intervals and captures total
    variability over the recording.  RMSSD is the root mean square of the
    successive differences and is dominated by short-term, vagally mediated
    changes.  pNN50 is the percentage of successive differences exceeding
    50 ms and measures the same short-term component with a hard cutoff.

    Parameters
    ----------
    rr : sequence of float
        NN (normal-to-normal) intervals in seconds.  At least two are needed
        for SDNN and RMSSD.

    Returns
    -------
    RichResult
        payload["sdnn"], ["rmssd"], ["pnn50"] with intervals in ms, plus mean
        NN, mean heart rate, and the count used.
    """
    rr = _rgcheck(rr, 2, "RR series")
    if any(v <= 0.0 for v in rr):
        raise ValueError("RR intervals must be positive")
    ms = [v * 1000.0 for v in rr]
    n = len(ms)
    mean = fsum(ms) / n
    sdnn = sqrt(fsum((v - mean) ** 2 for v in ms) / (n - 1))
    d = [ms[i] - ms[i - 1] for i in range(1, n)]
    rmssd = sqrt(fsum(v * v for v in d) / len(d))
    nn50 = sum(1 for v in d if abs(v) > 50.0)
    return RichResult(payload={
        "sdnn": sdnn,
        "rmssd": rmssd,
        "nn50": nn50,
        "pnn50": 100.0 * nn50 / len(d),
        "meannn": mean,
        "meanhr": 60000.0 / mean,
        "n": n,
        "units": "ms",
        "method": "time-domain HRV (SDNN, RMSSD, pNN50), Task Force of the ESC and NASPE, Circulation 93(5):1043-1065, 1996; Rangayyan (2024) Section 2.2.5 motivates but does not define these",
    })


rangayyan_hrv_time_domain = hrvtime  # pre-policy spelling


# -- rghsnd: Heart sound (S1/S2) identification via PCG-ECG timing.
def hsoundid(ecg, cp, fs):
    """Identify S1 and S2 in a PCG by transferring ECG and carotid pulse timing.

    Rangayyan (2024) Section 4.9.  S1 onset is taken at the onset of the QRS,
    detected here with the Pan-Tompkins method of Section 4.3.2.  S2 onset is
    harder: it is derived from the dicrotic notch of the carotid pulse, found
    with the Lehner and Rangayyan method of Section 4.3.5 within 500 ms of the
    QRS, minus the standardised S2-to-notch delay.

    Lehner and Rangayyan measured that delay over 60 pediatric subjects as
    42.6 +/- 5 ms; the book standardises it as mean + 2 SD = 52.6 ms, which is
    the value used here.  The delay exists because the notch is the same
    aortic-valve closure event observed after propagation up the arterial
    tree, so it necessarily lags the sound.

    The book's supporting arithmetic: with PEPCmax = 144 ms and HRmin = 60 bpm
    PEPmax = 120 ms, and with ETCmax = 425 ms ETmax = 329 ms, so the notch can
    be at most about 380 ms after the QRS, which is why a 500 ms search window
    suffices.

    Parameters
    ----------
    ecg : sequence of float
        ECG samples.
    cp : sequence of float
        Simultaneously recorded carotid pulse, same length and rate.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    RichResult
        payload["s1"] and payload["s2"] onset sample indices, payload["notch"],
        payload["qrs"], and the delay actually subtracted.
    """
    fs = _rgfs(fs)
    ecg = _rgcheck(ecg, 40, "ECG")
    cp = _rgcheck(cp, 40, "carotid pulse")
    if len(ecg) != len(cp):
        raise ValueError("ECG and carotid pulse must have the same length")
    q = list(qrsdetect(ecg, fs)["qrs"])
    if not q:
        raise ValueError("no QRS complexes detected; cannot place S1 or S2")
    notch = list(dicnotch(cp, fs, qrs=q)["notch"])
    lag = int(round(0.0526 * fs))
    s2 = [max(0, d - lag) for d in notch]
    return RichResult(payload={
        "s1": q,
        "s2": s2,
        "notch": notch,
        "qrs": q,
        "s2delayms": 52.6,
        "s2delaymeasured": (42.6, 5.0),
        "searchwindowms": 500.0,
        "fs": fs,
        "method": "S1/S2 identification from ECG and carotid pulse timing, Rangayyan (2024) Section 4.9 (Lehner and Rangayyan)",
    })


rangayyan_heart_sound_id = hsoundid  # pre-policy spelling


# -- rgmatefp: Maternal ECG filtering from abdominal ECG recording.
def mecgfilt(abd, thor, order=16, mu=0.01):
    """Cancel the maternal ECG from an abdominal recording by adaptive filtering.

    Rangayyan (2024) Sections 3.3.5 and 9.7.2 describe extraction of the fetal
    ECG by adaptive filtering: the abdominal lead carries the fetal ECG buried
    under a much larger maternal complex, and a thoracic lead carries the
    maternal ECG essentially alone.  The thoracic signal is used as the
    reference input to an adaptive noise canceller whose primary input is the
    abdominal signal; the filter learns the transfer path from thorax to
    abdomen, so its output estimates the maternal contribution at the
    abdominal electrode and the residual error is the fetal ECG.

    The assumption the method rests on is that the reference is correlated
    with the interference and uncorrelated with the signal of interest, which
    holds here because the fetal complexes are not present in the thoracic
    lead.

    The weights adapt by the normalised least-mean-squares rule.  Adaptation
    is deterministic: weights start at zero and there is no random
    initialisation.

    Parameters
    ----------
    abd : sequence of float
        Abdominal ECG (primary input).
    thor : sequence of float
        Thoracic maternal ECG (reference input), same length and rate.
    order : int
        Number of filter taps.  Choose it to span the thorax-to-abdomen path
        and no more: a filter much longer than that path starts fitting the
        fetal complexes with the maternal reference and cancels the very
        signal it is meant to preserve.  About 80 ms of samples works well.
    mu : float
        Adaptation step size, 0 < mu < 2 for stability of the normalised rule.

    Returns
    -------
    RichResult
        payload["fetal"] the error signal (the extracted fetal ECG),
        payload["maternal"] the cancelled maternal estimate, payload["weights"]
        the final tap weights.
    """
    abd = _rgcheck(abd, 4, "abdominal ECG")
    thor = _rgcheck(thor, 4, "thoracic ECG")
    if len(abd) != len(thor):
        raise ValueError("abdominal and thoracic signals must have the same length")
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    if len(abd) <= order:
        raise ValueError("signals must be longer than the filter order")
    mu = float(mu)
    if not 0.0 < mu < 2.0:
        raise ValueError("mu must lie in (0, 2)")

    n = len(abd)
    # Regularise the normalised step by a fraction of the reference's own mean
    # power.  A fixed tiny constant is not safe here: the thoracic lead is near
    # zero between maternal complexes, and mu divided by an almost-zero tap
    # power makes the weights diverge on exactly the quiet stretches where the
    # fetal ECG is the only thing present.
    eps = 1e-3 * order * (fsum(v * v for v in thor) / n)
    if eps <= 0.0:
        raise ValueError("thoracic reference is identically zero")
    w = [0.0] * order
    est = [0.0] * n
    err = [0.0] * n
    for i in range(n):
        xv = [thor[i - k] if i - k >= 0 else 0.0 for k in range(order)]
        yv = fsum(w[k] * xv[k] for k in range(order))
        e = abd[i] - yv
        est[i] = yv
        err[i] = e
        p = fsum(v * v for v in xv)
        g = mu / (p + eps)
        for k in range(order):
            w[k] += g * e * xv[k]
    return RichResult(payload={
        "fetal": err,
        "maternal": est,
        "weights": w,
        "order": order,
        "mu": mu,
        "n": n,
        "method": "adaptive cancellation of the maternal ECG with a thoracic reference, Rangayyan (2024) Sections 3.3.5 and 9.7.2 (normalised LMS)",
    })


rangayyan_maternal_ecg_filter = mecgfilt  # pre-policy spelling


# -- rgmtnart: Motion artifact detection and removal from ECG/PPG.
def motionart(x, fs, win=1.0, factor=4.0):
    """Detect and blank motion artifact in an ECG or PPG record.

    Rangayyan (2024) Section 1.2.11 states the problem for the PPG directly:
    any movement, including physical activity, produces motion artifacts that
    corrupt the signal and reduce the precision of the parameters derived from
    it, and pressure or contact-force changes deform the underlying arteries
    and alter the AC component.  The book characterises the artifact but gives
    no detection equation, so none is attributed to it.

    The detector implemented here uses the two properties that distinguish
    artifact from cardiac signal in those descriptions: an artifact segment has
    both an amplitude range and a sample-to-sample activity far above the
    record's own typical level.  Both are compared against a median-based
    baseline rather than a mean, because the mean is dragged upward by the
    very segments being detected.  Flagged segments are replaced by linear
    interpolation across the gap, which keeps the sample grid intact for any
    downstream filter.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    win : float
        Analysis window length in seconds.
    factor : float
        Multiple of the median window statistic above which a window is
        flagged as artifact.

    Returns
    -------
    RichResult
        payload["clean"] the repaired signal, payload["artifact"] as
        (start, end) sample pairs, payload["fraction"] the proportion of the
        record flagged.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 16, "signal")
    win = float(win)
    factor = float(factor)
    if not win > 0.0:
        raise ValueError("win must be positive")
    if not factor > 1.0:
        raise ValueError("factor must be greater than 1")
    w = max(4, int(round(win * fs)))
    n = len(x)
    starts = list(range(0, n, w))
    rng, act = [], []
    for a in starts:
        seg = x[a:min(n, a + w)]
        rng.append(max(seg) - min(seg))
        act.append(fsum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg))) / max(1, len(seg) - 1))

    def _median(v):
        s = sorted(v)
        m = len(s)
        return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])

    mr, ma = _median(rng), _median(act)
    bad = [(rng[i] > factor * mr and mr > 0.0) or (act[i] > factor * ma and ma > 0.0)
           for i in range(len(starts))]

    segs = []
    i = 0
    while i < len(starts):
        if bad[i]:
            j = i
            while j < len(starts) and bad[j]:
                j += 1
            segs.append((starts[i], min(n, starts[j - 1] + w)))
            i = j
        else:
            i += 1

    clean = list(x)
    for a, b in segs:
        lv = x[a - 1] if a > 0 else (x[b] if b < n else 0.0)
        rv = x[b] if b < n else lv
        span = max(1, b - a)
        for k in range(a, min(n, b)):
            t = (k - a + 1) / float(span + 1)
            clean[k] = lv * (1.0 - t) + rv * t
    flagged = fsum(min(n, b) - a for a, b in segs)
    return RichResult(payload={
        "clean": clean,
        "artifact": segs,
        "nsegments": len(segs),
        "fraction": flagged / float(n),
        "win": win,
        "factor": factor,
        "fs": fs,
        "method": "motion-artifact detection by window range and activity against their medians; Rangayyan (2024) Section 1.2.11 characterises the artifact but gives no detection equation",
    })


rangayyan_motion_artifact = motionart  # pre-policy spelling


# -- rgpantp: Pan-Tompkins QRS detection algorithm.
def qrsdetect(x, fs=200.0):
    """Real-time QRS detection by the Pan-Tompkins algorithm.

    The full chain of Rangayyan (2024) Section 4.3.2: lowpass (Eq 4.8),
    highpass (Eq 4.13), five-point derivative (Eq 4.14), squaring, and
    moving-window integration (Eq 4.15), followed by the adaptive thresholds
    of Eqs 4.16 to 4.18 and the search-back procedure.

    Each stage exists for a reason.  The bandpass keeps roughly 5 to 11 Hz,
    which is where QRS energy is concentrated, and rejects P and T waves,
    baseline drift, and mains interference.  The derivative rewards the steep
    QRS slopes.  Squaring makes everything positive and widens the gap between
    QRS and residual P/T energy.  The integrator merges the several derivative
    peaks inside one complex into a single pulse.  Adaptive thresholds let the
    detector follow slow amplitude changes, and search-back recovers a beat
    whenever no QRS has been found for 1.66 times the recent average RR.

    The book's filter coefficients are integers designed for fs = 200 Hz and
    do not rescale.  fs is taken explicitly and is reported in the payload;
    the timing constants (150 ms integrator, 200 ms refractory period) are
    scaled by fs, the coefficients are not.

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    fs : float
        Sampling rate in Hz.  Defaults to the book's 200 Hz.

    Returns
    -------
    RichResult
        payload["qrs"] sample indices of detected QRS complexes (already
        corrected for the cumulative filter delay), payload["rr"] the RR
        intervals in seconds, payload["hr"] the average rate in bpm, plus the
        integrator output and the final threshold state.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 40, "ECG")
    bp, dv, sq, ig, wint = _rgchain(x, fs)
    n = len(ig)

    # cumulative group delay: 5 samples (Eq 4.8 at 200 Hz), 16 samples for the
    # allpass branch of Eq 4.13, 2 samples for the derivative, and half the
    # integrator window.
    delay = int(round(5.0 * fs / 200.0)) + int(round(16.0 * fs / 200.0)) + 2 + wint // 2
    refrac = max(1, int(round(0.200 * fs)))

    # local maxima of the integrator output are the candidate PEAKI values
    cand = [i for i in range(1, n - 1) if ig[i] > ig[i - 1] and ig[i] >= ig[i + 1]]

    learn = min(n, max(2 * wint, int(round(2.0 * fs))))
    seg = ig[:learn]
    spki = max(seg) / 3.0
    npki = (fsum(seg) / len(seg)) / 2.0
    t1 = npki + 0.25 * (spki - npki)
    t2 = 0.5 * t1

    qrs = []
    peaks = []
    rr1 = []
    rr2 = []
    rrave2 = None
    searchback = 0

    def _rrlimits():
        if rrave2 is None:
            return None, None, None
        return 0.92 * rrave2, 1.16 * rrave2, 1.66 * rrave2

    for i in cand:
        pk = ig[i]
        if qrs and (i - qrs[-1]) < refrac:
            if pk > peaks[-1]:
                qrs[-1] = i
                peaks[-1] = pk
            continue

        lo, hi, missed = _rrlimits()
        # search-back: no QRS for longer than RR MISSED LIMIT
        if qrs and missed is not None and (i - qrs[-1]) / fs > missed:
            window = [j for j in cand if qrs[-1] + refrac <= j < i and ig[j] > t2]
            if window:
                best = max(window, key=lambda j: ig[j])
                qrs.append(best)
                peaks.append(ig[best])
                spki = 0.25 * ig[best] + 0.75 * spki          # Eq 4.18
                searchback += 1
                t1 = npki + 0.25 * (spki - npki)
                t2 = 0.5 * t1

        if pk > t1:
            qrs.append(i)
            peaks.append(pk)
            spki = 0.125 * pk + 0.875 * spki                   # Eq 4.16
            if len(qrs) > 1:
                rr = (qrs[-1] - qrs[-2]) / fs
                rr1.append(rr)
                if len(rr1) > 8:
                    rr1.pop(0)
                if lo is None or (lo <= rr <= hi):
                    rr2.append(rr)
                    if len(rr2) > 8:
                        rr2.pop(0)
                if rr2:
                    rrave2 = fsum(rr2) / len(rr2)
                elif rr1:
                    rrave2 = fsum(rr1) / len(rr1)
        else:
            npki = 0.125 * pk + 0.875 * npki                   # Eq 4.16
        t1 = npki + 0.25 * (spki - npki)                       # Eq 4.17
        t2 = 0.5 * t1

    loc = sorted(set(max(0, i - delay) for i in qrs))
    rr = [(loc[i] - loc[i - 1]) / fs for i in range(1, len(loc))]
    hr = 60.0 * len(loc) / (len(x) / fs) if loc else 0.0
    return RichResult(payload={
        "qrs": loc,
        "rr": rr,
        "hr": hr,
        "integrated": ig,
        "bandpass": bp,
        "delay": delay,
        "spki": spki,
        "npki": npki,
        "thresh1": t1,
        "thresh2": t2,
        "searchback": searchback,
        "fs": fs,
        "fsnote": "filter coefficients are integers fixed for fs = 200 Hz; timing constants scale with fs",
        "method": "Pan-Tompkins QRS detection, Rangayyan (2024) Section 4.3.2, Eqs 4.8-4.18; Pan and Tompkins, IEEE TBME 32(3):230-236, 1985",
    })


rangayyan_pan_tompkins = qrsdetect  # pre-policy spelling


# -- rgpcg: PCG segmentation into S1/systole/S2/diastole using ECG gating.
def pcgparts(pcg, ecg, cp, fs):
    """Segment a PCG into S1, systole, S2 and diastole using ECG and carotid gating.

    The six-step procedure of Rangayyan (2024) Section 4.9:

    1. locate the QRS complexes in the ECG with the Pan-Tompkins method;
    2. take one PCG period as the interval between successive QRS positions,
       after subtracting the filter delay;
    3. detect the dicrotic notch in the carotid pulse (Section 4.3.5);
    4. subtract the standardised S2-D delay of 52.6 ms from each notch to get
       the onset of S2;
    5. the S1-to-S2 interval is the systolic part;
    6. the S2-to-next-S1 interval is the diastolic part.

    Gating on two other signals rather than on the PCG itself is the point of
    the method: S1 and S2 are not reliably the loudest events in a PCG once
    murmurs are present, so timing is imported from signals in which the
    corresponding events are unambiguous.

    Parameters
    ----------
    pcg, ecg, cp : sequence of float
        Simultaneously recorded PCG, ECG and carotid pulse, equal lengths.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    RichResult
        payload["systole"] and payload["diastole"] as lists of (start, end)
        sample index pairs, payload["s1"], payload["s2"], and per-segment RMS
        amplitudes of the PCG.
    """
    fs = _rgfs(fs)
    pcg = _rgcheck(pcg, 40, "PCG")
    ecg = _rgcheck(ecg, 40, "ECG")
    cp = _rgcheck(cp, 40, "carotid pulse")
    if not (len(pcg) == len(ecg) == len(cp)):
        raise ValueError("PCG, ECG and carotid pulse must have the same length")
    ids = hsoundid(ecg, cp, fs)
    s1 = list(ids["s1"])
    s2 = list(ids["s2"])
    if not s1 or not s2:
        raise ValueError("could not locate both S1 and S2 events")

    rms = lambda a, b: sqrt(fsum(v * v for v in pcg[a:b]) / (b - a)) if b > a else 0.0
    systole, diastole = [], []
    for a in s1:
        later = [v for v in s2 if v > a]
        if not later:
            continue
        b = later[0]
        systole.append((a, b))
        nxt = [v for v in s1 if v > b]
        if nxt:
            diastole.append((b, nxt[0]))
    return RichResult(payload={
        "s1": s1,
        "s2": s2,
        "systole": systole,
        "diastole": diastole,
        "systolerms": [rms(a, b) for a, b in systole],
        "diastolerms": [rms(a, b) for a, b in diastole],
        "fs": fs,
        "method": "PCG segmentation into systole and diastole, Rangayyan (2024) Section 4.9",
    })


rangayyan_pcg_segments = pcgparts  # pre-policy spelling


# -- rgpowerl: Powerline interference (50/60 Hz) removal from ECG.
def plinenotch(x, fs, f0=60.0, harmonics=1):
    """Remove power-line interference with a comb of unit-circle notch filters.

    Rangayyan (2024) eq. (3.150) (the section number 3.5.4 does not appear in the book; the equation does) and Equation 3.150.  For an interference
    frequency fo the required zeros sit on the unit circle at angles
    +/- (fo/fs) 2 pi, giving

        H(z) = (1 - z1 z^-1)(1 - z2 z^-1) = 1 - 2 cos(wo) z^-1 + z^-2,

    which for the book's worked case of fo = 60 Hz and fs = 1000 Hz is
    1 - 1.85955 z^-1 + z^-2.  The DC gain is H(1) = 2 - 2 cos(wo), which for
    that case is 0.14045; the filter is divided by it so the passband gain at
    DC is unity, exactly as the book prescribes.  Placing extra zeros at
    n fo / fs makes it a comb that also removes the harmonics that appear when
    the mains waveform is not a pure sinusoid.

    The book notes two consequences worth knowing: the notch attenuates a band
    around fo, not only fo itself, and the gain is largest at fs/2, so
    additional lowpass filtering is advisable for ECG.

    Parameters
    ----------
    x : sequence of float
        Signal samples.
    fs : float
        Sampling rate in Hz.
    f0 : float
        Interference frequency, 50 or 60 Hz.
    harmonics : int
        Number of harmonics to notch, including the fundamental.

    Returns
    -------
    RichResult
        payload["y"] the filtered signal, payload["coeffs"] the (b0, b1, b2)
        triple of each stage after DC normalisation, payload["notched"] the
        frequencies actually removed.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 3, "signal")
    f0 = float(f0)
    harmonics = int(harmonics)
    if not f0 > 0.0:
        raise ValueError("f0 must be positive")
    if harmonics < 1:
        raise ValueError("harmonics must be >= 1")
    y = list(x)
    coeffs, notched = [], []
    for h in range(1, harmonics + 1):
        f = h * f0
        if f >= fs / 2.0:
            break
        w = 2.0 * pi * f / fs
        b1 = -2.0 * cos(w)
        gain = 1.0 + b1 + 1.0
        if abs(gain) < 1e-12:
            raise ValueError("notch at %g Hz has zero DC gain; choose another frequency" % f)
        b = (1.0 / gain, b1 / gain, 1.0 / gain)
        coeffs.append(b)
        notched.append(f)
        z = [0.0] * len(y)
        for i in range(len(y)):
            z[i] = (b[0] * y[i]
                    + b[1] * (y[i - 1] if i >= 1 else 0.0)
                    + b[2] * (y[i - 2] if i >= 2 else 0.0))
        y = z
    if not notched:
        raise ValueError("f0 is at or above the Nyquist frequency; nothing to notch")
    return RichResult(payload={
        "y": y,
        "coeffs": coeffs,
        "notched": notched,
        "f0": f0,
        "fs": fs,
        "n": len(y),
        "method": "unit-circle comb notch filter, Rangayyan (2024) Eq 3.150",
    })


rangayyan_powerline_removal = plinenotch  # pre-policy spelling


# -- rgppg: PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak).
def ppgfeat(ppg, fs, mwin=16):
    """PPG waveform features: systolic peak, dicrotic notch, diastolic peak.

    Rangayyan (2024) Section 1.2.11 defines the signal: a photoplethysmogram
    records changes in blood volume in a microvascular bed, and consists of a
    pulsatile component riding on a low-frequency or DC component.  The ratio
    of the two is the perfusion index, which is reported here because it is
    the quantity that says how much of the reading is actually pulsatile
    blood volume rather than static tissue absorption.

    The pulse shape mirrors the carotid pulse of Section 1.2.10 -- rapid
    systolic upstroke, a dicrotic notch at aortic valve closure, then a
    diastolic wave -- so the notch is located with the same machinery the book
    gives for the carotid pulse in Section 4.3.5: the least-squares second
    derivative of Equation 4.22 squared and smoothed by Equation 4.23.  The
    second derivative is used for the same reason as there: the notch rides on
    a falling limb, which a first derivative cannot separate from the limb.

    Parameters
    ----------
    ppg : sequence of float
        PPG samples.
    fs : float
        Sampling rate in Hz.
    mwin : int
        Smoothing window M of Equation 4.23.

    Returns
    -------
    RichResult
        payload["systolic"], payload["notch"], payload["diastolic"] and
        payload["onset"] (the pulse foot) sample positions, payload["amplitude"]
        per pulse measured foot to systolic peak, payload["pi"] the perfusion
        index in percent, payload["rate"] pulses per minute.
    """
    fs = _rgfs(fs)
    ppg = _rgcheck(ppg, 32, "PPG")
    dn = dicnotch(ppg, fs, mwin=mwin)
    notch = list(dn["notch"])
    if not notch:
        raise ValueError("no dicrotic notch found; cannot delineate PPG pulses")
    n = len(ppg)
    back = max(2, int(round(0.400 * fs)))

    sysp, diap, feet, amp = [], [], [], []
    for d in notch:
        a = max(0, d - back)
        if d - a < 3:
            continue
        s = max(range(a, d), key=lambda i: ppg[i])
        if s <= a:
            continue
        f = min(range(a, s), key=lambda i: ppg[i])
        sysp.append(s)
        feet.append(f)
        hi = min(n, d + int(round(0.250 * fs)))
        diap.append(max(range(d, hi), key=lambda i: ppg[i]) if hi > d + 1 else d)
        amp.append(ppg[s] - ppg[f])
    if not amp:
        raise ValueError("could not delineate any complete PPG pulse")

    dc = fsum(ppg) / n
    ac = fsum(amp) / len(amp)
    pi = (100.0 * ac / dc) if dc != 0.0 else None
    rate = 60.0 * len(sysp) / (n / fs)
    return RichResult(payload={
        "systolic": sysp,
        "notch": notch,
        "diastolic": diap,
        "onset": feet,
        "amplitude": amp,
        "ac": ac,
        "dc": dc,
        "pi": pi,
        "rate": rate,
        "fs": fs,
        "method": "PPG pulse features with the carotid dicrotic-notch operator, Rangayyan (2024) Sections 1.2.11 and 4.3.5 (Eqs 4.22, 4.23)",
    })


rangayyan_ppg_features = ppgfeat  # pre-policy spelling


# -- rgpwave: P-wave detection in ECG using search window relative to R-peak.
def pwavedet(x, qrs, fs, template=None):
    """P-wave detection by QRS deletion, bandpass filtering and ternary matching.

    The method of Hengeveld and van Bemmel as given in Rangayyan (2024)
    Section 4.3.3.  The P wave is low in amplitude, variable in shape and
    often sits in noise, so the algorithm never searches for it directly.
    Instead:

    1. the QRS is detected, deleted and replaced by the baseline estimated
       from the samples preceding it;
    2. the result is bandpass filtered with -3 dB points at 3 and 11 Hz;
    3. the search interval runs from the end of the preceding T wave to the
       current QRS, with the T end estimated as QTmax = 29 RR + 250 ms;
    4. maximum and minimum are found in the search interval;
    5. the signal is rectified and thresholded at 50% and 75% of the maximum
       to give a ternary signal;
    6. that ternary signal is cross-correlated with a ternary template
       derived the same way from a representative set of P waves;
    7. the peak of the cross-correlation gives the P position.

    Matching a ternary version of the wave rather than the raw P wave is what
    makes the method tolerant of P-wave amplitude and shape variation.

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    qrs : sequence of int
        QRS sample positions.
    fs : float
        Sampling rate in Hz.
    template : sequence of float, optional
        Ternary P template.  When omitted it is derived from the data: the
        search intervals of all beats are averaged, then ternarised by the
        same 50%/75% rule, which is the book's "representative set of P waves".

    Returns
    -------
    RichResult
        payload["p"] one P-wave sample position per beat (None where the beat
        has no usable search interval), plus the ternary template used.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 16, "ECG")
    q = [int(v) for v in aslist(qrs)]
    if len(q) < 2:
        raise ValueError("need at least two QRS positions to define an RR interval")
    n = len(x)

    # step 1: replace each QRS with the baseline taken from the samples before it
    half = max(1, int(round(0.050 * fs)))
    base = max(1, int(round(0.040 * fs)))
    y = list(x)
    for pos in q:
        a, b = max(0, pos - half), min(n, pos + half)
        ref = x[max(0, a - base):a]
        lvl = (fsum(ref) / len(ref)) if ref else 0.0
        for i in range(a, b):
            y[i] = lvl

    # step 2: 3-11 Hz bandpass as the difference of two moving averages
    lo = _rgmavg(y, max(1, int(round(fs / 11.0))))
    hi = _rgmavg(y, max(1, int(round(fs / 3.0))))
    bp = [a - b for a, b in zip(lo, hi)]

    # step 3: search interval from the end of the preceding T wave to this QRS
    wins = []
    for k in range(1, len(q)):
        rr = (q[k] - q[k - 1]) / fs
        qtmax = 29.0 * rr + 0.250
        start = max(q[k - 1], q[k - 1] + int(round(min(qtmax, 0.75 * rr) * fs)))
        stop = max(start + 2, q[k] - half)
        if stop <= start or stop > n:
            wins.append(None)
        else:
            wins.append((start, stop))

    usable = [w for w in wins if w]
    if not usable:
        raise ValueError("no usable P-wave search interval between the supplied QRS positions")
    wlen = min(b - a for a, b in usable)

    def _ternary(seg):
        m = max(abs(v) for v in seg)
        if m <= 0.0:
            return [0.0] * len(seg)
        out = []
        for v in seg:
            r = abs(v) / m
            out.append(2.0 if r >= 0.75 else (1.0 if r >= 0.50 else 0.0))
        return out

    if template is None:
        acc = [0.0] * wlen
        for a, b in usable:
            seg = bp[a:a + wlen]
            for i in range(wlen):
                acc[i] += seg[i]
        template = _ternary([v / len(usable) for v in acc])
    else:
        template = _rgcheck(template, 2, "template")
    tl = len(template)

    ppos = []
    for w in wins:
        if w is None:
            ppos.append(None)
            continue
        a, b = w
        tern = _ternary(bp[a:b])
        if len(tern) < tl:
            ppos.append(None)
            continue
        best, bestv = 0, None
        for s in range(len(tern) - tl + 1):
            v = fsum(tern[s + i] * template[i] for i in range(tl))
            if bestv is None or v > bestv:
                best, bestv = s, v
        ppos.append(a + best + tl // 2)
    return RichResult(payload={
        "p": ppos,
        "template": list(template),
        "windows": wins,
        "bandpass": bp,
        "fs": fs,
        "method": "P-wave detection, Rangayyan (2024) Section 4.3.3 (Hengeveld and van Bemmel)",
    })


rangayyan_p_wave_detect = pwavedet  # pre-policy spelling


# -- rgrpsig: ECG-derived respiration (EDR) via R-wave amplitude modulation.
def edrsignal(x, qrs, fs, fsr=4.0):
    """ECG-derived respiration from R-wave amplitude modulation.

    Rangayyan (2024) Section 2.2.4 sets out the physiology: the fall in
    intrapleural pressure during inspiration impedes vagus nerve activity and
    raises the heart rate, so breathing modulates the cardiac signal; the book
    also notes that respiration modifies the ECG itself, not only the rate.
    Chest motion and the changing thoracic impedance during the breathing
    cycle swing the cardiac electrical axis, which modulates the R-wave
    amplitude beat by beat.  Sampling that amplitude once per beat gives a
    respiratory signal without a respiration sensor.

    The book gives no equation for the extraction; the algorithmic reference
    it cites is Arunachalam SP and Brown LF, Real-time estimation of the
    ECG-derived respiration (EDR) signal, Proc. 31st Annual International
    Conference of the IEEE EMBS, pp. 5681-5684, 2009 (reference [52] of the
    book's Chapter 2).

    The beat-sampled amplitude series is unevenly spaced, so it is
    interpolated onto a uniform grid before the respiratory rate is estimated
    from the dominant spectral peak between 0.1 and 0.5 Hz (6 to 30 breaths
    per minute).

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    qrs : sequence of int
        QRS (R peak) sample positions.
    fs : float
        Sampling rate in Hz.
    fsr : float
        Resampling rate for the EDR signal, in Hz.

    Returns
    -------
    RichResult
        payload["edr"] the uniformly sampled respiratory signal, payload["amp"]
        the raw beat-by-beat R amplitudes, payload["resprate"] in breaths per
        minute, payload["fsr"].
    """
    fs = _rgfs(fs)
    fsr = _rgfs(fsr)
    x = _rgcheck(x, 16, "ECG")
    q = [int(v) for v in aslist(qrs)]
    if len(q) < 8:
        raise ValueError("need at least eight beats to estimate respiration")
    n = len(x)
    pqa = max(1, int(round(0.080 * fs)))
    pqb = max(1, int(round(0.040 * fs)))
    half = max(1, int(round(0.050 * fs)))

    times, amps = [], []
    for pos in q:
        a = max(0, pos - pqa)
        b = max(a + 1, pos - pqb)
        ref = fsum(x[a:b]) / (b - a)
        lo, hi = max(0, pos - half), min(n, pos + half + 1)
        if hi - lo < 2:
            continue
        amps.append(max(x[lo:hi]) - ref)
        times.append(pos / fs)
    if len(amps) < 8:
        raise ValueError("too few usable beats for an EDR estimate")

    span = times[-1] - times[0]
    m = int(span * fsr)
    if m < 8:
        raise ValueError("recording too short for an EDR estimate at this fsr")
    grid = []
    j = 0
    for k in range(m):
        tk = times[0] + k / fsr
        while j < len(times) - 2 and times[j + 1] < tk:
            j += 1
        t0, t1 = times[j], times[j + 1]
        w = 0.0 if t1 <= t0 else (tk - t0) / (t1 - t0)
        w = min(1.0, max(0.0, w))
        grid.append(amps[j] * (1.0 - w) + amps[j + 1] * w)
    mu = fsum(grid) / len(grid)
    edr = [v - mu for v in grid]

    freqs, power = _rgpsd(edr, fsr)
    cand = [k for k in range(len(freqs)) if 0.1 <= freqs[k] <= 0.5]
    if cand:
        kbest = max(cand, key=lambda k: power[k])
        rate = 60.0 * freqs[kbest]
    else:
        rate = None
    return RichResult(payload={
        "edr": edr,
        "amp": amps,
        "times": times,
        "resprate": rate,
        "fsr": fsr,
        "nbeats": len(amps),
        "method": "ECG-derived respiration from R-wave amplitude modulation; Rangayyan (2024) Section 2.2.4 for the physiology, Arunachalam and Brown, Proc. IEEE EMBC 2009, pp. 5681-5684, for the estimator",
    })


rangayyan_resp_signal = edrsignal  # pre-policy spelling


# -- rgsapn: Sleep apnea detection via ECG-derived respiration + SpO2 fusion.
def apneaedr(edr, spo2, fs, hours=None, mindur=10.0, desat=3.0):
    """Sleep apnea event detection from an EDR signal fused with SpO2.

    Rangayyan (2024) Section 10.2.5 describes the problem: obstructive sleep
    apnea is recurrent partial or total upper-airway obstruction; laboratory
    polysomnography is the diagnostic standard but is slow and uncomfortable,
    so automatic detection from a smaller set of signals, scored against the
    apnea-hypopnea index, is of interest.  The book gives no detection
    equation and names no specific algorithm, so none is attributed to it.
    What is implemented here is the conventional scoring logic stated in the
    section's own terms: a respiratory pause of at least mindur seconds,
    confirmed by an oxygen desaturation, counted per hour of recording.

    An event is scored where the envelope of the respiratory signal falls
    below 30 percent of its running baseline for at least mindur seconds and
    the SpO2 trace drops by at least desat percentage points from its
    pre-event level within the following 30 seconds.  Requiring both is what
    keeps a motion-induced flat stretch of EDR from being scored as an apnea.

    Parameters
    ----------
    edr : sequence of float
        Respiratory signal, for example the output of the EDR routine.
    spo2 : sequence of float
        Oxygen saturation in percent, sampled at the same rate as edr.
    fs : float
        Sampling rate of both signals, in Hz.
    hours : float, optional
        Recording duration in hours for the index.  Defaults to the signal
        length divided by fs.
    mindur : float
        Minimum event duration in seconds.
    desat : float
        Minimum desaturation in percentage points.

    Returns
    -------
    RichResult
        payload["events"] as (start, end) sample pairs, payload["nevents"],
        payload["ahi"] events per hour, payload["desatdepth"] per event.
    """
    fs = _rgfs(fs)
    edr = _rgcheck(edr, 16, "EDR")
    spo2 = _rgcheck(spo2, 16, "SpO2")
    if len(edr) != len(spo2):
        raise ValueError("EDR and SpO2 must have the same length and rate")
    mindur = float(mindur)
    desat = float(desat)
    if not mindur > 0.0:
        raise ValueError("mindur must be positive")
    if not desat > 0.0:
        raise ValueError("desat must be positive")
    n = len(edr)
    if hours is None:
        hours = (n / fs) / 3600.0
    hours = float(hours)
    if not hours > 0.0:
        raise ValueError("hours must be positive")

    env = _rgmavg([abs(v) for v in edr], max(1, int(round(2.0 * fs))))
    base = _rgmavg(env, max(1, int(round(120.0 * fs))))
    low = [env[i] < 0.30 * base[i] for i in range(n)]
    need = max(1, int(round(mindur * fs)))
    look = max(1, int(round(30.0 * fs)))

    events, depth = [], []
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if j - i >= need:
                pre = spo2[max(0, i - look):i] or spo2[i:i + 1]
                lvl = max(pre)
                after = spo2[i:min(n, j + look)]
                drop = lvl - min(after) if after else 0.0
                if drop >= desat:
                    events.append((i, j))
                    depth.append(drop)
            i = j
        else:
            i += 1
    return RichResult(payload={
        "events": events,
        "nevents": len(events),
        "ahi": len(events) / hours,
        "desatdepth": depth,
        "hours": hours,
        "mindur": mindur,
        "desat": desat,
        "fs": fs,
        "method": "apnea event scoring from respiratory-envelope pauses confirmed by SpO2 desaturation; Rangayyan (2024) Section 10.2.5 frames the problem and the AHI but gives no detection algorithm",
    })


rangayyan_sleep_apnea = apneaedr  # pre-policy spelling


# -- rgspr: Spectral power ratio (LF/HF) for HRV analysis.
def lfhfratio(rr, fsr=4.0, bands="taskforce"):
    """LF to HF spectral power ratio of an RR interval series.

    Rangayyan (2024) Section 8.12 and its Figure 8.38 present the
    low-frequency to high-frequency power ratio as one of the parameters
    derived from a time-varying HRV spectrum, alongside RR variance and the
    percentage LF and HF powers, and report that LF power rises 1.5 to 2
    minutes before an ischemic event.  The ratio is read as an index of
    sympathovagal balance: LF reflects sympathetic and parasympathetic
    activation together, HF the vagal and respiratory component.

    This is the ratio alone, with the band powers and RR variance alongside it
    so the ratio can be interpreted rather than read bare.

    Parameters
    ----------
    rr : sequence of float
        RR intervals in seconds.
    fsr : float
        Resampling rate for the interpolated tachogram, in Hz.
    bands : str
        "taskforce" or "bianchi", as in the frequency-domain HRV routine.

    Returns
    -------
    RichResult
        payload["lfhf"], payload["lf"], payload["hf"], payload["lfpct"],
        payload["hfpct"] and payload["rrvar"].
    """
    res = hrvfreq(rr, fsr=fsr, bands=bands)
    v = _rgcheck(rr, 2, "RR series")
    mu = fsum(v) / len(v)
    var = fsum((x - mu) ** 2 for x in v) / (len(v) - 1)
    return RichResult(payload={
        "lfhf": res["lfhf"],
        "lf": res["lf"],
        "hf": res["hf"],
        "lfpct": res["lfpct"],
        "hfpct": res["hfpct"],
        "rrvar": var,
        "bands": bands,
        "n": len(v),
        "method": "LF/HF spectral power ratio, Rangayyan (2024) Section 8.12 and Figure 8.38 (Bianchi et al.)",
    })


rangayyan_spectral_power_ratio = lfhfratio  # pre-policy spelling


# -- rgtwa: T-wave alternans (TWA) detection via spectral method.
def twaspectr(twaves, noiselo=0.33, noisehi=0.45):
    """T-wave alternans detection by the spectral method.

    Rangayyan (2024) Section 9.10 describes TWA as repolarisation alternans:
    a shift in T-wave form or amplitude on every second beat, alternating
    between an "A" pattern and a "B" pattern, whose magnitude predicts risk of
    sudden cardiac death.  The section states that in the spectral method the
    power spectra of the T waves are aligned and averaged, citing Smith JM,
    Clancy EA, Valeri CR, Ruskin JN and Cohen RJ, Electrical alternans and
    cardiac electrical instability, Circulation 77(1):110-121, 1988
    (reference [5] of the book's Chapter 9).

    Alternation every other beat is a period of exactly two beats, so in the
    beat domain it appears at 0.5 cycles per beat, the very last bin of the
    beat-series spectrum.  The alternans power is that bin; the noise floor is
    estimated from a nearby band that contains no alternans.  The reported
    voltage is sqrt(alternans power minus noise mean), and the k-score is the
    number of noise standard deviations by which the alternans bin exceeds the
    noise mean, which is what makes a small microvolt-scale peak assessable.

    Parameters
    ----------
    twaves : sequence of sequences of float
        Aligned T waves, one per beat, all of the same length.  An even number
        of beats is required so that 0.5 cycles per beat is an exact bin.
    noiselo, noisehi : float
        Reference noise band in cycles per beat.

    Returns
    -------
    RichResult
        payload["valt"] the alternans voltage in the input units,
        payload["kscore"], payload["altpower"], payload["noisemean"],
        payload["noisesd"], and payload["present"] (k-score >= 3).
    """
    if not twaves:
        raise ValueError("need at least one T wave")
    beats = [_rgcheck(t, 2, "T wave") for t in twaves]
    m = len(beats)
    if m < 8:
        raise ValueError("spectral TWA needs at least eight beats")
    if m % 2:
        beats = beats[:-1]
        m -= 1
    npts = len(beats[0])
    if any(len(t) != npts for t in beats):
        raise ValueError("all T waves must have the same number of samples")
    if not 0.0 < noiselo < noisehi < 0.5:
        raise ValueError("noise band must satisfy 0 < noiselo < noisehi < 0.5")

    # average the beat-series power spectrum over all sample points of the T wave
    acc = [0.0] * (m // 2 + 1)
    for j in range(npts):
        series = [beats[i][j] for i in range(m)]
        mu = fsum(series) / m
        spec = _rgdft([v - mu for v in series])
        for k in range(m // 2 + 1):
            re, im = spec[k]
            p = (re * re + im * im) / (m * m)
            if 0 < k < m - k:
                p *= 2.0
            acc[k] += p
    acc = [v / npts for v in acc]

    cyc = [k / float(m) for k in range(m // 2 + 1)]
    kalt = m // 2                      # 0.5 cycles per beat
    band = [k for k in range(len(cyc)) if noiselo <= cyc[k] <= noisehi]
    if len(band) < 2:
        raise ValueError("noise band contains fewer than two spectral bins; use more beats")
    nm = fsum(acc[k] for k in band) / len(band)
    nsd = sqrt(fsum((acc[k] - nm) ** 2 for k in band) / (len(band) - 1))
    altp = acc[kalt]
    valt = sqrt(altp - nm) if altp > nm else 0.0
    kscore = ((altp - nm) / nsd) if nsd > 0.0 else None
    return RichResult(payload={
        "valt": valt,
        "kscore": kscore,
        "altpower": altp,
        "noisemean": nm,
        "noisesd": nsd,
        "nbeats": m,
        "npoints": npts,
        "cyclesperbeat": 0.5,
        "present": bool(kscore is not None and kscore >= 3.0),
        "method": "spectral method for T-wave alternans, Rangayyan (2024) Section 9.10 citing Smith et al., Circulation 77(1):110-121, 1988",
    })


rangayyan_twave_alternans = twaspectr  # pre-policy spelling


# -- rgtwamx: T-wave alternans spectral method.
def rangayyan_twa_spectral_mx(ecg, fs, r_peaks, t_window=None, n_beats=128):
    r"""Spectral T-wave alternans (Rangayyan Ch. 3).

    Builds a beat-by-sample matrix aligned on the R peaks, takes the
    FFT ALONG THE BEAT AXIS at each sample offset, and reads the power
    at 0.5 cycles/beat -- the frequency of a strictly ABAB alternation:

    .. math:: k_{alt} = \tfrac12 \text{ cycles per beat}.

    The alternans voltage is the excess over the neighbouring noise
    band, and the k-score is that excess in noise standard deviations;
    both are returned because a raw spectral peak means nothing
    without its noise floor. An even number of beats is required, or
    0.5 cycles/beat is not an exact FFT bin and the alternans power
    leaks.

    Parameters
    ----------
    ecg : array-like
        ECG signal.
    fs : float
        Sampling frequency.
    r_peaks : array-like of int
        R-peak indices.
    t_window : (int, int), optional
        Offsets after R defining the T wave; a physiological default
        of 100-300 ms is used otherwise.
    n_beats : int, default 128
        Beats to use (truncated to an even number).

    Returns
    -------
    RichResult
        keys: ``alternans_voltage``, ``k_score``, ``noise_mean``,
        ``noise_std``, ``spectrum``, ``n_beats_used``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (T-wave alternans).
    """
    x = np.asarray(ecg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 8:
        raise ValueError(f"need at least 8 beats, got {r.size}.")
    lo, hi = (int(0.10 * fs), int(0.30 * fs)) if t_window is None else (
        int(t_window[0]), int(t_window[1])
    )
    if not 0 <= lo < hi:
        raise ValueError("t_window must satisfy 0 <= start < stop.")
    usable = [p for p in r if p + hi <= x.size]
    M = min(int(n_beats), len(usable))
    M -= M % 2  # 0.5 cycles/beat must land on an exact FFT bin
    if M < 8:
        raise ValueError("fewer than 8 complete beats after alignment.")
    mat = np.array([x[p + lo : p + hi] for p in usable[:M]])
    mat = mat - mat.mean(axis=0, keepdims=True)
    S = np.abs(np.fft.rfft(mat, axis=0)) ** 2 / M
    spec = S.sum(axis=1)  # aggregate across the T-wave samples
    k_alt = M // 2  # the 0.5 cycles/beat bin
    noise_band = spec[int(0.33 * len(spec)) : k_alt]
    nm = float(noise_band.mean()) if noise_band.size else 0.0
    ns = float(noise_band.std()) if noise_band.size else 0.0
    excess = float(spec[k_alt]) - nm
    volt = float(np.sqrt(max(excess, 0.0)))
    return RichResult(payload={"alternans_voltage": volt,
                               "k_score": (excess / ns) if ns > 0 else np.inf,
                               "noise_mean": nm, "noise_std": ns, "spectrum": spec,
                               "n_beats_used": int(M),
                               "method": "FFT along the beat axis; power at 0.5 cyc/beat over noise"})


# -- rgtwave: T-wave detection in ECG.
def twavedet(chans, qrs, fs, tdur=0.160):
    """T-wave detection using the length transformation.

    Rangayyan (2024) Section 4.3.4 (Gritzali et al.).  The length transform of
    Equation 4.21 accumulates the multichannel arc length over a moving window
    whose width is set to the average duration of the wave being sought.  The
    QRS is detected first with the window at the QRS width, blanked to the
    isoelectric baseline, and the procedure repeated with the window at the
    average T duration.

    Working across channels is what makes T detection tractable: a T wave that
    is well defined in only one lead still raises the summed squared
    derivative, so it is detected even where most leads show nothing.

    Parameters
    ----------
    chans : sequence of sequences of float
        One ECG channel or several of equal length.
    qrs : sequence of int
        QRS sample positions, used to blank the QRS and to bound the search.
    fs : float
        Sampling rate in Hz.
    tdur : float
        Average T-wave duration in seconds used as the window width w.

    Returns
    -------
    RichResult
        payload["t"] the T-wave peak position for each beat, payload["onset"]
        and payload["offset"] its bounds, payload["length"] the transform.
    """
    fs = _rgfs(fs)
    tdur = float(tdur)
    if not tdur > 0.0:
        raise ValueError("tdur must be positive")
    q = [int(v) for v in aslist(qrs)]
    if len(q) < 1:
        raise ValueError("need at least one QRS position")
    first = chans[0] if len(chans) else None
    if first is None:
        raise ValueError("need at least one channel")
    if isinstance(first, (int, float)):
        chans = [chans]
    ch = [_rgcheck(c, 8, "channel") for c in chans]
    n = len(ch[0])
    if any(len(c) != n for c in ch):
        raise ValueError("all channels must have the same length")

    # blank the QRS to the isoelectric level taken just before it
    half = max(1, int(round(0.050 * fs)))
    base = max(1, int(round(0.040 * fs)))
    blank = []
    for c in ch:
        b = list(c)
        for pos in q:
            a1, a2 = max(0, pos - half), min(n, pos + half)
            ref = c[max(0, a1 - base):a1]
            lvl = (fsum(ref) / len(ref)) if ref else 0.0
            for i in range(a1, a2):
                b[i] = lvl
        blank.append(b)

    lt = list(lengthxfm(blank, tdur, fs)["length"])
    wsamp = max(1, int(round(tdur * fs)))
    lo = max(1, int(round(0.100 * fs)))
    hi = int(round(0.450 * fs))

    tpos, onset, offset = [], [], []
    for pos in q:
        a, b = min(n - 1, pos + lo), min(n, pos + hi)
        if b - a < 2:
            tpos.append(None)
            onset.append(None)
            offset.append(None)
            continue
        s = max(range(a, b), key=lambda i: lt[i])
        onset.append(s)
        offset.append(min(n - 1, s + wsamp))
        seg = range(s, min(n, s + wsamp))
        tpos.append(max(seg, key=lambda i: abs(fsum(c[i] for c in blank))))
    return RichResult(payload={
        "t": tpos,
        "onset": onset,
        "offset": offset,
        "length": lt,
        "tdur": tdur,
        "nchan": len(ch),
        "fs": fs,
        "method": "T-wave detection by the length transformation, Rangayyan (2024) Section 4.3.4, Eq 4.21 (Gritzali et al.)",
    })


rangayyan_t_wave_detect = twavedet  # pre-policy spelling


# -- rgvf: Ventricular fibrillation (VF) detection in ECG.
def vfdetect(x, fs, win=4.0, conc=0.60, crest=4.0):
    """Ventricular fibrillation detection from spectral concentration and QRS absence.

    What the book supports: Rangayyan (2024) Section 1.2.4 states that
    ventricular dissociation and fibrillation are a state of disorganised
    contraction, and Section 8.11 analyses porcine ventricular fibrillation
    waveforms with the wavelet transform in a cardiopulmonary resuscitation
    study.  Neither gives a detection algorithm or a threshold, and none is
    attributed to the book here.  No external primary source for a VF
    detector was verified in this session, so the decision rule below is
    stated as what it is: a two-part heuristic built from the two properties
    the book does assert about fibrillation.

    Part one, absence of discrete QRS complexes.  The Pan-Tompkins front end
    is run over the window and the crest factor of its integrator output --
    the peak divided by the mean -- is taken.  An organised rhythm produces a
    spiky integrator trace, one tall pulse per beat separated by near-zero
    stretches, so the crest factor is large.  Fibrillation has no steep,
    discrete complexes for the derivative-and-square stage to respond to, so
    the integrator output is comparatively flat and the crest factor small.
    Counting threshold crossings alone does not work here, because a
    quasi-sinusoidal fibrillation waveform still produces one integrator peak
    per cycle and so mimics a plausible beat rate.

    Part two, spectral concentration: fibrillation is quasi-sinusoidal at
    roughly 4 to 7 Hz, so the fraction of total power falling in a narrow band
    around the dominant spectral peak is high, whereas a normal ECG spreads
    its power over the harmonics of the QRS.

    A window is flagged when the crest factor is at or below crest, the
    spectral concentration is at or above conc, and the dominant frequency
    lies between 1 and 10 Hz.

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    fs : float
        Sampling rate in Hz.
    win : float
        Analysis window length in seconds.
    conc : float
        Minimum fraction of 0.5-25 Hz power within +/- 1.5 Hz of the dominant
        peak for a window to be flagged.
    crest : float
        Maximum integrator crest factor for a window to be flagged.

    Returns
    -------
    RichResult
        payload["flag"] one boolean per window, payload["domfreq"],
        payload["concentration"], payload["crest"], payload["rate"], and
        payload["fraction"], the proportion of windows flagged.
    """
    fs = _rgfs(fs)
    x = _rgcheck(x, 64, "ECG")
    win = float(win)
    conc = float(conc)
    crest = float(crest)
    if not win > 0.0:
        raise ValueError("win must be positive")
    if not 0.0 < conc <= 1.0:
        raise ValueError("conc must lie in (0, 1]")
    if not crest > 1.0:
        raise ValueError("crest must be greater than 1")
    wsamp = max(32, int(round(win * fs)))
    if len(x) < wsamp:
        raise ValueError("signal shorter than one analysis window")

    flags, doms, concs, rates, crests = [], [], [], [], []
    for a in range(0, len(x) - wsamp + 1, wsamp):
        seg = x[a:a + wsamp]
        mu = fsum(seg) / len(seg)
        seg = [v - mu for v in seg]
        freqs, power = _rgpsd(seg, fs)
        band = [k for k in range(len(freqs)) if 0.5 <= freqs[k] <= 25.0]
        if not band:
            raise ValueError("analysis window too short to resolve the 0.5-25 Hz band")
        kbest = max(band, key=lambda k: power[k])
        fdom = freqs[kbest]
        tot = fsum(power[k] for k in band)
        near = [k for k in band if abs(freqs[k] - fdom) <= 1.5]
        cval = (fsum(power[k] for k in near) / tot) if tot > 0.0 else 0.0
        try:
            det = qrsdetect(seg, fs)
            nb = len(det["qrs"])
            ig = det["integrated"]
            mu2 = fsum(ig) / len(ig)
            cf = (max(ig) / mu2) if mu2 > 0.0 else 0.0
        except ValueError:
            nb, cf = 0, 0.0
        rate = 60.0 * nb / (wsamp / fs)
        flags.append(bool(cf <= crest and cval >= conc and 1.0 <= fdom <= 10.0))
        doms.append(fdom)
        concs.append(cval)
        rates.append(rate)
        crests.append(cf)
    return RichResult(payload={
        "flag": flags,
        "domfreq": doms,
        "concentration": concs,
        "crest": crests,
        "rate": rates,
        "conc": conc,
        "crestmax": crest,
        "nwin": len(flags),
        "fraction": sum(1 for f in flags if f) / float(len(flags)),
        "win": win,
        "fs": fs,
        "method": "VF heuristic from QRS absence and spectral concentration; Rangayyan (2024) Sections 1.2.4 and 8.11 describe VF but give no detector, and no external primary source was verified for this rule",
    })


rangayyan_vf_detect = vfdetect  # pre-policy spelling


# -- rng176: Smoothed three-point first derivative used in QRS detection (Balda et al.)..
def qrsderiv1(x):
    """Smoothed three-point first derivative used for QRS detection.

    y0(n) = |x(n) - x(n-2)|, Rangayyan (2024) Equation 4.1.  The two-sample
    span is what makes it "smoothed": it is a first difference over a wider
    lag than one sample, which suppresses sample-to-sample noise while still
    responding to the steep Q-R and R-S swings.  The absolute value makes the
    output polarity-independent, so an inverted QRS gives the same response.

    Parameters
    ----------
    x : sequence of float
        ECG samples.  The book's illustration used fs = 200 Hz after
        lowpass filtering to 90 Hz and normalisation by the maximum.

    Returns
    -------
    RichResult
        payload["y0"] is the operator output, zero-padded on the left to the
        length of x (the equation is undefined for n < 2).
    """
    x = _rgcheck(x, 3, "ECG")
    y0 = [abs(x[n] - x[n - 2]) for n in range(2, len(x))]
    return RichResult(payload={
        "y0": _rgpad(y0, 2),
        "n": len(x),
        "method": "first-derivative QRS operator, Rangayyan (2024) Eq 4.1 (Balda et al.)",
    })


rangayyan_ch4_qrs_first_derivative_balda = qrsderiv1  # pre-policy spelling


# -- rng177: Approximation of the second derivative used in QRS detection..
def qrsderiv2(x):
    """Second-derivative approximation used for QRS detection.

    y1(n) = |x(n) - 2 x(n-2) + x(n-4)|, Rangayyan (2024) Equation 4.2.  The
    second difference removes any locally linear trend, so a constant slope
    (the T-wave limbs, baseline drift) produces no output while the curvature
    at the QRS peak produces a large one.

    Parameters
    ----------
    x : sequence of float
        ECG samples.

    Returns
    -------
    RichResult
        payload["y1"], left-zero-padded by four samples.
    """
    x = _rgcheck(x, 5, "ECG")
    y1 = [abs(x[n] - 2.0 * x[n - 2] + x[n - 4]) for n in range(4, len(x))]
    return RichResult(payload={
        "y1": _rgpad(y1, 4),
        "n": len(x),
        "method": "second-derivative QRS operator, Rangayyan (2024) Eq 4.2 (Balda et al.)",
    })


rangayyan_ch4_qrs_second_derivative_balda = qrsderiv2  # pre-policy spelling


# -- rng178: Weighted combination of first and second derivatives for QRS detection..
def qrsderivmx(y0, y1):
    """Weighted mix of the first- and second-derivative QRS operators.

    y2(n) = 1.3 y0(n) + 1.1 y1(n), Rangayyan (2024) Equation 4.3.  The two
    operators respond to different parts of the complex (slope versus
    curvature); combining them gives one detection function whose peak is
    reliably inside the QRS.  The book then scans y2 with a fixed threshold
    of 1.0 on a maximum-normalised ECG.

    Parameters
    ----------
    y0, y1 : sequence of float
        Outputs of Equations 4.1 and 4.2, of equal length.

    Returns
    -------
    RichResult
        payload["y2"] and the two weights actually applied.
    """
    y0 = _rgcheck(y0, 1, "y0")
    y1 = _rgcheck(y1, 1, "y1")
    if len(y0) != len(y1):
        raise ValueError("y0 and y1 must have the same length")
    return RichResult(payload={
        "y2": [1.3 * a + 1.1 * b for a, b in zip(y0, y1)],
        "w0": 1.3,
        "w1": 1.1,
        "n": len(y0),
        "method": "combined derivative QRS operator, Rangayyan (2024) Eq 4.3 (Balda et al.)",
    })


rangayyan_ch4_qrs_combined_balda = qrsderivmx  # pre-policy spelling


# -- rng179: Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj)..
def qrswsqdrv(x, nwin=8):
    """Weighted and squared first-derivative operator for QRS detection.

    g1(n) = sum_{i=1}^{N} |x(n-i+1) - x(n-i)|^2 (N - i + 1),
    Rangayyan (2024) Equation 4.4 (Murthy and Rangaraj).  Squaring makes every
    term positive and preferentially rewards the large differences of the QRS;
    the weight (N - i + 1) falls linearly from the newest difference back to
    the oldest, which smooths the result without a separate filter.

    Parameters
    ----------
    x : sequence of float
        ECG samples.
    nwin : int
        Window width N.  The book used N = 8 at fs = 100 Hz.

    Returns
    -------
    RichResult
        payload["g1"], left-zero-padded to the length of x.
    """
    x = _rgcheck(x, 2, "ECG")
    nwin = int(nwin)
    if nwin < 1:
        raise ValueError("nwin must be >= 1")
    if len(x) <= nwin:
        raise ValueError("signal must be longer than the window width")
    g1 = []
    for n in range(nwin, len(x)):
        g1.append(fsum(
            (x[n - i + 1] - x[n - i]) ** 2 * (nwin - i + 1)
            for i in range(1, nwin + 1)
        ))
    return RichResult(payload={
        "g1": _rgpad(g1, nwin),
        "nwin": nwin,
        "n": len(x),
        "method": "weighted squared first-derivative operator, Rangayyan (2024) Eq 4.4 (Murthy and Rangaraj)",
    })


rangayyan_ch4_filtered_derivative_murthy = qrswsqdrv  # pre-policy spelling


# -- rng180: MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector..
def qrsdrvsmth(g1, mwin=8):
    """Moving-average smoothing of the weighted-derivative QRS output.

    g(n) = (1/M) sum_{j=0}^{M-1} g1(n-j), Rangayyan (2024) Equation 4.5.  It
    collapses the multiple peaks that any derivative operator produces across
    the Q-R-S swings into one smooth pulse per beat, which is what makes a
    plain peak search usable.

    Parameters
    ----------
    g1 : sequence of float
        Output of Equation 4.4.
    mwin : int
        Filter length M.  The book used M = 8 at fs = 100 Hz.

    Returns
    -------
    RichResult
        payload["g"].  The first M-1 outputs use a shortened window rather
        than being discarded, so the length matches g1.
    """
    g1 = _rgcheck(g1, 1, "g1")
    mwin = int(mwin)
    if mwin < 1:
        raise ValueError("mwin must be >= 1")
    return RichResult(payload={
        "g": _rgmavg(g1, mwin),
        "mwin": mwin,
        "n": len(g1),
        "method": "MA smoothing of the weighted-derivative output, Rangayyan (2024) Eq 4.5",
    })


rangayyan_ch4_qrs_smoothing_ma_filter = qrsdrvsmth  # pre-policy spelling


# -- rng181: Lowpass transfer function used in the Pan-Tompkins QRS detector..
def qrslpasstf(freq, fs=200.0):
    """Transfer function of the Pan-Tompkins lowpass filter.

    H(z) = (1/32) (1 - z^-6)^2 / (1 - z^-1)^2, Rangayyan (2024) Equation 4.7.
    The integer coefficients are the point of the design: no multiplies are
    needed in a real-time implementation.  The ratio is evaluated in its
    equivalent finite form (sum_{k=0}^{5} z^-k)^2 / 32, which removes the
    apparent pole at z = 1 exactly instead of numerically.

    At fs = 200 Hz this gives fc = 11 Hz, a delay of 5 samples (25 ms) and
    more than 35 dB of attenuation at 60 Hz.  Those numbers are tied to
    200 Hz; the coefficients do not rescale with fs.

    Parameters
    ----------
    freq : float or sequence of float
        Frequencies in Hz at which to evaluate the response.
    fs : float
        Sampling rate.  Defaults to the book's 200 Hz, which is the rate the
        coefficients were designed for.

    Returns
    -------
    RichResult
        payload["mag"], payload["phase"] (radians), and the numerator and
        denominator coefficient lists b and a.
    """
    fs = _rgfs(fs)
    fr = aslist(freq)
    if not fr:
        raise ValueError("need at least one frequency")
    mag, ph = [], []
    for f in fr:
        w = 2.0 * pi * f / fs
        # sum_{k=0}^{5} z^-k with z = exp(jw)
        sr = fsum(cos(-w * k) for k in range(6))
        si = fsum(sin(-w * k) for k in range(6))
        # square it, then scale by 1/32
        re = (sr * sr - si * si) / 32.0
        im = (2.0 * sr * si) / 32.0
        mag.append(hypot(re, im))
        ph.append(atan2(im, re))
    return RichResult(payload={
        "freq": fr,
        "mag": mag,
        "phase": ph,
        "b": [1.0 / 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0 / 32.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 32.0],
        "a": [1.0, -2.0, 1.0],
        "fs": fs,
        "fsnote": "integer coefficients designed for fs = 200 Hz (fc = 11 Hz, 5-sample delay)",
        "method": "Pan-Tompkins lowpass transfer function, Rangayyan (2024) Eq 4.7",
    })


rangayyan_ch4_pan_tompkins_lowpass_transfer = qrslpasstf  # pre-policy spelling


# -- rng182: Difference equation of the Pan-Tompkins lowpass filter..
def qrslpassdf(x):
    """Recursive form of the Pan-Tompkins lowpass filter.

    y(n) = 2 y(n-1) - y(n-2) + (1/32) [x(n) - 2 x(n-6) + x(n-12)],
    Rangayyan (2024) Equation 4.8.  This is Equation 4.7 realised directly;
    the only arithmetic is adds and one shift by 32, which is why the filter
    was chosen for real-time use.  fc = 11 Hz at fs = 200 Hz, delay 5 samples.

    Parameters
    ----------
    x : sequence of float
        ECG samples.  Zero initial conditions are assumed.

    Returns
    -------
    RichResult
        payload["y"], the filtered signal, same length as x.
    """
    x = _rgcheck(x, 1, "ECG")
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        y[i] = (
            (2.0 * y[i - 1] if i >= 1 else 0.0)
            - (y[i - 2] if i >= 2 else 0.0)
            + (x[i] - (2.0 * x[i - 6] if i >= 6 else 0.0)
               + (x[i - 12] if i >= 12 else 0.0)) / 32.0
        )
    return RichResult(payload={
        "y": y,
        "delay": 5,
        "n": n,
        "fsnote": "coefficients fixed for fs = 200 Hz",
        "method": "Pan-Tompkins lowpass difference equation, Rangayyan (2024) Eq 4.8",
    })


rangayyan_ch4_pan_tompkins_lowpass_difference_eq = qrslpassdf  # pre-policy spelling


# -- rng183: Lowpass component of the Pan-Tompkins highpass filter..
def qrshplptf(freq, fs=200.0):
    """Lowpass component from which the Pan-Tompkins highpass is built.

    Hlp(z) = (1 - z^-32) / (1 - z^-1), Rangayyan (2024) Equation 4.9.  This is
    a running sum of 32 samples (an unnormalised 32-point moving average); the
    highpass of Equation 4.11 is formed as allpass minus this, scaled by 1/32.
    Evaluated as sum_{k=0}^{31} z^-k so the z = 1 point is exact.

    Parameters
    ----------
    freq : float or sequence of float
        Frequencies in Hz.
    fs : float
        Sampling rate; the design rate is 200 Hz.

    Returns
    -------
    RichResult
        payload["mag"], payload["phase"], and payload["b"] (32 unit taps).
    """
    fs = _rgfs(fs)
    fr = aslist(freq)
    if not fr:
        raise ValueError("need at least one frequency")
    mag, ph = [], []
    for f in fr:
        w = 2.0 * pi * f / fs
        re = fsum(cos(-w * k) for k in range(32))
        im = fsum(sin(-w * k) for k in range(32))
        mag.append(hypot(re, im))
        ph.append(atan2(im, re))
    return RichResult(payload={
        "freq": fr,
        "mag": mag,
        "phase": ph,
        "b": [1.0] * 32,
        "fs": fs,
        "method": "lowpass component of the Pan-Tompkins highpass, Rangayyan (2024) Eq 4.9",
    })


rangayyan_ch4_pan_tompkins_highpass_lp_component = qrshplptf  # pre-policy spelling


# -- rng184: Difference equation of the lowpass component used in the Pan-Tompkins highpass filter..
def qrshplpdf(x):
    """Recursive running sum used by the Pan-Tompkins highpass.

    y(n) = y(n-1) + x(n) - x(n-32), Rangayyan (2024) Equation 4.10.  One add
    and one subtract per sample regardless of the window length, which is why
    the 32-point sum is realised recursively rather than as an FIR convolution.

    Parameters
    ----------
    x : sequence of float
        Input samples, zero initial conditions.

    Returns
    -------
    RichResult
        payload["y"], same length as x.
    """
    x = _rgcheck(x, 1, "signal")
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        y[i] = (y[i - 1] if i >= 1 else 0.0) + x[i] - (x[i - 32] if i >= 32 else 0.0)
    return RichResult(payload={
        "y": y,
        "n": n,
        "method": "recursive 32-point running sum, Rangayyan (2024) Eq 4.10",
    })


rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq = qrshplpdf  # pre-policy spelling


# -- rng185: Transfer function of the Pan-Tompkins highpass filter..
def qrshpasstf(freq, fs=200.0):
    """Transfer function of the Pan-Tompkins highpass filter.

    Hhp(z) = z^-16 - (1/32) Hlp(z), Rangayyan (2024) Equation 4.11, with
    Hlp from Equation 4.9.  The filter is an allpass (a pure 16-sample delay)
    minus a scaled lowpass, so it needs no separate highpass design and shares
    the running sum already computed.  Cutoff 5 Hz, delay 80 ms at fs = 200 Hz.

    Parameters
    ----------
    freq : float or sequence of float
        Frequencies in Hz.
    fs : float
        Sampling rate; the design rate is 200 Hz.

    Returns
    -------
    RichResult
        payload["mag"], payload["phase"].
    """
    fs = _rgfs(fs)
    fr = aslist(freq)
    if not fr:
        raise ValueError("need at least one frequency")
    mag, ph = [], []
    for f in fr:
        w = 2.0 * pi * f / fs
        lre = fsum(cos(-w * k) for k in range(32)) / 32.0
        lim = fsum(sin(-w * k) for k in range(32)) / 32.0
        re = cos(-w * 16) - lre
        im = sin(-w * 16) - lim
        mag.append(hypot(re, im))
        ph.append(atan2(im, re))
    return RichResult(payload={
        "freq": fr,
        "mag": mag,
        "phase": ph,
        "fs": fs,
        "fsnote": "fc = 5 Hz and 80 ms delay hold at fs = 200 Hz",
        "method": "Pan-Tompkins highpass transfer function, Rangayyan (2024) Eq 4.11",
    })


rangayyan_ch4_pan_tompkins_highpass_transfer = qrshpasstf  # pre-policy spelling


# -- rng186: Difference equation of the Pan-Tompkins highpass filter (intermediate)..
def qrshpassdf(x):
    """Intermediate difference equation of the Pan-Tompkins highpass.

    p(n) = x(n-16) - (1/32) [y(n-1) + x(n) - x(n-32)],
    Rangayyan (2024) Equation 4.12, where y is the running sum of Equation
    4.10.  The bracketed group is exactly y(n), so this form reuses the
    running-sum state directly.

    Parameters
    ----------
    x : sequence of float
        Input samples (in the full detector, the lowpass output).

    Returns
    -------
    RichResult
        payload["p"] (the highpass output) and payload["y"] (the running sum).
    """
    x = _rgcheck(x, 1, "signal")
    n = len(x)
    y = [0.0] * n
    p = [0.0] * n
    for i in range(n):
        y[i] = (y[i - 1] if i >= 1 else 0.0) + x[i] - (x[i - 32] if i >= 32 else 0.0)
        p[i] = (x[i - 16] if i >= 16 else 0.0) - (
            (y[i - 1] if i >= 1 else 0.0) + x[i] - (x[i - 32] if i >= 32 else 0.0)
        ) / 32.0
    return RichResult(payload={
        "p": p,
        "y": y,
        "n": n,
        "method": "Pan-Tompkins highpass difference equation, Rangayyan (2024) Eq 4.12",
    })


rangayyan_ch4_pan_tompkins_highpass_difference_eq = qrshpassdf  # pre-policy spelling


# -- rng187: Combined input-output relationship of the Pan-Tompkins highpass filter..
def qrshpassio(x):
    """Combined input-output relation of the Pan-Tompkins highpass filter.

    p(n) = p(n-1) - (1/32) x(n) + x(n-16) - x(n-17) + (1/32) x(n-32),
    Rangayyan (2024) Equation 4.13.  This folds Equations 4.9 to 4.12 into a
    single recursion, so the whole highpass costs four adds per sample and
    carries only one state variable.

    Parameters
    ----------
    x : sequence of float
        Input samples (in the full detector, the lowpass output of Eq 4.8).

    Returns
    -------
    RichResult
        payload["p"], same length as x.
    """
    x = _rgcheck(x, 1, "signal")
    n = len(x)
    p = [0.0] * n
    for i in range(n):
        p[i] = (
            (p[i - 1] if i >= 1 else 0.0)
            - x[i] / 32.0
            + (x[i - 16] if i >= 16 else 0.0)
            - (x[i - 17] if i >= 17 else 0.0)
            + ((x[i - 32] / 32.0) if i >= 32 else 0.0)
        )
    return RichResult(payload={
        "p": p,
        "n": n,
        "delayms": 80.0,
        "fsnote": "the 80 ms delay and 5 Hz cutoff hold at fs = 200 Hz",
        "method": "combined Pan-Tompkins highpass relation, Rangayyan (2024) Eq 4.13",
    })


rangayyan_ch4_pan_tompkins_highpass_combined = qrshpassio  # pre-policy spelling


# -- rng188: Derivative operator used by Pan and Tompkins for QRS detection..
def qrsderivop(x):
    """Five-point derivative operator used by the Pan-Tompkins detector.

    y(n) = (1/8) [2 x(n) + x(n-1) - x(n-3) - 2 x(n-4)],
    Rangayyan (2024) Equation 4.14.  It approximates d/dt up to about 30 Hz,
    so it gives the QRS a large output while suppressing the low-frequency P
    and T waves, and its antisymmetric taps make it exactly zero on any
    constant or linear baseline.

    Parameters
    ----------
    x : sequence of float
        Input samples (in the full detector, the bandpass output).

    Returns
    -------
    RichResult
        payload["y"], same length as x.
    """
    x = _rgcheck(x, 1, "signal")
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        y[i] = (
            2.0 * x[i]
            + (x[i - 1] if i >= 1 else 0.0)
            - (x[i - 3] if i >= 3 else 0.0)
            - 2.0 * (x[i - 4] if i >= 4 else 0.0)
        ) / 8.0
    return RichResult(payload={
        "y": y,
        "b": [2.0 / 8.0, 1.0 / 8.0, 0.0, -1.0 / 8.0, -2.0 / 8.0],
        "n": n,
        "fsnote": "linear up to about 30 Hz at fs = 200 Hz",
        "method": "Pan-Tompkins derivative operator, Rangayyan (2024) Eq 4.14",
    })


rangayyan_ch4_pan_tompkins_derivative_operator = qrsderivop  # pre-policy spelling


# -- rng189: Moving-window integrator used in the Pan-Tompkins QRS detector..
def qrsmwint(x, nwin=30, fs=None):
    """Moving-window integrator of the Pan-Tompkins detector.

    y(n) = (1/N) {x[n-(N-1)] + ... + x(n)}, Rangayyan (2024) Equation 4.15.
    It turns the multi-peaked squared derivative into one trapezoidal pulse
    per beat of total width W + QS, so a single peak search finds each QRS.
    N matters: too wide merges QRS and T, too narrow leaves multiple peaks.
    The book found N = 30 samples suitable for fs = 200 Hz, i.e. 150 ms.

    Parameters
    ----------
    x : sequence of float
        Squared-derivative signal.
    nwin : int
        Window width N in samples.  Ignored when fs is given.
    fs : float, optional
        If supplied, N is set to round(0.150 * fs) so the 150 ms window of the
        book is preserved at any sampling rate.

    Returns
    -------
    RichResult
        payload["y"] and the window width actually used.
    """
    x = _rgcheck(x, 1, "signal")
    if fs is not None:
        nwin = max(1, int(round(0.150 * _rgfs(fs))))
    nwin = int(nwin)
    if nwin < 1:
        raise ValueError("nwin must be >= 1")
    return RichResult(payload={
        "y": _rgmavg(x, nwin),
        "nwin": nwin,
        "widthsec": (nwin / float(fs)) if fs is not None else None,
        "n": len(x),
        "method": "Pan-Tompkins moving-window integrator, Rangayyan (2024) Eq 4.15",
    })


rangayyan_ch4_pan_tompkins_moving_window_integrator = qrsmwint  # pre-policy spelling


# -- rng191: Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm..
def qrsthresh(peaki, spki, npki, issignal):
    """Adaptive signal/noise levels and thresholds of the Pan-Tompkins detector.

    SPKI = 0.125 PEAKI + 0.875 SPKI for a signal peak, NPKI = 0.125 PEAKI +
    0.875 NPKI for a noise peak (Rangayyan (2024) Equation 4.16), and
    THRESHOLD I1 = NPKI + 0.25 (SPKI - NPKI),
    THRESHOLD I2 = 0.5 THRESHOLD I1 (Equation 4.17).

    The 0.125/0.875 split is a first-order recursive average with a long
    memory, so the running estimates track slow amplitude drift without being
    thrown by a single artifact.  I1 sits a quarter of the way from the noise
    level up to the signal level; I2 is half of I1 and is the looser threshold
    reserved for search-back.

    Parameters
    ----------
    peaki : float
        Amplitude of the newly detected peak in the integrator output.
    spki, npki : float
        Current running signal and noise peak estimates.
    issignal : bool
        True when the peak was classified as a QRS.

    Returns
    -------
    RichResult
        payload["spki"], payload["npki"], payload["thresh1"], payload["thresh2"].
    """
    peaki = float(peaki)
    spki = float(spki)
    npki = float(npki)
    if issignal:
        spki = 0.125 * peaki + 0.875 * spki
    else:
        npki = 0.125 * peaki + 0.875 * npki
    t1 = npki + 0.25 * (spki - npki)
    return RichResult(payload={
        "spki": spki,
        "npki": npki,
        "thresh1": t1,
        "thresh2": 0.5 * t1,
        "peaki": peaki,
        "issignal": bool(issignal),
        "method": "Pan-Tompkins adaptive thresholds, Rangayyan (2024) Eqs 4.16 and 4.17",
    })


rangayyan_ch4_pan_tompkins_thresholds = qrsthresh  # pre-policy spelling


# -- rng192: Updated SPKI rule when a QRS is detected in the search-back procedure..
def qrsspkiupd(peaki, spki):
    """SPKI update rule when a QRS is recovered by search-back.

    SPKI = 0.25 PEAKI + 0.75 SPKI, Rangayyan (2024) Equation 4.18, replacing
    the 0.125/0.875 rule of Equation 4.16.  A beat found only by search-back
    was missed by the primary threshold, which means the running signal
    estimate is too high; the heavier weight on the new peak pulls it down
    faster so the detector re-locks instead of missing the next beats too.

    Parameters
    ----------
    peaki : float
        Amplitude of the peak recovered by search-back.
    spki : float
        Current running signal peak estimate.

    Returns
    -------
    RichResult
        payload["spki"], the updated estimate.
    """
    peaki = float(peaki)
    spki = float(spki)
    new = 0.25 * peaki + 0.75 * spki
    return RichResult(payload={
        "spki": new,
        "previous": spki,
        "peaki": peaki,
        "method": "Pan-Tompkins search-back SPKI update, Rangayyan (2024) Eq 4.18",
    })


rangayyan_ch4_pan_tompkins_searchback_update = qrsspkiupd  # pre-policy spelling


# -- rng193: Heart rate computed from number of QRS complexes detected over duration T..
def hrfromcnt(nbeats, duration):
    """Average heart rate from a QRS count over a known duration.

    HR = 60 NB / T bpm, Rangayyan (2024) Equation 4.19, where NB is the number
    of QRS complexes detected over T seconds.  This is the counting estimate,
    which averages over the whole window and so is insensitive to a single
    mis-detected interval, unlike the beat-to-beat form of Equation 4.20.

    Parameters
    ----------
    nbeats : int
        Number of QRS complexes detected.
    duration : float
        Observation duration T in seconds; must be positive.

    Returns
    -------
    RichResult
        payload["hr"] in bpm.
    """
    nbeats = int(nbeats)
    duration = float(duration)
    if nbeats < 0:
        raise ValueError("nbeats must be non-negative")
    if not duration > 0.0:
        raise ValueError("duration must be positive")
    return RichResult(payload={
        "hr": 60.0 * nbeats / duration,
        "nbeats": nbeats,
        "duration": duration,
        "method": "average heart rate from beat count, Rangayyan (2024) Eq 4.19",
    })


rangayyan_ch4_heart_rate_from_count = hrfromcnt  # pre-policy spelling


# -- rng194: Heart rate from RR interval.
def rangayyan_ch4_heart_rate_from_rr(RR_a):
    r"""Instantaneous heart rate from the RR interval (Rangayyan
    Ch. 4):

    .. math:: HR = \frac{60}{RR_a},

    with RR in seconds and HR in beats per minute. Vectorised, so a
    series of RR intervals gives the instantaneous rate at each beat;
    the mean of those is NOT the same as 60 / mean(RR) (Jensen), and
    both are returned.

    Parameters
    ----------
    RR_a : float or array-like
        RR interval(s) in seconds, strictly positive.

    Returns
    -------
    RichResult
        keys: ``heart_rate``, ``mean_instantaneous_hr``,
        ``hr_from_mean_rr``, ``n``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4.
    """
    rr = np.atleast_1d(np.asarray(RR_a, dtype=float))
    if np.any(rr <= 0):
        raise ValueError("RR intervals must be strictly positive.")
    hr = 60.0 / rr
    scalar = np.ndim(RR_a) == 0
    return RichResult(
        payload={"heart_rate": float(hr[0]) if scalar else hr,
                 "mean_instantaneous_hr": float(np.mean(hr)),
                 "hr_from_mean_rr": float(60.0 / np.mean(rr)), "n": int(rr.size),
                 "method": "HR = 60/RR; mean of rates != rate of mean interval"})


# -- rng195: Length transformation used to detect P, QRS, and T waves across multiple ECG channels..
def lengthxfm(chans, wwin, fs):
    """Length transformation for multichannel P, QRS and T wave detection.

    L(N, w, t) = integral_t^{t+w} sqrt( sum_j (dx_j/dt)^2 ) dt,
    Rangayyan (2024) Equation 4.21 (Gritzali et al.).  Discretely, since
    (dx/dt) dt = dx, this is the arc length of the multichannel trajectory
    accumulated over a window of w seconds:
    L(n) = sum_{k=n}^{n+W-1} sqrt( sum_j (x_j(k+1) - x_j(k))^2 ).

    Summing the squared derivative across channels before integrating is the
    whole point: a P or T wave that is well defined in only one lead still
    contributes, so waves invisible in most leads are still detected.  The
    book sets w to the average width of the wave being sought (QRS first, then
    T, then P, blanking each wave to baseline before the next pass).

    Parameters
    ----------
    chans : sequence of sequences of float
        One sequence per ECG channel, all of the same length.  A single flat
        sequence is accepted and treated as one channel.
    wwin : float
        Window width w in seconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    RichResult
        payload["length"], the transformation output, one value per sample
        position (right-padded with zeros where the window runs past the end).
    """
    fs = _rgfs(fs)
    wwin = float(wwin)
    if not wwin > 0.0:
        raise ValueError("wwin must be positive")
    if not chans:
        raise ValueError("need at least one channel")
    first = chans[0]
    if not hasattr(first, "__len__") and not hasattr(first, "__iter__"):
        chans = [chans]
    elif isinstance(first, (int, float)):
        chans = [chans]
    ch = [_rgcheck(c, 2, "channel") for c in chans]
    nlen = len(ch[0])
    if any(len(c) != nlen for c in ch):
        raise ValueError("all channels must have the same length")
    step = [
        sqrt(fsum((c[k + 1] - c[k]) ** 2 for c in ch))
        for k in range(nlen - 1)
    ]
    wsamp = max(1, int(round(wwin * fs)))
    out = [0.0] * nlen
    run = 0.0
    for k in range(len(step)):
        run += step[k]
        if k >= wsamp:
            run -= step[k - wsamp]
        start = max(0, k - wsamp + 1)
        out[start] = run
    return RichResult(payload={
        "length": out,
        "nchan": len(ch),
        "wsamp": wsamp,
        "wsec": wwin,
        "fs": fs,
        "n": nlen,
        "method": "length transformation, Rangayyan (2024) Eq 4.21 (Gritzali et al.)",
    })


rangayyan_ch4_length_transformation = lengthxfm  # pre-policy spelling


# -- rng196: Noncausal least-squares second derivative used to detect the dicrotic notch.
_COEF = np.array([2.0, -1.0, -2.0, -1.0, 2.0])  # taps for y(n-2)..y(n+2)


def rangayyan_ch4_dicrotic_notch_second_derivative(y, causal=False):
    r"""Lehner-Rangayyan least-squares second derivative.

    .. math:: p(n) = 2y(n-2) - y(n-1) - 2y(n) - y(n+1) + 2y(n+2)

    The five-tap least-squares estimate of the second derivative of the
    carotid pulse. It is deliberately noncausal (it looks two samples
    ahead); the book notes it "may be made causal by applying a delay
    of two samples", which ``causal=True`` does. The second derivative
    removes the constant downward slope of the carotid pulse and leaves
    the dicrotic notch standing out.

    Parameters
    ----------
    y : array-like, shape (n,)
        Carotid pulse signal, n >= 5.
    causal : bool, default False
        Delay the output by two samples so that ``p[n]`` depends only
        on ``y[..n]``.

    Returns
    -------
    RichResult
        keys: ``p`` (n,, zero-padded at the unusable ends), ``valid``
        (slice of fully-supported indices), ``coefficients``,
        ``causal``, ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Eq. (4.22), p. 228 (Sec. 4.3.5, detection of the
    dicrotic notch; after Lehner & Rangayyan).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 5:
        raise ValueError(f"need at least 5 samples for the 5-tap estimate, got {n}.")

    core = np.convolve(y, _COEF[::-1], mode="valid")  # p at indices 2 .. n-3
    p = np.zeros(n)
    if causal:
        p[4:] = core  # two-sample delay: p[n] uses y[n-4..n]
        valid = slice(4, n)
    else:
        p[2 : n - 2] = core
        valid = slice(2, n - 2)

    return RichResult(
        payload={
            "p": p,
            "valid": valid,
            "coefficients": _COEF.copy(),
            "causal": bool(causal),
            "n": int(n),
            "method": "Lehner-Rangayyan LS second derivative (Rangayyan Eq. 4.22, p. 228)",
        }
    )


# -- rng197: Squared and weighted smoothing of the second derivative for dicrotic notch detection..
def dnotchsmth(p, mwin=16):
    """Squared and linearly weighted smoothing of the carotid second derivative.

    s(n) = sum_{k=1}^{M} p^2(n-k+1) w(k) with w(k) = (M - k + 1),
    Rangayyan (2024) Equation 4.23 (Lehner and Rangayyan), where p is the
    least-squares second derivative of Equation 4.22.  Squaring discards the
    sign of the curvature and the linearly decaying weight smooths the result
    while keeping the response prompt at the newest sample.  The book used
    M = 16 at fs = 256 Hz and notes M should be raised for smoother output.

    s(n) has two peaks per cardiac cycle: the first is the carotid upstroke
    onset, the second is the dicrotic notch.

    Parameters
    ----------
    p : sequence of float
        Second-derivative signal from Equation 4.22.
    mwin : int
        Window width M in samples.

    Returns
    -------
    RichResult
        payload["s"] and payload["weights"].
    """
    p = _rgcheck(p, 1, "second derivative")
    mwin = int(mwin)
    if mwin < 1:
        raise ValueError("mwin must be >= 1")
    w = [float(mwin - k + 1) for k in range(1, mwin + 1)]
    n = len(p)
    s = [0.0] * n
    for i in range(n):
        s[i] = fsum(
            p[i - k + 1] ** 2 * w[k - 1]
            for k in range(1, mwin + 1)
            if 0 <= i - k + 1
        )
    return RichResult(payload={
        "s": s,
        "weights": w,
        "mwin": mwin,
        "n": n,
        "method": "squared weighted smoothing for dicrotic notch, Rangayyan (2024) Eq 4.23 (Lehner and Rangayyan)",
    })


rangayyan_ch4_dicrotic_notch_smoothed_squared = dnotchsmth  # pre-policy spelling


_CHEATSHEET = [
    'baseline-wander removal, eqs. (3.132)-(3.133)',
    'rgcpulse: Carotid pulse waveform feature extraction.',
    'rgderqrs: Derivative-based QRS detection (first and second differences).',
    'rgdnot: Dicrotic notch detection in carotid pulse waveform.',
    'rgecgemu: ECG-EMG coupling during physical effort (VMG correlation).',
    'rgecgf: ECG waveform feature extraction (P, QRS, T amplitudes and durations).',
    'rgecgwvf: ECG waveform analysis for ischemia and bundle branch block.',
    'rgexecg: Exercise ECG analysis: ST deviation, slope, and ischemia detection.',
    'rghrvf: HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio.',
    'rghrvt: HRV time-domain metrics: SDNN, RMSSD, pNN50.',
    'rghsnd: Heart sound (S1/S2) identification via PCG-ECG timing.',
    'rgmatefp: Maternal ECG filtering from abdominal ECG recording.',
    'rgmtnart: Motion artifact detection and removal from ECG/PPG.',
    'rgpantp: Pan-Tompkins QRS detection algorithm.',
    'rgpcg: PCG segmentation into S1/systole/S2/diastole using ECG gating.',
    'rgpowerl: Powerline interference (50/60 Hz) removal from ECG.',
    'rgppg: PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak).',
    'rgpwave: P-wave detection in ECG using search window relative to R-peak.',
    'rgrpsig: ECG-derived respiration (EDR) via R-wave amplitude modulation.',
    'rgsapn: Sleep apnea detection via ECG-derived respiration + SpO2 fusion.',
    'rgspr: Spectral power ratio (LF/HF) for HRV analysis.',
    'rgtwa: T-wave alternans (TWA) detection via spectral method.',
    'rgtwamx: T-wave alternans spectral method.',
    'rgtwave: T-wave detection in ECG.',
    'rgvf: Ventricular fibrillation (VF) detection in ECG.',
    'rng176: Smoothed three-point first derivative used in QRS detection (Balda et al.)..',
    'rng177: Approximation of the second derivative used in QRS detection..',
    'rng178: Weighted combination of first and second derivatives for QRS detection..',
    'rng179: Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj)..',
    'rng180: MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector..',
    'rng181: Lowpass transfer function used in the Pan-Tompkins QRS detector..',
    'rng182: Difference equation of the Pan-Tompkins lowpass filter..',
    'rng183: Lowpass component of the Pan-Tompkins highpass filter..',
    'rng184: Difference equation of the lowpass component used in the Pan-Tompkins highpass filter..',
    'rng185: Transfer function of the Pan-Tompkins highpass filter..',
    'rng186: Difference equation of the Pan-Tompkins highpass filter (intermediate)..',
    'rng187: Combined input-output relationship of the Pan-Tompkins highpass filter..',
    'rng188: Derivative operator used by Pan and Tompkins for QRS detection..',
    'rng189: Moving-window integrator used in the Pan-Tompkins QRS detector..',
    'rng191: Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm..',
    'rng192: Updated SPKI rule when a QRS is detected in the search-back procedure..',
    'rng193: Heart rate computed from number of QRS complexes detected over duration T..',
    'rng194: Heart rate from RR interval.',
    'rng195: Length transformation used to detect P, QRS, and T waves across multiple ECG channels..',
    'rng196: Noncausal least-squares second derivative used to detect the dicrotic notch.',
    'rng197: Squared and weighted smoothing of the second derivative for dicrotic notch detection..',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
