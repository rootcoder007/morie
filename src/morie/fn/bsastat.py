# morie.fn -- bsastat (rootcoder007/morie)
"""Statistics of random processes: moments, entropy, covariance, signal-level and fractal features.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 36
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import cos, fsum, log, log10, pi, sin, sqrt
from math import cos, fsum, log, pi, sin, sqrt
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
    'corrcoef',
    'rangayyan_correlation_coeff',
    'specentropy',
    'rangayyan_spectral_entropy',
    'fdpsd',
    'rangayyan_fd_psd_slope',
    'formfactor',
    'rangayyan_form_factor',
    'fdvag',
    'rangayyan_fractal_vag',
    'rangayyan_higuchi_fd',
    'katzfd',
    'rangayyan_katz_fd',
    'firingrate',
    'rangayyan_muap_firing_rate',
    'nlfeatures',
    'rangayyan_nonlinear_features',
    'rangayyan_pdf_estimate',
    'rms',
    'rangayyan_rms',
    'rangayyan_rms_noise',
    'syncavg',
    'rangayyan_sync_average',
    'sigfeatures',
    'rangayyan_signal_features',
    'snrfilt',
    'rangayyan_signal_to_noise',
    'snr',
    'rangayyan_snr',
    'turnscount',
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
    'obsreal',
    'rangayyan_ch3_observed_signal_kth_realization',
]



# -- rgcorec: Pearson correlation coefficient for morphological analysis.
def corrcoef(x, y):
    """Pearson correlation coefficient.

        r = sum (x - mx)(y - my)
            / sqrt( sum (x - mx)^2 sum (y - my)^2 )

    The MEANS ARE REMOVED, which is what separates this from the
    normalized dot product of Chapter 4: r measures linear association
    about the means, that one measures the angle between the raw vectors.
    For signals with a large common offset the two differ sharply, and
    both are returned so the difference is visible.

    r is invariant to any positive affine change of either variable, so
    it says nothing about scale or agreement -- only about how tightly
    the points hug a straight line.
    """
    from .bsacorr import dotprod          # eqs (4.24)-(4.25), one copy

    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two paired observations")
    centred = dotprod(xs, ys, subtract_mean=True)
    if centred["gamma"] is None:
        raise ValueError("a variable is constant; the correlation is "
                         "undefined")
    r = centred["gamma"]
    mx, my = fsum(xs) / n, fsum(ys) / n
    return RichResult(payload={
        "r": r, "r_squared": r * r, "n": n, "means": [mx, my],
        "cosine_without_removing_means": dotprod(xs, ys)["gamma"],
        "means_are_removed": True,
        "invariant_to_positive_affine_change": True,
        "says_nothing_about_agreement": True,
        "method": "Pearson correlation; Rangayyan (2024) Ch. 5"})


rangayyan_correlation_coeff = corrcoef  # pre-policy spelling


# -- rgentrp: Spectral entropy for signal complexity measurement.
def specentropy(psd, freqs=None, fmin=None, fmax=None):
    """Spectral entropy: Shannon entropy of the normalized PSD.

    The PSD is normalized to sum to one and read as a probability mass
    over frequency, then eq. (3.11) of Rangayyan (2024) is applied:

        p_k = S(f_k) / sum_j S(f_j),   H = - sum_k p_k log2 p_k.

    A flat spectrum gives the maximum log2(K) bits; a pure tone gives
    zero.  ``normalized`` divides by that maximum so records of different
    length are comparable.

    Rangayyan (2024) defines the spectral moments of Section 6.4.4 --
    mean frequency, bandwidth, skewness, kurtosis of the PSD -- but does
    not print a spectral-entropy equation, so this is eq. (3.11) applied
    to the PSD rather than a formula quoted from the book.
    """
    p = aslist(psd)
    if not p:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in p):
        raise ValueError("a PSD cannot be negative")
    if freqs is not None:
        f = aslist(freqs)
        if len(f) != len(p):
            raise ValueError("psd and freqs must have the same length")
        keep = [(a, b) for a, b in zip(f, p)
                if (fmin is None or a >= fmin)
                and (fmax is None or a <= fmax)]
        if not keep:
            raise ValueError("the band retains no bins")
        p = [b for _, b in keep]
    total = fsum(p)
    if total <= 0:
        raise ValueError("the PSD has zero total power")
    probs = [v / total for v in p]
    ln2 = log(2.0)
    h = -fsum(q * log(q) / ln2 for q in probs if q > 0)
    k = len(probs)
    hmax = log(k) / ln2 if k > 1 else 0.0
    return RichResult(payload={
        "entropy": h, "units": "bits", "max_entropy": hmax,
        "normalized": (h / hmax) if hmax > 0 else 0.0,
        "n_bins": k, "probabilities": probs,
        "method": "Rangayyan (2024) eq. (3.11) applied to the PSD"})


rangayyan_spectral_entropy = specentropy  # pre-policy spelling


# -- rgfdpsd: Fractal dimension from PSD slope (1/f noise model).
def fdpsd(psd, freqs, fmin=None, fmax=None):
    """Fractal dimension from the slope of the PSD on log-log axes.

    Rangayyan (2024) Section 6.6.2, eqs. (6.50)-(6.52).  An fBm signal
    has PSD proportional to 1/f^beta, and for a 1-D signal (E = 1)

        H  = (beta - 1) / 2                                       (6.51)
        FD = (5 - beta) / 2                                       (6.52)

    with the Hurst coefficient H in [0, 1] and FD in [1, 2].  beta is the
    NEGATIVE of the least-squares slope of log10 P against log10 f.

    The DC bin is excluded: log(0) is undefined and the DC term carries
    the mean, not the scaling.  ``in_range`` records whether the fitted
    beta lands in [0.5, 1.5], the interval the book cites from Voss for
    most natural phenomena -- a beta far outside it usually means the
    band was chosen badly rather than that the signal is exotic.
    """
    p, f = aslist(psd), aslist(freqs)
    if len(p) != len(f):
        raise ValueError("psd and freqs must have the same length")
    pts = [(a, b) for a, b in zip(f, p) if a > 0 and b > 0
           and (fmin is None or a >= fmin)
           and (fmax is None or a <= fmax)]
    if len(pts) < 3:
        raise ValueError("need at least three positive-frequency bins in "
                         "the band to fit a slope")
    lx = [log10(a) for a, _ in pts]
    ly = [log10(b) for _, b in pts]
    n = len(lx)
    mx, my = fsum(lx) / n, fsum(ly) / n
    sxx = fsum((v - mx) ** 2 for v in lx)
    if sxx <= 0:
        raise ValueError("all retained bins share one frequency")
    slope = fsum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    beta = -slope
    ss_tot = fsum((v - my) ** 2 for v in ly)
    inter = my - slope * mx
    ss_res = fsum((b - (inter + slope * a)) ** 2 for a, b in zip(lx, ly))
    return RichResult(payload={
        "fd": (5.0 - beta) / 2.0, "beta": beta, "hurst": (beta - 1.0) / 2.0,
        "slope": slope, "intercept": inter, "n_bins": n,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "in_range": 0.5 <= beta <= 1.5,
        "band": (pts[0][0], pts[-1][0]),
        "method": "Rangayyan (2024) eqs. (6.50)-(6.52)"})


rangayyan_fd_psd_slope = fdpsd  # pre-policy spelling


# -- rgff: Form factor (ratio of RMS to mean absolute value).
def formfactor(x):
    """Hjorth's activity, mobility, and form factor (complexity).

    Rangayyan (2024) Section 5.6.4, eqs. (5.25)-(5.26):
        activity = sigma_x^2
        mobility M_x = sigma_{x'} / sigma_x                       (5.25)
        form factor FF = M_{x'} / M_x
                       = (sigma_{x''} / sigma_{x'})
                         / (sigma_{x'} / sigma_x)                 (5.26)

    where x' and x'' are the first and second derivatives.  The book
    states the complexity of a sinusoid is unity and that other
    waveforms give larger values as their variation increases.

    The placeholder this replaces defined FF as RMS(x) / mean(|x|),
    which is the crest-factor family of measures, not Hjorth's.  For a
    sinusoid that ratio is pi / (2 sqrt(2)) = 1.111, not 1 -- the two
    definitions do not even agree on the book's stated reference value.

    Derivatives are taken as first differences, which is what a sampled
    signal admits; the ratios in eq. (5.26) are dimensionless, so the
    sampling interval cancels and no fs is needed.
    """
    xs = aslist(x)
    if len(xs) < 4:
        raise ValueError("need at least four samples for a second derivative")

    def var(v):
        mu = fsum(v) / len(v)
        return fsum((u - mu) ** 2 for u in v) / len(v)

    d1 = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    a0, a1, a2 = var(xs), var(d1), var(d2)
    if a0 <= 0:
        raise ValueError("a constant signal has zero activity; mobility and "
                         "form factor are undefined")
    mob = sqrt(a1 / a0)
    if a1 <= 0:
        raise ValueError("the first derivative is constant; the form factor "
                         "is undefined")
    mob1 = sqrt(a2 / a1)
    return RichResult(payload={
        "form_factor": mob1 / mob, "complexity": mob1 / mob,
        "mobility": mob, "activity": a0,
        "mobility_of_derivative": mob1,
        "n": len(xs), "method": "Rangayyan (2024) eqs. (5.25)-(5.26)"})


rangayyan_form_factor = formfactor  # pre-policy spelling


# -- rgfracv: Fractal analysis of VAG signals via power spectral slope.
def fdvag(x, fs, fmin=100.0, fmax=500.0, nperseg=None):
    """Fractal dimension of a VAG signal via power spectral analysis.

    Rangayyan (2024) Section 6.6.2 calls PSA the best available method
    for the FD of a self-affine signal, and Section 6.6.3 applies it to
    knee-joint vibroarthrographic signals.  The periodogram is fitted
    over a band -- 100-500 Hz by default, where VAG energy sits above
    muscle-contraction interference and below the noise floor -- and the
    slope is converted by eq. (6.52).

    The band is an argument, not a constant, because it is the choice
    that decides the answer: a band that includes the low-frequency
    baseline drift fits the drift's slope instead of the signal's.
    """
    xs = aslist(x)
    if len(xs) < 8:
        raise ValueError("need at least eight samples")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    m = len(xs) if nperseg is None else min(int(nperseg), len(xs))
    seg = xs[:m]
    mu = fsum(seg) / m
    seg = [v - mu for v in seg]
    p, f = [], []
    for k in range(m // 2 + 1):
        ang = -2.0 * pi * k / m
        re = fsum(v * cos(ang * i) for i, v in enumerate(seg))
        im = fsum(v * sin(ang * i) for i, v in enumerate(seg))
        p.append((re * re + im * im) / m)
        f.append(k * fsv / m)
    r = fdpsd(p, f, fmin=fmin, fmax=fmax)
    out = dict(r)
    out["fs"] = fsv
    out["nperseg"] = m
    out["method"] = "Rangayyan (2024) Sections 6.6.2-6.6.3"
    return RichResult(payload=out)


rangayyan_fractal_vag = fdvag  # pre-policy spelling


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
def katzfd(x, dt=1.0):
    """Katz's fractal dimension of a waveform.

    Katz (1988), Computers in Biology and Medicine 18(3):145-156:

        FD = log10(n) / ( log10(n) + log10(d / L) ),

    where L is the total path length of the waveform, d is the greatest
    distance from the first point to any other, and n = L / a is the
    number of steps, a being the mean step length.  Equivalently
    n = L / a with a = L / (N - 1) for a uniformly sampled signal, so
    n = N - 1.

    Rangayyan (2024) Section 5.13.2 covers the ruler method, box counting
    and Higuchi's algorithm; Katz's estimator is not among them, so this
    is cited to Katz rather than to the book.  It is included because it
    is the estimator the module was named for and it is cheap, but note
    it is scale-sensitive: distances mix the amplitude and time axes, so
    rescaling the signal changes the answer.  The book's Section 5.13.2
    warns about exactly this when choosing a ruler size.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 3:
        raise ValueError("need at least three samples")
    step = float(dt)
    if step <= 0:
        raise ValueError("dt must be positive")
    lengths = [sqrt(step * step + (xs[i + 1] - xs[i]) ** 2)
               for i in range(n - 1)]
    total = fsum(lengths)
    if total <= 0:
        raise ValueError("the waveform has zero length")
    d = max(sqrt((i * step) ** 2 + (xs[i] - xs[0]) ** 2) for i in range(n))
    if d <= 0:
        raise ValueError("every point coincides with the first")
    a = total / (n - 1)
    steps = total / a
    denom = log10(steps) + log10(d / total)
    if denom == 0:
        raise ValueError("the Katz ratio is degenerate for this waveform")
    return RichResult(payload={
        "fd": log10(steps) / denom, "total_length": total,
        "max_distance": d, "mean_step": a, "n_steps": steps, "n": n,
        "scale_sensitive": True,
        "method": "Katz (1988); Rangayyan (2024) Section 5.13.2 covers "
                  "the ruler, box-counting and Higuchi methods instead"})


rangayyan_katz_fd = katzfd  # pre-policy spelling


# -- rgmufr: Motor unit mean firing rate and inter-discharge interval (IDI).
def firingrate(times, fs=None):
    """Motor-unit mean firing rate and inter-discharge-interval statistics.

    Rangayyan (2024) Section 4.2 describes temporal recruitment as an
    increase in the frequency of discharge of each motor unit, and the
    exercises in Chapter 5 ask for the firing rate of each detected
    motor unit.  From the discharge instants:

        IDI_k  = t_{k+1} - t_k
        MFR    = 1 / mean(IDI)
        CV_IDI = SD(IDI) / mean(IDI).

    MFR is the RECIPROCAL OF THE MEAN interval, not the mean of the
    reciprocals; those differ whenever the intervals vary (Jensen), and
    the reciprocal-of-mean is the one that equals discharges per unit
    time.  Both are returned so the difference is visible.

    Parameters
    ----------
    times : array-like
        Discharge instants: seconds if ``fs`` is None, otherwise sample
        indices converted with the given sampling rate.
    fs : float, optional
        Sampling rate in Hz.
    """
    ts = aslist(times)
    if len(ts) < 2:
        raise ValueError("need at least two discharges to form an interval")
    if fs is not None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        ts = [v / float(fs) for v in ts]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("discharge instants must be strictly increasing")
    idi = [b - a for a, b in zip(ts, ts[1:])]
    m = fsum(idi) / len(idi)
    sd = sqrt(fsum((v - m) ** 2 for v in idi) / len(idi))
    return RichResult(payload={
        "mfr": 1.0 / m, "mean_idi": m, "sd_idi": sd, "cv_idi": sd / m,
        "idi": idi, "n_discharges": len(ts),
        "mean_instantaneous_rate": fsum(1.0 / v for v in idi) / len(idi),
        "duration": ts[-1] - ts[0],
        "method": "Rangayyan (2024) Sections 4.2, 5.x (motor-unit "
                  "discharge statistics)"})


rangayyan_muap_firing_rate = firingrate  # pre-policy spelling


# -- rgnl: Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov).
def nlfeatures(x, m=2, r=None, dt=1.0):
    """Nonlinear feature vector: ApEn, SampEn, DFA exponent, Lyapunov.

    Rangayyan (2024) Section 5.13 groups fractal and nonlinear measures
    as descriptors of waveform complexity.  Each component here is
    computed by the module that owns it rather than reimplemented:

    - approximate entropy, Pincus (1991)
    - sample entropy, Richman and Moorman (2000)
    - detrended fluctuation analysis exponent, Peng et al. (1994)
    - largest Lyapunov exponent, Rosenstein et al. (1993)

    Only the fractal dimension of Section 5.13.2 is the book's own; the
    other four are cited to their primary sources, which is why they
    are listed by author here.  A component that fails on a given record
    (too few samples, a degenerate embedding) is reported as None with
    its reason, rather than aborting the whole feature vector.
    """
    xs = aslist(x)
    if len(xs) < 16:
        raise ValueError("need at least sixteen samples for these measures")
    out, errs = {}, {}
    parts = [
        ("apen", "rgapn", "rangayyan_approximate_entropy", (), {"m": m, "r": r}),
        ("sampen", "rgsam", "rangayyan_sample_entropy", (), {"m": m, "r": r}),
        ("dfa", "rgdfa", "rangayyan_dfa", (), {}),
        ("lyapunov", "rglyp", "rangayyan_lyapunov", (), {"dt": dt}),
    ]
    import importlib

    for key, mod, fname, args, kw in parts:
        try:
            fn = getattr(importlib.import_module("." + mod, __package__),
                         fname)
            res = fn(xs, *args, **kw)
            out[key] = res
        except Exception as exc:                     # noqa: BLE001
            out[key] = None
            errs[key] = "%s: %s" % (type(exc).__name__, exc)
    return RichResult(payload={
        "features": out, "failures": errs, "n": len(xs),
        "method": "Rangayyan (2024) Section 5.13; components cited to "
                  "Pincus (1991), Richman and Moorman (2000), Peng et al. "
                  "(1994), Rosenstein et al. (1993)"})


rangayyan_nonlinear_features = nlfeatures  # pre-policy spelling


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
def rms(x, window=None):
    """Root-mean-square value of a signal, whole or short-time.

    Rangayyan (2024) eq. (3.9):
        RMS = sqrt( (1/N) sum_{n} x(n)^2 ),

    the average signal level; the divisor is N, not N-1.  With
    ``window`` the same quantity is computed in a causal moving window,
    which is how the book uses it for EMG activity in Section 5.6 and
    Figure 5.10 (a 70 ms window there).
    """
    from .bsastat import srms

    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    whole = srms(xs)
    out = {"rms": whole["rms"], "ms": whole["ms"], "n": len(xs),
           "method": "Rangayyan (2024) eq. (3.9)"}
    if window is not None:
        w = int(window)
        if w < 1:
            raise ValueError("window must be at least one sample")
        short = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            seg = xs[lo:i + 1]
            short.append(sqrt(fsum(v * v for v in seg) / len(seg)))
        out["short_time"] = short
        out["window"] = w
    return RichResult(payload=out)


rangayyan_rms = rms  # pre-policy spelling


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
def syncavg(observations):
    """Synchronized (ensemble) averaging of M aligned realizations.

    Rangayyan (2024) Section 3.5, eqs. (3.95)-(3.96):
        y_k(n) = x_k(n) + eta_k(n)                                (3.95)
        sum_k y_k(n) = sum_k x_k(n) + sum_k eta_k(n)              (3.96)

    If the repetitions are identical and aligned, the signal sum is
    M x(n) while the zero-mean noise sum grows only as sqrt(M) -- so
    dividing by M leaves the signal intact and shrinks the noise SD by
    1/sqrt(M), an SNR gain of sqrt(M), or 10 log10(M) dB.

    The two premises are what make it work and what make it fail, so
    both are reported: ``alignment_note`` records that alignment is the
    caller's responsibility (the book aligns ERPs on the stimulus and
    ECGs on the QRS), and the per-instant SD lets a caller see whether
    the realizations really are repetitions of one signal.
    """
    recs = [aslist(r) for r in observations]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one observation")
    n = len(recs[0])
    if n == 0:
        raise ValueError("records must be nonempty")
    if any(len(r) != n for r in recs):
        raise ValueError("all realizations must have the same length; "
                         "averaging ragged records would average a "
                         "different number of traces at different instants")
    avg = [fsum(r[i] for r in recs) / m for i in range(n)]
    sd = [sqrt(fsum((r[i] - avg[i]) ** 2 for r in recs) / m)
          for i in range(n)]
    return RichResult(payload={
        "average": avg, "sd": sd, "m": m, "n": n,
        "se": [s / sqrt(m) for s in sd],
        "snr_gain": sqrt(m), "snr_gain_db": 10.0 * log10(m),
        "alignment_note": "eqs. (3.95)-(3.96) assume the realizations are "
                          "already aligned; misalignment smears the average",
        "method": "Rangayyan (2024) eqs. (3.95)-(3.96)"})


rangayyan_sync_average = syncavg  # pre-policy spelling


# -- rgsf: Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear.
def sigfeatures(x, fs=1.0, threshold=0.0):
    """Time- and frequency-domain feature vector for a biomedical signal.

    Collects the descriptors Rangayyan (2024) uses across Chapters 3, 5
    and 6, each computed by the function that owns its definition so the
    vector cannot disagree with the individual measures:

    - mean and SD, eqs. (3.7), (3.10)
    - RMS, eq. (3.9)
    - zero-crossing rate, Section 5.6.2
    - turns count, Section 5.6.3 (Willison)
    - Hjorth activity, mobility, form factor, eqs. (5.25)-(5.26)
    - spectral centroid and bandwidth from the periodogram
    - spectral entropy, eq. (3.11) applied to the PSD

    ``threshold`` is passed to the turns count and defaults to 0, which
    counts every direction change: for a real EMG record pass the book's
    100 microvolt threshold, or the count is dominated by noise.
    """
    xs = aslist(x)
    if len(xs) < 8:
        raise ValueError("need at least eight samples")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    n = len(xs)
    mu = fsum(xs) / n
    sd = sqrt(fsum((v - mu) ** 2 for v in xs) / n)
    rms_v = sqrt(fsum(v * v for v in xs) / n)
    zc = sum(1 for i in range(1, n)
             if (xs[i - 1] < 0 <= xs[i]) or (xs[i - 1] >= 0 > xs[i]))
    hj = formfactor(xs)
    seg = [v - mu for v in xs]
    p, f = [], []
    for k in range(n // 2 + 1):
        ang = -2.0 * pi * k / n
        re = fsum(v * cos(ang * i) for i, v in enumerate(seg))
        im = fsum(v * sin(ang * i) for i, v in enumerate(seg))
        p.append((re * re + im * im) / n)
        f.append(k * fsv / n)
    tot = fsum(p)
    centroid = fsum(a * b for a, b in zip(f, p)) / tot if tot > 0 else 0.0
    bw = sqrt(fsum((a - centroid) ** 2 * b for a, b in zip(f, p)) / tot) \
        if tot > 0 else 0.0
    ent = specentropy(p)
    return RichResult(payload={
        "mean": mu, "sd": sd, "rms": rms_v,
        "zero_crossings": zc, "zcr": zc * fsv / n,
        "turns": turnscount(xs, threshold=threshold)["turns"],
        "activity": hj["activity"], "mobility": hj["mobility"],
        "form_factor": hj["form_factor"],
        "spectral_centroid": centroid, "spectral_bandwidth": bw,
        "spectral_entropy": ent["entropy"],
        "n": n, "fs": fsv,
        "method": "Rangayyan (2024) Chapters 3, 5, 6 feature set"})


rangayyan_signal_features = sigfeatures  # pre-policy spelling


# -- rgsig2n: Signal-to-noise ratio calculation after filtering.
def snrfilt(clean, filtered):
    """SNR improvement achieved by a filter.

    The residual is what the filter failed to remove, so with a known
    clean reference the output SNR is

        SNR_out = 10 log10( sum clean^2 / sum (filtered - clean)^2 ),

    the power form of Rangayyan (2024) Section 3.2.1 applied to that
    residual.  Note this measures the DISTORTED output against the truth:
    a filter that removes noise but also smooths the signal is penalised
    for both, which is the honest accounting -- comparing only the noise
    power would flatter an over-smoothing filter.
    """
    c, f = aslist(clean), aslist(filtered)
    if len(c) != len(f):
        raise ValueError("clean and filtered must have the same length")
    if not c:
        raise ValueError("need at least one sample")
    resid = [b - a for a, b in zip(c, f)]
    ps = fsum(v * v for v in c)
    pr = fsum(v * v for v in resid)
    if pr <= 0:
        return RichResult(payload={
            "snr_db": float("inf"), "residual_power": 0.0,
            "signal_power": ps, "residual": resid, "n": len(c),
            "method": "Rangayyan (2024) Section 3.2.1"})
    return RichResult(payload={
        "snr_db": 10.0 * log10(ps / pr), "residual_power": pr,
        "signal_power": ps, "residual": resid, "n": len(c),
        "method": "Rangayyan (2024) Section 3.2.1"})


rangayyan_signal_to_noise = snrfilt  # pre-policy spelling


# -- rgsnr: Signal-to-noise ratio (dB).
def snr(signal, noise, definition="power"):
    """Signal-to-noise ratio in dB.

    Rangayyan (2024), Section 3.2.1, gives two definitions in the same
    sentence, and they are not interchangeable:

    ``definition="power"``
        the ratio of the average power of the signal to that of the
        noise,  SNR = 10 log10( P_signal / P_noise ).
    ``definition="peak"``
        the ratio of the PEAK-TO-PEAK amplitude range of the signal of
        interest to the RMS value of the noise,
        SNR = 20 log10( (max - min) / RMS_noise ).

    The peak form is an amplitude ratio, hence 20 log10, and it runs
    roughly 9 dB above the power form for a sinusoid -- so reporting one
    while the reader assumes the other is a real error, not a rounding
    difference.  Both are returned; ``snr_db`` is the one named.
    """
    s, e = aslist(signal), aslist(noise)
    if not s or not e:
        raise ValueError("both signal and noise need at least one sample")
    ps = fsum(v * v for v in s) / len(s)
    pn = fsum(v * v for v in e) / len(e)
    if pn <= 0:
        raise ValueError("noise power is zero; the SNR is unbounded")
    power_db = 10.0 * log10(ps / pn)
    peak_db = 20.0 * log10((max(s) - min(s)) / sqrt(pn)) \
        if max(s) > min(s) else float("-inf")
    if definition not in ("power", "peak"):
        raise ValueError("definition must be 'power' or 'peak'")
    return RichResult(payload={
        "snr_db": power_db if definition == "power" else peak_db,
        "snr_power_db": power_db, "snr_peak_db": peak_db,
        "signal_power": ps, "noise_power": pn,
        "noise_rms": sqrt(pn), "definition": definition,
        "method": "Rangayyan (2024) Section 3.2.1"})


rangayyan_snr = snr  # pre-policy spelling


# -- rgturns: Turns count of an EMG signal (number of direction reversals above threshold).
def turnscount(x, threshold=100.0, window=None):
    """Willison's turns count.

    Rangayyan (2024) Section 5.6.3, after Willison: a turn is a change in
    the direction (slope) of the signal, and a turn is COUNTED only when
    the excursion since the previous counted turn exceeds a threshold --
    100 microvolts in the book, chosen so that insignificant noise
    fluctuations are not counted.

    The threshold is measured against the last counted turn, not against
    the immediately preceding sample.  That is what separates this from
    simply counting turning points (Section 3.2.1) and is the reason the
    book calls it robust in noise: a small wobble on a long rising edge
    produces turning points but no turns.

    Parameters
    ----------
    x : array-like
        Signal, in the same units as ``threshold``.
    threshold : float
        Minimum excursion for a turn to count; 100 (microvolts) in the
        book's EMG application.
    window : int, optional
        Length of a causal moving window, giving a turns-count series as
        in Figure 5.10 (210 samples there, 70 ms at 3 kHz).
    """
    xs = aslist(x)
    if len(xs) < 3:
        raise ValueError("need at least three samples to have a turn")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    def count(seg):
        if len(seg) < 3:
            return 0, []
        turns, idx = 0, []
        last = seg[0]
        direction = 0
        for i in range(1, len(seg)):
            step = seg[i] - seg[i - 1]
            if step == 0:
                continue
            d = 1 if step > 0 else -1
            if direction == 0:
                direction = d
                continue
            if d != direction:
                # a reversal at sample i-1; count it only if the swing
                # since the last counted turn is large enough
                if abs(seg[i - 1] - last) > threshold:
                    turns += 1
                    idx.append(i - 1)
                    last = seg[i - 1]
                direction = d
        return turns, idx

    total, positions = count(xs)
    out = {"turns": total, "positions": positions, "threshold": threshold,
           "n": len(xs), "method": "Rangayyan (2024) Section 5.6.3"}
    if window is not None:
        w = int(window)
        if w < 3:
            raise ValueError("window must hold at least three samples")
        series = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            series.append(count(xs[lo:i + 1])[0])
        out["short_time"] = series
        out["window"] = w
        out["rate"] = total / len(xs)
    return RichResult(payload=out)


rangayyan_turns_count = turnscount  # pre-policy spelling


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
def obsreal(x, eta):
    """The k-th observed realization of a signal in noise.

    Rangayyan (2024) eq. (3.95):
        y_k(n) = x_k(n) + eta_k(n),

    with k = 1..M the ensemble index and n = 0..N-1 the time index.  This
    is the model synchronized averaging assumes; forming the ensemble
    here and handing it to :func:`syncavg` is what eq. (3.96) then does.

    Parameters
    ----------
    x : array-like or sequence of array-like
        The uncorrupted signal.  A single record is read as the same
        signal repeated for every realization, which is the book's
        "identical and aligned" case.
    eta : sequence of array-like
        The noise in each of the M observations.
    """
    noises = [aslist(e) for e in eta]
    m = len(noises)
    if m == 0:
        raise ValueError("need at least one noise realization")
    n = len(noises[0])
    if any(len(e) != n for e in noises):
        raise ValueError("all noise realizations must have the same length")
    first = aslist(x[0]) if x and hasattr(x[0], "__len__") else aslist(x)
    signals = [aslist(r) for r in x] if x and hasattr(x[0], "__len__") \
        else [first] * m
    if len(signals) != m:
        raise ValueError("give one signal per realization, or one for all")
    if any(len(s) != n for s in signals):
        raise ValueError("signal and noise records must have equal length")
    y = [[s[i] + e[i] for i in range(n)] for s, e in zip(signals, noises)]
    identical = all(all(abs(a - b) < 1e-12 for a, b in zip(s, first))
                    for s in signals)
    return RichResult(payload={
        "y": y, "m": m, "n": n, "identical_repetitions": identical,
        "method": "Rangayyan (2024) eq. (3.95)"})


rangayyan_ch3_observed_signal_kth_realization = obsreal  # pre-policy spelling


_CHEATSHEET = [
    'Pearson correlation coefficient',
    'rgentrp: Spectral entropy for signal complexity measurement.',
    'rgfdpsd: Fractal dimension from PSD slope (1/f noise model).',
    'rgff: Form factor (ratio of RMS to mean absolute value).',
    'rgfracv: Fractal analysis of VAG signals via power spectral slope.',
    'rghfd: Higuchi fractal dimension -- Rangayyan Sec. 5.13.2, eqs (5.39)-(5.41).',
    'rgkatzfd: Katz fractal dimension of a waveform.',
    'rgmufr: Motor unit mean firing rate and inter-discharge interval (IDI).',
    'rgnl: Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov).',
    'rgpdfest: Probability density estimate.',
    'rgrms: Root mean square (RMS) value of a signal.',
    'rgrmsnw: RMS noise level.',
    'rgsavg: Synchronized (ensemble) averaging for SNR enhancement.',
    'rgsf: Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear.',
    'rgsig2n: Signal-to-noise ratio calculation after filtering.',
    'rgsnr: Signal-to-noise ratio (dB).',
    'rgturns: Turns count of an EMG signal (number of direction reversals above threshold).',
    'rgzcr: Zero-crossing rate -- Rangayyan & Krishnan Sec 5.6.2.',
    'rng001: Mean of a random process from its PDF (Rangayyan eq. 3.1).',
    'rng002: Mean-squared value of a random process (Rangayyan eq. 3.2).',
    'rng003: Variance of a random process (Rangayyan eq. 3.3).',
    'rng004: Skewness of a random process (Rangayyan eq. 3.4).',
    'rng005: Kurtosis of a random process (Rangayyan eq. 3.5).',
    'rng006: Differential entropy of a continuous PDF (Rangayyan eq. 3.6).',
    'rng007: Sample mean of an observed signal (Rangayyan eq. 3.7).',
    'rng008: Sample mean square.',
    'rng009: Sample RMS, MS, and SD of an observed signal (Rangayyan eqs. 3.8-3.10).',
    'rng010: Sample standard deviation.',
    'rng012: Additive signal-plus-noise model (Rangayyan eqs. 3.12-3.14).',
    'rng013: Mean of a sum of random processes (Rangayyan eq. 3.13).',
    'rng014: Variance of a sum of two uncorrelated random processes (Rangayyan Eq 3.14).',
    'rng015: Ensemble mean at one instant (Rangayyan eq. 3.15).',
    'rng019: Time-average mean.',
    'rng021: Covariance and correlation coefficient (Rangayyan eqs. 3.21-3.22).',
    'rng022: Correlation coefficient as normalised covariance (Rangayyan Eq 3.22).',
    'rng084: kth observed realization of a signal in noise (signal-plus-noise model)..',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
