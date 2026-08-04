# morie.fn -- bsaar (rootcoder007/morie)
"""Parametric modelling: AR/ARMA, LPC, Levinson-Durbin, Burg, Yule-Walker, pole-zero models, model-order selection.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 20
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import atan2 as _atan2, cos, fsum, log, pi, sin, sqrt
import math as _math
from . import _array_core as np
from ._containers import DescriptiveResult
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from .bsacorr import rangayyan_acf_estimate

__all__ = [
    'burg_psd',
    'rangayyan_ar_order_aic',
    'rangayyan_ar_burg',
    'rangayyan_ar_spectrum',
    'rangayyan_burg_method',
    'fpeorder',
    'rangayyan_ar_order_fpe',
    'hrvratio',
    'rangayyan_hrv_ar_ratio',
    'hrvar',
    'rangayyan_hrv_ar_model',
    'levinson',
    'rangayyan_levinson_durbin',
    'lpc',
    'rangayyan_lpc_analysis',
    'lpcsynth',
    'rangayyan_lpc_synthesis',
    'mdlorder',
    'rangayyan_ar_order_mdl',
    'arfit',
    'rangayyan_parametric_sysid',
    'pcgar',
    'rangayyan_pcg_ar_model',
    'armafit',
    'rangayyan_pole_zero_model',
    'polezero',
    'rangayyan_pole_zero_plot',
    'rangayyan_yule_walker',
    'pzform',
    'rangayyan_ch3_pole_zero_factored_form',
    'pzformz',
    'rangayyan_ch3_pole_zero_factored_form_alt',
    'pzresp',
    'rangayyan_ch3_frequency_response_from_pole_zero',
]

def _angle(z):
    """Principal argument in (-pi, pi], without importing cmath."""
    return _atan2(z.imag, z.real)



# -- burgp: Burg AR spectral estimation.
def burg_psd(
    x: np.ndarray,
    order: int = 16,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    r"""Burg autoregressive spectral estimation.

    Estimates the AR coefficients using the Burg method (minimizes
    forward + backward prediction error simultaneously) and computes
    the power spectral density from the AR model:

    .. math::

        P(f) = \\frac{\\sigma^2}{\\left| 1 - \\sum_{k=1}^{p}
        a_k e^{-j 2\\pi f k / f_s} \\right|^2}

    Parameters
    ----------
    x : array-like
        1-D input signal.
    order : int
        AR model order (default 16).
    nfft : int
        Number of frequency bins for the PSD (default 512).
    fs : float
        Sampling frequency in Hz (default 1.0).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``frequencies``, ``psd``, ``ar_coeffs``,
        ``noise_variance``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Burg, J.P. (1967). Maximum entropy spectral analysis. *Proc. 37th
    Meeting of the Society of Exploration Geophysicists*.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    ef = x.copy()
    eb = x.copy()
    sigma2 = np.dot(x, x) / n
    ar = np.array([1.0])

    for p in range(1, order + 1):
        efp = ef[p:]
        ebp = eb[p - 1 : n - 1]
        num = -2.0 * np.dot(efp, ebp)
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        if abs(den) < 1e-20:
            break
        k = num / den

        ar_new = np.zeros(p + 1)
        ar_new[0] = 1.0
        ar_new[p] = k
        for i in range(1, p):
            ar_new[i] = ar[i] + k * ar[p - i]
        ar = ar_new

        sigma2 *= 1.0 - k * k

        ef_new = np.zeros(n)
        eb_new = np.zeros(n)
        for i in range(p, n):
            ef_new[i] = ef[i] + k * eb[i - 1]
            eb_new[i] = eb[i - 1] + k * ef[i]
        ef = ef_new
        eb = eb_new

    ar_coeffs = ar

    p_actual = len(ar_coeffs)
    freqs = np.linspace(0, fs / 2, nfft)
    psd = np.zeros(nfft)
    for i, f in enumerate(freqs):
        z = np.exp(-1j * 2 * np.pi * f / fs * np.arange(p_actual))
        denom = np.abs(np.dot(ar_coeffs, z)) ** 2
        psd[i] = sigma2 / max(denom, 1e-20)

    return DescriptiveResult(
        name="burg_psd",
        value=float(order),
        extra={
            "frequencies": freqs,
            "psd": psd,
            "ar_coeffs": ar_coeffs[1:],
            "noise_variance": sigma2,
        },
    )


burgp = burg_psd


# compact alias per ledger/NAMING.md
burgpsd = burg_psd


# -- rgaic: AR model order selection by Akaike's information criterion.
def rangayyan_ar_order_aic(prediction_errors, n_samples, window="hamming"):
    r"""Akaike order selection for an AR model, Rangayyan eq. (7.60).

    .. math:: I(P) = \log \varepsilon_P + \frac{2P}{N_e}

    where :math:`\varepsilon_P` is the total squared prediction error of
    the order-:math:`P` model and :math:`N_e` is the EFFECTIVE number of
    data points after windowing -- the book gives :math:`N_e = 0.4N` for
    a Hamming window.  The chosen order is the one minimising
    :math:`I(P)`.

    This is not the textbook :math:`N\log\sigma^2 + 2p` form: Rangayyan
    normalises by the effective count, which is what windowed data
    actually supplies, and takes the log of the error directly.  The
    placeholder docstring for this module stated the textbook form; the
    book is followed here.

    Parameters
    ----------
    prediction_errors : sequence
        Total squared prediction error for orders 1, 2, ..., P_max.
    n_samples : int
        Number of data samples N.
    window : str or float
        "hamming" (N_e = 0.4 N), "rectangular"/"none" (N_e = N), or a
        float giving the effective-sample fraction directly.

    Returns
    -------
    RichResult
        ``order`` (minimising order), ``criterion`` (I(P) per order),
        ``n_effective``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, eq. (7.60).  Akaike, H. (1974). A new look at the
    statistical model identification. *IEEE Trans. Automatic Control*
    19(6), 716-723.
    """
    eps = [float(v) for v in prediction_errors]
    if not eps:
        raise ValueError("need at least one prediction error")
    if any(v <= 0 for v in eps):
        raise ValueError("prediction errors must be positive")
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    if isinstance(window, str):
        frac = {"hamming": 0.4, "rectangular": 1.0, "none": 1.0}.get(
            window.lower())
        if frac is None:
            raise ValueError("unknown window %r" % window)
    else:
        frac = float(window)
        if not (0.0 < frac <= 1.0):
            raise ValueError("effective-sample fraction must be in (0, 1]")
    n_eff = frac * n
    if n_eff <= 0:
        raise ValueError("effective sample size must be positive")

    crit = [_math.log(e) + 2.0 * (p + 1) / n_eff for p, e in enumerate(eps)]
    best = min(range(len(crit)), key=lambda i: crit[i])
    return RichResult(
        title="Akaike order selection (Rangayyan eq. 7.60)",
        summary_lines=[("order", best + 1), ("min I(P)", crit[best])],
        payload={"order": best + 1, "criterion": crit,
                 "n_effective": n_eff,
                 "method": "Rangayyan (2024) eq. (7.60)"},
    )


aicorder = rangayyan_ar_order_aic


# -- rgarb: AR(p) model via Burg's recursion -- Rangayyan & Krishnan Sec 7.5 / 8.6.2.
def rangayyan_ar_burg(x, order=10):
    """Burg's method for autoregressive (AR) coefficients.

    Estimates AR(p) coefficients ``a_1, …, a_p`` and innovation variance
    ``σ²`` from data ``x`` by minimising the sum of forward+backward
    prediction error energies (always yields a stable model).

    Sign convention: ``x[n] = -Σ a_k x[n-k] + e[n]``.

    Parameters
    ----------
    x : array-like
    order : int
        AR model order.

    Returns
    -------
    RichResult with keys ``ar_coeffs``, ``variance``, ``order``, ``reflection``.

    ``ar_coeffs`` are the coefficients of the PREDICTION-ERROR FILTER,

        A(z) = 1 + a_1 z^-1 + ... + a_p z^-p,

    i.e. the returned array is ``[a_1, ..., a_p]`` with the leading 1
    omitted. The recursion form is therefore

        x[n] = -sum_i a_i x[n-i] + e[n],

    so the coefficients of the usual ``x[n] = sum_i c_i x[n-i]`` form are
    ``c = -ar_coeffs``. Stability is checked on ``[1, *ar_coeffs]``: Burg's
    method guarantees every root lies inside the unit circle (p.456, "The
    magnitudes of the reflection coefficients are less than unity").

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 7.5 "Autoregressive or All-pole
        Modeling", p.369; the Burg-lattice recursion is described in
        Sec 8.6.2, p.456. The previous docstring cited Ch 4.
    Burg, J. P. (1975). *Maximum Entropy Spectral Analysis* (PhD thesis).
        Stanford University.
    Marple, S. L. (1987). *Digital Spectral Analysis*, Ch 8. Prentice-Hall.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    p = int(order)
    if p < 1 or p >= N:
        raise ValueError("order must be 1 <= p < len(x)")

    f = x.copy()
    b = x.copy()
    a = np.zeros(p + 1)
    a[0] = 1.0
    var = float(np.mean(x**2))
    k = np.zeros(p)
    for m in range(p):
        num = -2.0 * np.sum(f[m + 1 : N] * b[m : N - 1])
        den = np.sum(f[m + 1 : N] ** 2) + np.sum(b[m : N - 1] ** 2)
        km = (num / den) if den > 0 else 0.0
        k[m] = km
        new_a = a.copy()
        for i in range(1, m + 2):
            new_a[i] = a[i] + km * a[m + 1 - i]
        a = new_a
        f_new = f[m + 1 : N] + km * b[m : N - 1]
        b_new = b[m : N - 1] + km * f[m + 1 : N]
        f[m + 1 : N] = f_new
        b[m + 1 : N] = b_new
        var = var * (1.0 - km * km)

    res = RichResult(
        title=f"AR({p}) Burg model",
        summary_lines=[
            ("Order", p),
            ("Innovation variance σ²", float(var)),
            ("First reflection k_1", float(k[0])),
        ],
        interpretation=f"Stable AR({p}) fit; residual variance {var:.4g}.",
        payload={"ar_coeffs": a[1:], "variance": float(var), "order": p, "reflection": k},
    )
    return with_describe_pointer(res, "rgarb")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = rangayyan_ar_burg(x, order=4)
# >>> r["ar_coeffs"].shape
# (4,)


# -- rgarsp: AR power spectrum.
def rangayyan_ar_spectrum(x, order=8, fs=1.0, n_freqs=512):
    r"""Autoregressive (parametric) power spectrum (Rangayyan Ch. 3):

    .. math:: S_{AR}(f) = \frac{\sigma^2}
              {\big|1 + \sum_k a_k e^{-j2\pi f k T}\big|^2}.

    Unlike the periodogram this is a smooth, all-pole spectrum with
    resolution not limited by the record length -- which is its appeal
    and its danger: choosing the order too high invents spectral peaks
    that are not in the data. The fitted model's stability is
    reported, since an unstable fit makes the spectrum meaningless.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 8
        AR order.
    fs : float, default 1.0
        Sampling frequency.
    n_freqs : int, default 512
        Frequency grid size.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``a``, ``sigma2``, ``order``,
        ``stable``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (parametric spectral estimation).
    """
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    n_freqs = int(n_freqs)
    if n_freqs < 8:
        raise ValueError(f"n_freqs must be at least 8, got {n_freqs}.")
    yw = rangayyan_yule_walker(x, order=order)
    a = yw["a"]
    freqs = np.linspace(0.0, fs / 2.0, n_freqs)
    k = np.arange(1, a.size + 1)
    expo = np.exp(-2j * np.pi * np.outer(freqs / fs, k))
    denom = np.abs(1.0 + expo @ a) ** 2
    psd = yw["sigma2"] / np.maximum(denom, 1e-300)
    return RichResult(payload={"freqs": freqs, "psd": psd, "a": a,
                               "sigma2": yw["sigma2"], "order": yw["order"],
                               "stable": yw["stable"],
                               "method": "All-pole AR spectrum; high order invents peaks"})


# -- rgburg: Burg AR estimation.
def rangayyan_burg_method(x, order=8, fs=1.0):
    r"""Burg's lattice method for AR estimation (Rangayyan Ch. 3).

    Recursively chooses each reflection coefficient to minimise the
    sum of FORWARD and BACKWARD prediction error powers:

    .. math:: k_m = \frac{-2\sum_n f_{m-1}(n) b_{m-1}(n-1)}
              {\sum_n f_{m-1}^2(n) + \sum_n b_{m-1}^2(n-1)}.

    Unlike Yule-Walker it never forms an autocorrelation estimate, so
    it needs no windowing and gives better resolution on short
    records; and because every :math:`|k_m| \le 1` by construction,
    the resulting model is guaranteed stable.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 8
        AR order.
    fs : float, default 1.0
        Sampling frequency, carried through for spectra.

    Returns
    -------
    RichResult
        keys: ``a``, ``reflection``, ``sigma2``, ``order``,
        ``stable``, ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Burg's method).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = int(order)
    N = x.size
    if p < 1:
        raise ValueError(f"order must be at least 1, got {p}.")
    if N < p + 1:
        raise ValueError(f"need more than order = {p} samples, got {N}.")
    if float(fs) <= 0:
        raise ValueError("fs must be positive.")
    f = x.copy()
    b = x.copy()
    a = np.zeros(0)
    E = float(np.mean(x**2))
    ks = []
    for m in range(1, p + 1):
        fm = f[m:]
        bm = b[m - 1 : -1]
        den = float(np.dot(fm, fm) + np.dot(bm, bm))
        k = -2.0 * float(np.dot(fm, bm)) / den if den > 0 else 0.0
        k = float(np.clip(k, -1.0, 1.0))  # |k| <= 1 keeps the model stable
        ks.append(k)
        a_new = np.r_[a, 0.0] + k * np.r_[a[::-1], 1.0] if a.size else np.array([k])
        a = a_new
        f_new = fm + k * bm
        b_new = bm + k * fm
        f = np.r_[np.zeros(m), f_new]
        b = np.r_[np.zeros(m), b_new]
        E *= 1.0 - k**2
    roots = np.roots(np.r_[1.0, a])
    return RichResult(payload={"a": a, "reflection": np.array(ks), "sigma2": float(E),
                               "order": p, "stable": bool(np.all(np.abs(roots) < 1.0)),
                               "fs": float(fs),
                               "method": "Burg lattice; |k| <= 1 guarantees a stable model"})


# -- rgfpe: Final prediction error (FPE) criterion for AR model order.
def fpeorder(errors, n_samples):
    """Akaike's final prediction error criterion for AR model order.

    Akaike (1970), Annals of the Institute of Statistical Mathematics
    22:203-217:

        FPE(p) = sigma_p^2 (N + p + 1) / (N - p - 1),

    minimized over p.  The factor penalises order: the residual variance
    sigma_p^2 falls monotonically with p (Rangayyan eq. 7.39 guarantees
    it), so without a penalty the criterion would always choose the
    largest order offered.

    Rangayyan (2024) Section 7.5.2 discusses order selection via the
    trend in the total squared error and gives Akaike's INFORMATION
    criterion at eq. (7.60); FPE is not printed in the book, so it is
    cited to Akaike.  The two usually agree; where they differ, FPE is
    the more conservative for small N because its penalty blows up as
    p approaches N.

    Parameters
    ----------
    errors : array-like
        Residual variance for orders p = 1, 2, ..., P (or from p = 0 if
        the first entry is phi(0); see ``start_order``).
    n_samples : int
        Number of samples N used to fit.
    """
    eps = aslist(errors)
    if not eps:
        raise ValueError("need at least one error value")
    if any(v <= 0 for v in eps):
        raise ValueError("residual variances must be positive")
    n = int(n_samples)
    if n <= len(eps) + 1:
        raise ValueError("N must exceed the largest order by more than 1")
    crit = []
    for i, s2 in enumerate(eps, start=1):
        crit.append(s2 * (n + i + 1) / (n - i - 1))
    best = min(range(len(crit)), key=lambda i: crit[i])
    return RichResult(payload={
        "order": best + 1, "criterion": crit, "n": n,
        "start_order": 1,
        "method": "Akaike (1970) FPE; Rangayyan (2024) Section 7.5.2 "
                  "gives AIC at eq. (7.60) instead"})


rangayyan_ar_order_fpe = fpeorder  # pre-policy spelling


# -- rghrvar: HRV AR model LF/HF ratio (sympathovagal balance).
def hrvratio(rr, order=16, fs=4.0):
    """LF/HF ratio of heart-rate variability from the AR model PSD.

    The ratio of low-frequency to high-frequency band power in the AR
    spectrum of :func:`hrvar`, with the band edges of the Task Force of
    the ESC and NASPE (1996).

    The ratio is often called a sympathovagal balance.  That reading is
    contested -- HF is reasonably attributed to vagal activity, but LF
    reflects both branches plus the baroreflex, so the ratio is not a
    clean index of either.  The components are returned individually,
    and in normalized units, so a caller is not forced to rest an
    interpretation on the ratio alone.
    """
    r = hrvar(rr, order=order, fs=fs)
    return RichResult(payload={
        "lf_hf_ratio": r["lf_hf_ratio"], "lf": r["lf"], "hf": r["hf"],
        "vlf": r["vlf"], "total_power": r["total_power"],
        "lf_nu": r["lf_nu"],
        "hf_nu": (100.0 - r["lf_nu"]) if r["lf_nu"] is not None else None,
        "order": r["order"], "bands": r["bands"],
        "interpretation_caveat":
            "LF reflects both autonomic branches and the baroreflex; the "
            "ratio is not a clean index of sympathovagal balance",
        "method": "Rangayyan (2024) Section 7.5 AR model; bands per Task "
                  "Force of the ESC and NASPE (1996)"})


rangayyan_hrv_ar_ratio = hrvratio  # pre-policy spelling


# -- rghrvmod: AR spectral model of HRV for LF/HF decomposition.
def hrvar(rr, order=16, fs=4.0, nfreq=512):
    """AR spectral model of heart-rate variability.

    The RR-interval series is unevenly sampled by construction, so it is
    first resampled onto a uniform grid at ``fs`` (4 Hz is the usual
    choice, comfortably above the 0.4 Hz upper edge of the HF band),
    then modelled all-pole as in Rangayyan (2024) Section 7.5 and the
    model PSD integrated over the standard bands.

    Band edges follow the Task Force of the European Society of
    Cardiology and NASPE (1996), Circulation 93:1043-1065:
        VLF  0.003 - 0.04 Hz
        LF   0.04  - 0.15 Hz
        HF   0.15  - 0.40 Hz.

    The mean is removed before modelling: the DC term of an RR series is
    around 0.8 s, hundreds of times the size of the variability being
    measured, and leaving it in puts a huge pole at zero frequency that
    leaks into the VLF band.
    """
    intervals = aslist(rr)
    if len(intervals) < 8:
        raise ValueError("need at least eight RR intervals")
    if any(v <= 0 for v in intervals):
        raise ValueError("RR intervals must be positive")
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    t = [0.0]
    for v in intervals:
        t.append(t[-1] + v)
    beats = t[1:]
    duration = beats[-1] - beats[0]
    if duration <= 0:
        raise ValueError("the RR series has zero duration")
    n = max(16, int(duration * fsv))
    grid = [beats[0] + i / fsv for i in range(n)]
    series, j = [], 0
    for tv in grid:
        while j < len(beats) - 2 and beats[j + 1] < tv:
            j += 1
        t0, t1 = beats[j], beats[j + 1]
        v0, v1 = intervals[j], intervals[min(j + 1, len(intervals) - 1)]
        w = 0.0 if t1 == t0 else (tv - t0) / (t1 - t0)
        series.append(v0 + w * (v1 - v0))
    mu = fsum(series) / len(series)
    series = [v - mu for v in series]
    p = min(int(order), len(series) - 1)
    fit = arfit(series, p, fs=fsv, nfreq=int(nfreq))
    bands = {"vlf": (0.003, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.40)}
    power = {}
    f, s = fit["freqs"], fit["psd"]
    df = f[1] - f[0] if len(f) > 1 else 0.0
    for name, (lo, hi) in bands.items():
        power[name] = fsum(v * df for u, v in zip(f, s) if lo <= u < hi)
    total = power["vlf"] + power["lf"] + power["hf"]
    out = dict(fit)
    out.update({
        "mean_rr": mu, "resampled": series, "resample_fs": fsv,
        "vlf": power["vlf"], "lf": power["lf"], "hf": power["hf"],
        "total_power": total,
        "lf_hf_ratio": power["lf"] / power["hf"] if power["hf"] > 0 else None,
        "lf_nu": 100.0 * power["lf"] / (power["lf"] + power["hf"])
        if (power["lf"] + power["hf"]) > 0 else None,
        "bands": bands, "order": p,
        "method": "Rangayyan (2024) Section 7.5 AR model; bands per Task "
                  "Force of the ESC and NASPE (1996)"})
    return RichResult(payload=out)


rangayyan_hrv_ar_model = hrvar  # pre-policy spelling


# -- rglevd: Levinson-Durbin recursion for efficient AR model fitting.
def levinson(acf, order=None):
    """Levinson-Durbin recursion for the AR normal equations.

    Rangayyan (2024) eqs. (7.37)-(7.39), Section 7.5.1.  Initialize
    i = 0 and eps_0 = phi_y(0); then for i = 1, 2, ..., P:

        gamma_i = -(1/eps_{i-1}) [ phi_y(i)
                    + sum_{j=1}^{i-1} a_{i-1,j} phi_y(i-j) ]    (7.37)
        a_{i,i} = gamma_i
        a_{i,j} = a_{i-1,j} + gamma_i a_{i-1,i-j},  1 <= j <= i-1  (7.38)
        eps_i   = (1 - gamma_i^2) eps_{i-1}                       (7.39)

    with the final parameters a_k = a_{P,k}.

    The book states two properties that are worth returning rather than
    trusting: the error is monotone, 0 <= eps_i <= eps_{i-1}, and the
    model is stable exactly when |gamma_i| < 1 for every i.  Both are
    checked here, and ``stable`` is False rather than the routine
    silently producing an unstable filter.

    Every lower-order model is computed on the way to order P, so the
    error trajectory needed for order selection (Section 7.5.2) comes
    free; it is returned as ``errors``.
    """
    r = aslist(acf)
    if len(r) < 2:
        raise ValueError("need phi(0) and at least one lag")
    p = len(r) - 1 if order is None else int(order)
    if p < 1:
        raise ValueError("order must be at least 1")
    if p > len(r) - 1:
        raise ValueError("order %d needs %d ACF lags, got %d"
                         % (p, p + 1, len(r) - 1))
    if r[0] <= 0:
        raise ValueError("phi(0) must be positive")
    a = []
    eps = r[0]
    errors = [eps]
    gammas = []
    for i in range(1, p + 1):
        acc = r[i] + fsum(a[j - 1] * r[i - j] for j in range(1, i))
        g = -acc / eps
        gammas.append(g)
        new = [a[j - 1] + g * a[i - j - 1] for j in range(1, i)]
        new.append(g)
        a = new
        eps = (1.0 - g * g) * eps
        errors.append(eps)
    return RichResult(payload={
        "a": a, "reflection": gammas, "error": eps, "errors": errors,
        "gain": sqrt(eps) if eps > 0 else 0.0, "order": p,
        "stable": all(abs(g) < 1.0 for g in gammas),
        "monotone": all(errors[i] <= errors[i - 1] + 1e-12
                        for i in range(1, len(errors))),
        "normalized_error": eps / r[0],
        "sign_convention": "A(z) = 1 + sum a_k z^-k, per eq. (7.18)",
        "method": "Rangayyan (2024) eqs. (7.37)-(7.39)"})


rangayyan_levinson_durbin = levinson  # pre-policy spelling


# -- rglpca: Linear predictive coding (LPC) analysis of speech/biosignals.
def lpc(x, order, method="autocorrelation"):
    """Linear prediction (AR) analysis by the autocorrelation method.

    Rangayyan (2024) Section 7.5, eqs. (7.17)-(7.18), (7.25), (7.35):

        y~(n) = - sum_{k=1}^{P} a_k y(n-k)                       (7.17)
        e(n)  =   y(n) + sum_{k=1}^{P} a_k y(n-k)                (7.18)
        sum_k a_k phi_y(|i-k|) = -phi_y(i),  1 <= i <= P         (7.25)
        G^2 = eps_P = phi_y(0) + sum_k a_k phi_y(k)              (7.35)

    Note the sign: the prediction is MINUS the weighted sum, so A(z) =
    1 + sum a_k z^-k and a stable all-pole model has H(z) = G / A(z).
    Under the other common convention every a_k flips sign; the returned
    ``sign_convention`` says which one these coefficients are in.

    The ACF is the biased estimator, sum over the available overlap
    divided by N.  That is what makes the autocorrelation method's
    Toeplitz system positive-definite and hence the model stable; the
    unbiased 1/(N-m) estimator does not guarantee it.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int
        Model order P.
    method : {"autocorrelation"}
        Only the autocorrelation method is implemented; the covariance
        method of eq. (7.40) uses a finite summation range and does not
        guarantee stability, so it is refused rather than silently
        substituted.
    """
    xs = aslist(x)
    n = len(xs)
    p = int(order)
    if p < 1:
        raise ValueError("order must be at least 1")
    if n <= p:
        raise ValueError("need more samples (%d) than the order (%d)"
                         % (n, p))
    if method != "autocorrelation":
        raise ValueError("only the autocorrelation method is implemented; "
                         "eq. (7.40)'s covariance method is not")
    acf = [fsum(xs[i] * xs[i + m] for i in range(n - m)) / n
           for m in range(p + 1)]
    if acf[0] <= 0:
        raise ValueError("the signal has zero energy")
    lev = levinson(acf, order=p)
    a = lev["a"]
    resid = []
    for i in range(n):
        acc = xs[i]
        for k in range(1, p + 1):
            if i - k >= 0:
                acc += a[k - 1] * xs[i - k]
        resid.append(acc)
    tse = fsum(v * v for v in resid[p:])
    return RichResult(payload={
        "a": a, "gain": lev["gain"], "error": lev["error"],
        "reflection": lev["reflection"], "acf": acf, "order": p,
        "residual": resid, "residual_energy": tse,
        "stable": lev["stable"],
        "normalized_error": lev["normalized_error"],
        "sign_convention": "A(z) = 1 + sum a_k z^-k, per eq. (7.18)",
        "method": "Rangayyan (2024) eqs. (7.17)-(7.18), (7.25), (7.35)"})


rangayyan_lpc_analysis = lpc  # pre-policy spelling


# -- rglpcs: LPC synthesis filter for signal reconstruction.
def lpcsynth(a, excitation, gain=1.0, initial=None):
    """All-pole synthesis filter: run H(z) = G / A(z).

    Rangayyan (2024) Section 7.5.  Inverting eq. (7.18) gives the
    recursion

        y(n) = G e(n) - sum_{k=1}^{P} a_k y(n-k),

    the minus following from the book's sign convention, A(z) =
    1 + sum a_k z^-k.  Driving this with an impulse train models voiced
    speech and with white noise models unvoiced speech (Figure 7.2).

    Feeding coefficients from the opposite convention produces a filter
    with every pole reflected through the origin -- usually unstable and
    always wrong -- so ``diverged`` is reported when the output grows
    past a bound, rather than returning a wall of infinities.
    """
    ak = aslist(a)
    e = aslist(excitation)
    if not ak:
        raise ValueError("need at least one AR coefficient")
    if not e:
        raise ValueError("need an excitation sequence")
    p = len(ak)
    hist = [0.0] * p if initial is None else aslist(initial)
    if len(hist) != p:
        raise ValueError("initial state must hold %d samples" % p)
    y = []
    limit = 1e12 * (1.0 + max(abs(v) for v in e))
    diverged = False
    for v in e:
        acc = gain * v - fsum(ak[k] * hist[k] for k in range(p))
        if not (abs(acc) < limit):
            diverged = True
            acc = float("inf") if acc > 0 else float("-inf")
            y.append(acc)
            break
        y.append(acc)
        hist = [acc] + hist[:-1]
    return RichResult(payload={
        "y": y, "n": len(y), "order": p, "gain": float(gain),
        "diverged": diverged,
        "sign_convention": "y(n) = G e(n) - sum a_k y(n-k)",
        "method": "Rangayyan (2024) Section 7.5 (all-pole synthesis)"})


rangayyan_lpc_synthesis = lpcsynth  # pre-policy spelling


# -- rgmdl: Minimum description length (MDL) criterion for AR model order.
def mdlorder(errors, n_samples):
    """Minimum description length criterion for AR model order.

    Rissanen (1978), Automatica 14:465-471:

        MDL(p) = N log(sigma_p^2) + p log(N),

    minimized over p.  The penalty grows as log(N) per parameter, faster
    than AIC's constant 2 per parameter for any N > 7, so MDL selects
    the same order or a lower one -- and unlike AIC it is consistent:
    as N grows it converges on the true order rather than overfitting.

    Rangayyan (2024) Section 7.5.2 covers order selection by the error
    trend and gives AIC at eq. (7.60); MDL is not printed in the book,
    so it is cited to Rissanen.  ``aic`` is returned alongside so the
    two penalties can be compared on the same errors.
    """
    eps = aslist(errors)
    if not eps:
        raise ValueError("need at least one error value")
    if any(v <= 0 for v in eps):
        raise ValueError("residual variances must be positive")
    n = int(n_samples)
    if n < 2:
        raise ValueError("N must be at least 2")
    mdl = [n * log(s2) + i * log(n) for i, s2 in enumerate(eps, start=1)]
    aic = [n * log(s2) + 2.0 * i for i, s2 in enumerate(eps, start=1)]
    bm = min(range(len(mdl)), key=lambda i: mdl[i])
    ba = min(range(len(aic)), key=lambda i: aic[i])
    return RichResult(payload={
        "order": bm + 1, "criterion": mdl, "aic": aic,
        "aic_order": ba + 1, "n": n, "start_order": 1,
        "penalty_per_parameter": log(n),
        "stricter_than_aic": log(n) > 2.0,
        "method": "Rissanen (1978) MDL; Rangayyan (2024) Section 7.5.2 "
                  "gives AIC at eq. (7.60) instead"})


rangayyan_ar_order_mdl = mdlorder  # pre-policy spelling


# -- rgparmod: Parametric system identification: AR all-pole model fitting.
def arfit(x, order, fs=1.0, nfreq=256):
    """Fit an all-pole model and evaluate its PSD.

    Rangayyan (2024) Section 7.5: the parametric route to a spectrum.
    The model is H(z) = G / A(z) with A(z) = 1 + sum a_k z^-k, so the
    model PSD is

        S(f) = G^2 / |A(exp(-j 2 pi f / fs))|^2,

    a smooth function of P + 1 parameters rather than a periodogram of
    N/2 + 1 noisy bins.  That smoothness is the point and also the trap:
    the model can only produce P/2 spectral peaks, so a spectrum with
    more resonances than the order allows is silently merged.
    ``max_peaks`` records that ceiling.
    """
    fit = lpc(x, order)
    a = fit["a"]
    g2 = fit["error"]
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    k = int(nfreq)
    if k < 2:
        raise ValueError("nfreq must be at least 2")
    freqs, psd = [], []
    for i in range(k):
        f = 0.5 * fsv * i / (k - 1)
        w = 2.0 * pi * f / fsv
        re, im = 1.0, 0.0
        for j, av in enumerate(a, start=1):
            re += av * cos(-w * j)
            im += av * sin(-w * j)
        denom = re * re + im * im
        freqs.append(f)
        psd.append(g2 / denom if denom > 0 else float("inf"))
    out = dict(fit)
    out.update({"freqs": freqs, "psd": psd, "fs": fsv,
                "max_peaks": len(a) // 2,
                "method": "Rangayyan (2024) Section 7.5 (all-pole PSD)"})
    return RichResult(payload=out)


rangayyan_parametric_sysid = arfit  # pre-policy spelling


# -- rgpcgar: AR/ARMA model of PCG for heart sound characterization.
def pcgar(x, fs, order=None, segment=None):
    """AR model of a PCG segment, with the resonances its poles imply.

    Rangayyan (2024) Chapter 7 applies AR modelling to the PCG: the
    poles of the all-pole model track the resonances of S1 and S2, and
    a murmur adds high-frequency poles.  Each pole p_k on the z-plane
    corresponds to a resonance at

        f_k = (fs / 2 pi) angle(p_k),   bandwidth = -(fs / pi) ln|p_k|,

    so a pole close to the unit circle is a narrow, strong resonance.
    Only poles in the upper half plane are reported: for a real signal
    the poles come in conjugate pairs and listing both halves would
    double-count every resonance.

    ``order`` defaults to 2 + fs/1000 rounded, the usual rule of thumb
    of two poles per kHz plus a couple for the spectral tilt; it is a
    starting point, not a fit -- use :func:`mdlorder` on the error
    trajectory to choose it properly.
    """
    xs = aslist(x)
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if segment is not None:
        lo, hi = segment
        xs = xs[int(lo):int(hi)]
    if len(xs) < 16:
        raise ValueError("need at least sixteen samples in the segment")
    p = int(order) if order is not None else max(4, int(round(2 + fsv / 1000.0)))
    fit = arfit(xs, p, fs=fsv)
    pz = polezero([1.0], fit["a"])
    res = []
    for pole in pz["poles"]:
        ang = _angle(pole)
        if ang <= 0:
            continue                       # keep one of each conjugate pair
        mag = abs(pole)
        if mag <= 0:
            continue
        res.append({"frequency": fsv * ang / (2.0 * pi),
                    "bandwidth": -fsv * log(mag) / pi,
                    "radius": mag, "pole": pole})
    res.sort(key=lambda d: d["frequency"])
    out = dict(fit)
    out.update({"poles": pz["poles"], "resonances": res, "order": p,
                "stable": pz["stable"], "fs": fsv,
                "method": "Rangayyan (2024) Chapter 7 (AR modelling of "
                          "the PCG)"})
    return RichResult(payload=out)


rangayyan_pcg_ar_model = pcgar  # pre-policy spelling


# -- rgpzmod: ARMA pole-zero model identification.
def armafit(x, p, q, fs=1.0):
    """ARMA (pole-zero) model: all-pole part, then the MA part.

    Rangayyan (2024) Section 7.7 contrasts the all-pole model of Section
    7.5, H(z) = G / A(z), with the pole-zero model H(z) = B(z) / A(z),
    which is needed when the signal has spectral nulls as well as
    resonances -- an all-pole model can only produce peaks.

    Fitted in two stages, which is the standard practical route: the AR
    part from the ACF via Levinson-Durbin, then the MA part from the ACF
    of the AR residual.  This is NOT joint maximum likelihood, and it is
    biased when the true zeros are close to the poles; the stage
    structure is stated here rather than the result being presented as
    an optimal fit.
    """
    xs = aslist(x)
    pi_, qi = int(p), int(q)
    if pi_ < 1:
        raise ValueError("the AR order p must be at least 1")
    if qi < 0:
        raise ValueError("the MA order q cannot be negative")
    ar = lpc(xs, pi_)
    resid = ar["residual"][pi_:]
    n = len(resid)
    if n <= qi:
        raise ValueError("too few residual samples (%d) for MA order %d"
                         % (n, qi))
    if qi == 0:
        b = [ar["gain"]]
    else:
        rr = [fsum(resid[i] * resid[i + m] for i in range(n - m)) / n
              for m in range(qi + 1)]
        if rr[0] <= 0:
            raise ValueError("the residual has zero energy")
        # Durbin's two-stage MA estimate: fit a long AR to the residual
        # and invert it to get the MA coefficients
        long_order = min(4 * qi, n - 1)
        acf_long = [fsum(resid[i] * resid[i + m] for i in range(n - m)) / n
                    for m in range(long_order + 1)]
        inner = levinson(acf_long, order=long_order)
        b = levinson([1.0] + inner["a"][:qi], order=qi)["a"]
        b = [ar["gain"]] + b
    pz = polezero(b, ar["a"])
    return RichResult(payload={
        "a": ar["a"], "b": b, "p": pi_, "q": qi,
        "gain": ar["gain"], "poles": pz["poles"], "zeros": pz["zeros"],
        "stable": pz["stable"], "ar_error": ar["error"],
        "two_stage": True,
        "method": "Rangayyan (2024) Section 7.7 (pole-zero model), "
                  "fitted AR-then-MA rather than jointly"})


rangayyan_pole_zero_model = armafit  # pre-policy spelling


# -- rgpzp: Pole-zero plot from transfer function coefficients.
def polezero(b, a=None):
    """Poles and zeros of a rational transfer function.

    Rangayyan (2024) eqs. (3.67), (3.69): with

        H(z) = ( sum_{k=0}^{N} b_k z^-k ) / ( 1 + sum_{k=1}^{M} a_k z^-k ),

    the zeros are the roots of the numerator polynomial and the poles the
    roots of the denominator.  Section 3.4.3 reads the plot: a zero on
    the unit circle is a spectral null, a pole near it a resonance.

    The denominator is taken in the book's normalized form, leading
    coefficient 1 -- so pass ``a`` WITHOUT that leading 1, matching the
    a_1..a_M of eq. (3.67).  Roots are found by Durand-Kerner iteration,
    which needs no companion-matrix eigensolver and converges for simple
    roots from any generic start.
    """
    bs = aslist(b)
    if not bs:
        raise ValueError("need at least one numerator coefficient")
    as_ = [] if a is None else aslist(a)

    def roots(coeffs):
        # coeffs are in ascending powers of z^-1; the roots in z of
        # sum c_k z^-k are the roots of the reversed polynomial in z
        c = list(coeffs)
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        if len(c) < 2:
            return []
        deg = len(c) - 1
        lead = c[0]
        if lead == 0:
            raise ValueError("leading coefficient must be nonzero")
        mono = [v / lead for v in c]           # z^deg + m1 z^(deg-1) + ...
        est = [complex(0.4, 0.9) ** k for k in range(1, deg + 1)]
        for _ in range(500):
            shift = 0.0
            for i in range(deg):
                num = est[i] ** deg
                for k in range(1, deg + 1):
                    num += mono[k] * est[i] ** (deg - k)
                den = 1.0 + 0j
                for j in range(deg):
                    if j != i:
                        den *= (est[i] - est[j])
                if den == 0:
                    continue
                step = num / den
                est[i] -= step
                shift = max(shift, abs(step))
            if shift < 1e-14:
                break
        return est

    zeros = roots(bs)
    poles = roots([1.0] + as_)
    return RichResult(payload={
        "zeros": zeros, "poles": poles,
        "n_zeros": len(zeros), "n_poles": len(poles),
        "stable": all(abs(p) < 1.0 for p in poles),
        "minimum_phase": all(abs(z) < 1.0 for z in zeros),
        "zeros_on_unit_circle": [z for z in zeros
                                 if abs(abs(z) - 1.0) < 1e-9],
        "method": "Rangayyan (2024) eqs. (3.67), (3.69)"})


rangayyan_pole_zero_plot = polezero  # pre-policy spelling


# -- rgyw: Yule-Walker AR estimation.
def rangayyan_yule_walker(x, order=4):
    r"""Yule-Walker autoregressive parameter estimation (Rangayyan
    Ch. 3):

    .. math:: \mathbf{R}_{xx}\,\mathbf{a} = -\mathbf{r},
              \qquad R_{xx}(i,j) = R_{xx}(|i-j|),

    a Toeplitz system solved for the AR coefficients. The BIASED
    autocorrelation is used deliberately: it is positive
    semi-definite, which guarantees the Toeplitz matrix is invertible
    and the fitted AR model is stable. The unbiased estimate can yield
    an unstable model, so it is the wrong input here even though it is
    the better estimate of the ACF itself.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 4
        AR order.

    Returns
    -------
    RichResult
        keys: ``a`` (AR coefficients), ``sigma2`` (innovation
        variance), ``order``, ``stable`` (all roots inside the unit
        circle), ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Yule-Walker / AR modelling).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = int(order)
    if p < 1:
        raise ValueError(f"order must be at least 1, got {p}.")
    if x.size < p + 1:
        raise ValueError(f"need more than order = {p} samples, got {x.size}.")
    R = rangayyan_acf_estimate(x, max_lag=p, biased=True)["acf_biased"]
    Rm = np.array([[R[abs(i - j)] for j in range(p)] for i in range(p)])
    r = R[1 : p + 1]
    try:
        a = np.linalg.solve(Rm, -r)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(Rm, -r, rcond=None)[0]
    sigma2 = float(R[0] + a @ r)
    roots = np.roots(np.r_[1.0, a])
    return RichResult(payload={"a": a, "sigma2": sigma2, "order": p,
                               "stable": bool(np.all(np.abs(roots) < 1.0)),
                               "reflection_roots": roots,
                               "method": "Toeplitz Yule-Walker on the BIASED ACF (guarantees stability)"})


# -- rng058: Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors..
def pzform(zeros, poles, z=None, gain=1.0):
    """Transfer function in pole-zero factored form.

    Rangayyan (2024) eq. (3.69):
        H(z) = prod_{k=1}^{N} (1 - z_k z^-1) / prod_{k=1}^{M} (1 - p_k z^-1),

    the factorization of the rational H(z) of eq. (3.67), with z_k the
    zeros (roots of the numerator) and p_k the poles (roots of the
    denominator).  This is the z^-1 form; eq. (3.70) is the equivalent
    z form and differs by a factor z^(M-N), which matters whenever
    N != M.

    A pole ON the unit circle makes H undefined there and a pole outside
    it makes a causal system unstable, so both are reported rather than
    left for the caller to notice in the output.
    """
    zs = [complex(v) for v in zeros]
    ps = [complex(v) for v in poles]
    out = {"zeros": zs, "poles": ps, "n_zeros": len(zs),
           "n_poles": len(ps), "gain": complex(gain),
           "stable": all(abs(p) < 1.0 for p in ps),
           "poles_on_unit_circle": [p for p in ps if abs(abs(p) - 1.0) < 1e-12],
           "method": "Rangayyan (2024) eq. (3.69)"}
    if z is None:
        out["H"] = None
        return RichResult(payload=out)
    scalar = isinstance(z, (int, float, complex))
    pts = [complex(z)] if scalar else [complex(v) for v in z]
    vals = []
    for zv in pts:
        if zv == 0:
            raise ValueError("the z^-1 form of eq. (3.69) is undefined at "
                             "z = 0; use pzformz for eq. (3.70)")
        num = complex(gain)
        for zk in zs:
            num *= (1.0 - zk / zv)
        den = 1.0 + 0j
        for pk in ps:
            den *= (1.0 - pk / zv)
        if den == 0:
            raise ValueError("z coincides with a pole of H")
        vals.append(num / den)
    out["H"] = vals[0] if scalar else vals
    out["z"] = pts[0] if scalar else pts
    return RichResult(payload=out)


rangayyan_ch3_pole_zero_factored_form = pzform  # pre-policy spelling


# -- rng059: Alternative pole-zero factored transfer function with z^(M-N) gain factor..
def pzformz(zeros, poles, z=None, gain=1.0):
    """Pole-zero form in z, with the explicit z^(M-N) factor.

    Rangayyan (2024) eq. (3.70):
        H(z) = z^(M-N) prod_{k=1}^{N} (z - z_k)
                      / prod_{k=1}^{M} (z - p_k),

    the same transfer function as eq. (3.69) rewritten in z rather than
    z^-1.  The z^(M-N) factor is exactly what the change of variable
    produces, and dropping it -- the usual slip -- multiplies H by
    z^(N-M), a pure delay or advance that is invisible in the magnitude
    response and wrecks the phase.

    Both forms are evaluated here and compared, so the equality is
    demonstrated on the caller's own poles and zeros.
    """
    zs = [complex(v) for v in zeros]
    ps = [complex(v) for v in poles]
    n, m = len(zs), len(ps)
    out = {"zeros": zs, "poles": ps, "exponent": m - n,
           "gain": complex(gain), "stable": all(abs(p) < 1.0 for p in ps),
           "method": "Rangayyan (2024) eq. (3.70)"}
    if z is None:
        out["H"] = None
        return RichResult(payload=out)
    scalar = isinstance(z, (int, float, complex))
    pts = [complex(v) for v in ([z] if scalar else z)]
    vals, other = [], []
    for zv in pts:
        num = complex(gain) * (zv ** (m - n))
        for zk in zs:
            num *= (zv - zk)
        den = 1.0 + 0j
        for pk in ps:
            den *= (zv - pk)
        if den == 0:
            raise ValueError("z coincides with a pole of H")
        vals.append(num / den)
        other.append(pzform(zs, ps, z=zv, gain=gain)["H"])
    gap = max(abs(a - b) for a, b in zip(vals, other))
    scale = max(abs(v) for v in vals) or 1.0
    out["H"] = vals[0] if scalar else vals
    out["H_from_eq369"] = other[0] if scalar else other
    out["max_difference"] = gap
    out["agrees_with_eq369"] = gap <= 1e-9 * scale
    out["z"] = pts[0] if scalar else pts
    return RichResult(payload=out)


rangayyan_ch3_pole_zero_factored_form_alt = pzformz  # pre-policy spelling


# -- rng060: Frequency response evaluated at z_0 on the unit circle from pole-zero form..
def pzresp(zeros, poles, omega, gain=1.0):
    """Frequency response read off the pole-zero plot.

    Rangayyan (2024) eqs. (3.71)-(3.73).  Evaluating eq. (3.70) at a
    point z0 = exp(j omega) on the unit circle,

        H(omega_0) = z0^(M-N) prod (z0 - z_k) / prod (z0 - p_k)   (3.71)
        |H(omega_0)| = prod l_k / prod r_k                        (3.72)
        angle H(omega_0) = (M-N) angle(z0)
                           + sum alpha_k - sum beta_k             (3.73)

    where l_k is the distance from z0 to the k-th zero, r_k the distance
    to the k-th pole, and alpha_k, beta_k the angles of those vectors.

    The book's reading follows directly: a zero ON the unit circle sends
    some l_k to zero, so the magnitude has a spectral null there and the
    phase jumps by 180 degrees as omega crosses it; a pole close to the
    circle makes some r_k small, so the magnitude has a resonance.  The
    per-pole and per-zero distances are returned so that reading can be
    made rather than taken on trust.
    """
    zs = [complex(v) for v in zeros]
    ps = [complex(v) for v in poles]
    n, m = len(zs), len(ps)
    scalar = isinstance(omega, (int, float))
    ws = [float(omega)] if scalar else [float(v) for v in omega]
    H, mags, phases, dist_z, dist_p = [], [], [], [], []
    for w in ws:
        z0 = complex(cos(w), sin(w))
        lk = [abs(z0 - zk) for zk in zs]
        rk = [abs(z0 - pk) for pk in ps]
        if any(v == 0 for v in rk):
            raise ValueError("a pole lies exactly on the evaluation point")
        num = complex(gain) * (z0 ** (m - n))
        for zk in zs:
            num *= (z0 - zk)
        den = 1.0 + 0j
        for pk in ps:
            den *= (z0 - pk)
        val = num / den
        H.append(val)
        prod_l = 1.0
        for v in lk:
            prod_l *= v
        prod_r = 1.0
        for v in rk:
            prod_r *= v
        mags.append(abs(gain) * prod_l / prod_r)
        alpha = fsum(_angle(z0 - zk) for zk in zs)
        beta = fsum(_angle(z0 - pk) for pk in ps)
        phases.append((m - n) * w + alpha - beta)
        dist_z.append(lk)
        dist_p.append(rk)
    gap = max(abs(abs(a) - b) for a, b in zip(H, mags))
    return RichResult(payload={
        "H": H[0] if scalar else H,
        "magnitude": mags[0] if scalar else mags,
        "phase": phases[0] if scalar else phases,
        "zero_distances": dist_z[0] if scalar else dist_z,
        "pole_distances": dist_p[0] if scalar else dist_p,
        "omega": ws[0] if scalar else ws,
        "magnitude_matches_product": gap <= 1e-9 * (1 + max(mags)),
        "method": "Rangayyan (2024) eqs. (3.71)-(3.73)"})


rangayyan_ch3_frequency_response_from_pole_zero = pzresp  # pre-policy spelling


_CHEATSHEET = [
    'burgp: Burg AR spectral estimation.',
    "rgaic: AR model order selection by Akaike's information criterion.",
    "rgarb: AR(p) model via Burg's recursion -- Rangayyan & Krishnan Sec 7.5 / 8.6.2.",
    'rgarsp: AR power spectrum.',
    'rgburg: Burg AR estimation.',
    'rgfpe: Akaike final prediction error criterion (Akaike 1970)',
    'rghrvar: HRV LF/HF ratio from the AR model PSD',
    'rghrvmod: AR spectral model of HRV, Section 7.5 + Task Force bands',
    'rglevd: Levinson-Durbin recursion, Rangayyan eqs. (7.37)-(7.39)',
    'rglpca: LPC/AR analysis, Rangayyan eqs. (7.17)-(7.35)',
    'rglpcs: all-pole synthesis filter, Rangayyan Section 7.5',
    'rgmdl: minimum description length order criterion (Rissanen 1978)',
    'rgparmod: all-pole model and its PSD, Rangayyan Section 7.5',
    'rgpcgar: AR model of the PCG and its resonances, Chapter 7',
    'rgpzmod: ARMA pole-zero model, Rangayyan Section 7.7',
    'rgpzp: poles and zeros of H(z), Rangayyan eqs. (3.67), (3.69)',
    'rgyw: Yule-Walker AR estimation.',
    'rng058: pole-zero factored form, Rangayyan eq. (3.69)',
    'rng059: pole-zero form in z, Rangayyan eq. (3.70)',
    'rng060: frequency response from the pole-zero plot, eqs. (3.71)-(3.73)',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
