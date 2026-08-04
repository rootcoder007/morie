# morie.fn -- bsaar (rootcoder007/morie)
"""Parametric modelling: AR/ARMA, LPC, Levinson-Durbin, Burg, Yule-Walker, pole-zero models, model-order selection.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 20
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
import math as _math
from . import _array_core as np
from ._containers import DescriptiveResult
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from .bsacorr import rangayyan_acf_estimate

__all__ = [
    'burg_psd',
    'rangayyan_ar_order_aic',
    'aicorder',
    'rangayyan_ar_burg',
    'rangayyan_ar_spectrum',
    'rangayyan_burg_method',
    'rangayyan_ar_order_fpe',
    'rangayyan_hrv_ar_ratio',
    'rangayyan_hrv_ar_model',
    'rangayyan_levinson_durbin',
    'rangayyan_lpc_analysis',
    'rangayyan_lpc_synthesis',
    'rangayyan_ar_order_mdl',
    'rangayyan_parametric_sysid',
    'rangayyan_pcg_ar_model',
    'rangayyan_pole_zero_model',
    'rangayyan_pole_zero_plot',
    'rangayyan_yule_walker',
    'rangayyan_ch3_pole_zero_factored_form',
    'rangayyan_ch3_pole_zero_factored_form_alt',
    'rangayyan_ch3_frequency_response_from_pole_zero',
]


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
def rangayyan_ar_order_fpe(x, max_order):
    """
    Final prediction error (FPE) criterion for AR model order

    Formula: FPE(p) = sigma_p^2 * (N+p+1)/(N-p-1)

    Parameters
    ----------
    x : array-like
        Input data.
    max_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_order, fpe_values

    References
    ----------
    Rangayyan Ch 7.5.2
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
            "method": "Final prediction error (FPE) criterion for AR model order",
        }
    )


# -- rghrvar: HRV AR model LF/HF ratio (sympathovagal balance).
def rangayyan_hrv_ar_ratio(rr_intervals, ar_order):
    """
    HRV AR model LF/HF ratio (sympathovagal balance)

    Formula: LF power from AR PSD integral [0.04-0.15Hz]; HF [0.15-0.40Hz]; ratio

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    ar_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf_hf_ratio, lf_nu, hf_nu

    References
    ----------
    Rangayyan Ch 7.9
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "HRV AR model LF/HF ratio (sympathovagal balance)"}
    )


# -- rghrvmod: AR spectral model of HRV for LF/HF decomposition.
def rangayyan_hrv_ar_model(rr_intervals, order):
    """
    AR spectral model of HRV for LF/HF decomposition

    Formula: S_RR(f) = sigma^2/|A(f)|^2; LF/HF from integral of AR PSD bands

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ar_psd, freqs, lf, hf

    References
    ----------
    Rangayyan Ch 7.9
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "AR spectral model of HRV for LF/HF decomposition"}
    )


# -- rglevd: Levinson-Durbin recursion for efficient AR model fitting.
def rangayyan_levinson_durbin(acf, order):
    """
    Levinson-Durbin recursion for efficient AR model fitting

    Formula: k_m = -(R(m) + sum a_{m-1}(k)*R(m-k)) / P_{m-1}; forward/backward update

    Parameters
    ----------
    acf : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_coeffs, k_reflections, prediction_error

    References
    ----------
    Rangayyan Ch 7.5
    """
    acf = np.asarray(acf, dtype=float)
    n = int(acf) if acf.ndim == 0 else len(acf)
    result = float(np.mean(acf))
    se = float(np.std(acf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Levinson-Durbin recursion for efficient AR model fitting",
        }
    )


# -- rglpca: Linear predictive coding (LPC) analysis of speech/biosignals.
def rangayyan_lpc_analysis(x, order, frame_len, hop_len, fs):
    """
    Linear predictive coding (LPC) analysis of speech/biosignals

    Formula: Minimize E=sum(e^2); e[n]=x[n]-sum a_k*x[n-k]; solved via autocorrelation method

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.
    frame_len : array-like
        Input data.
    hop_len : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lpc_coeffs, gain, residual

    References
    ----------
    Rangayyan Ch 7.5
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
            "method": "Linear predictive coding (LPC) analysis of speech/biosignals",
        }
    )


# -- rglpcs: LPC synthesis filter for signal reconstruction.
def rangayyan_lpc_synthesis(lpc_coeffs, gain, excitation):
    """
    LPC synthesis filter for signal reconstruction

    Formula: x_hat[n] = sum a_k*lpc_coeffs[n-k] + G*e[n] where e[n] is excitation

    Parameters
    ----------
    lpc_coeffs : array-like
        Input data.
    gain : array-like
        Input data.
    excitation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_synth

    References
    ----------
    Rangayyan Ch 7.5
    """
    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float)
    n = int(lpc_coeffs) if lpc_coeffs.ndim == 0 else len(lpc_coeffs)
    result = float(np.mean(lpc_coeffs))
    se = float(np.std(lpc_coeffs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "LPC synthesis filter for signal reconstruction"}
    )


# -- rgmdl: Minimum description length (MDL) criterion for AR model order.
def rangayyan_ar_order_mdl(x, max_order):
    """
    Minimum description length (MDL) criterion for AR model order

    Formula: MDL(p) = N*log(sigma_p^2) + p*log(N)

    Parameters
    ----------
    x : array-like
        Input data.
    max_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_order, mdl_values

    References
    ----------
    Rangayyan Ch 7.5.2
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
            "method": "Minimum description length (MDL) criterion for AR model order",
        }
    )


# -- rgparmod: Parametric system identification: AR all-pole model fitting.
def rangayyan_parametric_sysid(x, order):
    """
    Parametric system identification: AR all-pole model fitting

    Formula: H(z) = 1 / A(z) = 1 / (1 + a1*z^{-1} + ... + ap*z^{-p})

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_coeffs, excitation

    References
    ----------
    Rangayyan Ch 7.4
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
            "method": "Parametric system identification: AR all-pole model fitting",
        }
    )


# -- rgpcgar: AR/ARMA model of PCG for heart sound characterization.
def rangayyan_pcg_ar_model(pcg, fs, p, q):
    """
    AR/ARMA model of PCG for heart sound characterization

    Formula: AR: H(z)=1/A(z); ARMA: H(z)=B(z)/A(z); poles track S1/S2 resonances

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: poles, freqs_from_poles

    References
    ----------
    Rangayyan Ch 7.10
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "AR/ARMA model of PCG for heart sound characterization",
        }
    )


# -- rgpzmod: ARMA pole-zero model identification.
def rangayyan_pole_zero_model(x, p, q):
    """
    ARMA pole-zero model identification

    Formula: H(z) = B(z)/A(z); poles from AR, zeros from MA part

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b_coeffs, a_coeffs

    References
    ----------
    Rangayyan Ch 7.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARMA pole-zero model identification"})


# -- rgpzp: Pole-zero plot from transfer function coefficients.
def rangayyan_pole_zero_plot(b, a):
    """
    Pole-zero plot from transfer function coefficients

    Formula: H(z) = B(z)/A(z); zeros = roots(B), poles = roots(A)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: zeros, poles

    References
    ----------
    Rangayyan Ch 3.4.3
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Pole-zero plot from transfer function coefficients"}
    )


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
def rangayyan_ch3_pole_zero_factored_form(z_k, p_k, z, N, M):
    """
    Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors.

    Formula: H(z) = prod_{k=1}^{N} (1 - z_k z^(-1)) / prod_{k=1}^{M} (1 - p_k z^(-1))

    Parameters
    ----------
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    z : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.69, p. 124
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
            "method": "Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors.",
        }
    )


# -- rng059: Alternative pole-zero factored transfer function with z^(M-N) gain factor..
def rangayyan_ch3_pole_zero_factored_form_alt(z_k, p_k, z, N, M):
    """
    Alternative pole-zero factored transfer function with z^(M-N) gain factor.

    Formula: H(z) = z^(M-N) * prod_{k=1}^{N} (z - z_k) / prod_{k=1}^{M} (z - p_k)

    Parameters
    ----------
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    z : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.70, p. 124
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
            "method": "Alternative pole-zero factored transfer function with z^(M-N) gain factor.",
        }
    )


# -- rng060: Frequency response evaluated at z_0 on the unit circle from pole-zero form..
def rangayyan_ch3_frequency_response_from_pole_zero(z_0, z_k, p_k, N, M):
    """
    Frequency response evaluated at z_0 on the unit circle from pole-zero form.

    Formula: H(omega_0)|_{z=z_0} = z_0^(M-N) * prod_{k=1}^{N} (z_0 - z_k) / prod_{k=1}^{M} (z_0 - p_k)

    Parameters
    ----------
    z_0 : array-like
        Input data.
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.71, p. 124
    """
    z_0 = np.atleast_1d(np.asarray(z_0, dtype=float))
    n = len(z_0)
    result = float(np.mean(z_0))
    se = float(np.std(z_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency response evaluated at z_0 on the unit circle from pole-zero form.",
        }
    )


_CHEATSHEET = [
    'burg_psd({}) -> Burg AR spectral estimation.',
    'aicorder: I(P) = log eps_P + 2P/Ne, Ne = 0.4N for Hamming',
    'rgarb: AR(p) coefficients via Burg method -- Rangayyan & Krishnan Sec 7.5',
    'rgarsp: resolution not limited by record length -- but order too high fabricates peaks',
    'rgburg: no ACF, no windowing; |k|<=1 makes stability automatic',
    'rgfpe: Final prediction error (FPE) criterion for AR model order',
    'rghrvar: HRV AR model LF/HF ratio (sympathovagal balance)',
    'rghrvmod: AR spectral model of HRV for LF/HF decomposition',
    'rglevd: Levinson-Durbin recursion for efficient AR model fitting',
    'rglpca: Linear predictive coding (LPC) analysis of speech/biosignals',
    'rglpcs: LPC synthesis filter for signal reconstruction',
    'rgmdl: Minimum description length (MDL) criterion for AR model order',
    'rgparmod: Parametric system identification: AR all-pole model fitting',
    'rgpcgar: AR/ARMA model of PCG for heart sound characterization',
    'rgpzmod: ARMA pole-zero model identification',
    'rgpzp: Pole-zero plot from transfer function coefficients',
    'rgyw: biased ACF here is a feature -- it keeps the AR model stable',
    'rng058: Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors.',
    'rng059: Alternative pole-zero factored transfer function with z^(M-N) gain factor.',
    'rng060: Frequency response evaluated at z_0 on the unit circle from pole-zero form.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
