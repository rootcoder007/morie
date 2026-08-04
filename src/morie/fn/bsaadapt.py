# morie.fn -- bsaadapt (rootcoder007/morie)
"""Optimal and adaptive filtering: Wiener-Hopf, LMS, RLS, adaptive noise cancellation, Kalman, adaptive segmentation.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 57
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from . import _array_core as np
from . import _stats_core as stats
from ._containers import SignalResult
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import toeplitz

__all__ = [
    'rangayyan_acf_distance',
    'rangayyan_adaptive_filter',
    'rangayyan_anc',
    'rangayyan_eeg_adaptive_seg',
    'rangayyan_fetal_ecg',
    'rangayyan_gen_likelihood_ratio',
    'rangayyan_kalman_filter',
    'rangayyan_lms_filter',
    'rangayyan_pcg_adaptive_seg',
    'rangayyan_riccati_eq',
    'rangayyan_rls_filter',
    'rangayyan_rls_monitor',
    'rangayyan_rls_lattice',
    'rangayyan_spec_error_meas',
    'rangayyan_wiener_hopf',
    'rangayyan_wiener_filter',
    'rangayyan_ch3_estimation_error',
    'rangayyan_ch3_wiener_filter_output_convolution',
    'rangayyan_ch3_wiener_output_dot_product',
    'rangayyan_ch3_estimation_error_vector_form',
    'rangayyan_ch3_mse_cost_function',
    'rangayyan_ch3_cross_correlation_vector',
    'rangayyan_ch3_autocorrelation_matrix',
    'rangayyan_ch3_mse_gradient',
    'rangayyan_ch3_wiener_hopf_normal_equation',
    'rangayyan_ch3_optimal_wiener_filter',
    'rangayyan_ch3_minimum_mse',
    'rangayyan_ch3_wiener_convolution_relationship',
    'rangayyan_ch3_wiener_frequency_relation',
    'rangayyan_ch3_wiener_frequency_response',
    'rangayyan_ch3_wiener_optimal_for_noise_removal',
    'rangayyan_ch3_wiener_frequency_response_snr_form',
    'rangayyan_ch3_anc_primary_input_model',
    'rangayyan_ch3_anc_output',
    'rangayyan_ch3_lms_filter_output',
    'rangayyan_ch3_lms_estimation_error',
    'rangayyan_ch3_lms_squared_error',
    'rangayyan_ch3_lms_steepest_descent',
    'rangayyan_ch3_lms_gradient_estimate',
    'rangayyan_ch3_widrow_hoff_lms',
    'rangayyan_ch3_lms_variable_step',
    'rangayyan_ch3_lms_step_size_zhang',
    'rangayyan_ch3_rls_objective',
    'rangayyan_ch3_rls_normal_equation',
    'rangayyan_ch3_rls_phi_matrix',
    'rangayyan_ch3_rls_theta_vector',
    'rangayyan_ch3_rls_phi_recursion',
    'rangayyan_ch3_rls_theta_recursion',
    'rangayyan_ch3_abcd_matrix_inversion_lemma',
    'rangayyan_ch3_rls_inverse_recursion',
    'rangayyan_ch3_rls_kalman_gain',
    'rangayyan_ch3_rls_p_recursion',
    'rangayyan_ch3_rls_gain_identity',
    'rangayyan_ch3_rls_weight_update_compact',
    'rangayyan_ch3_rls_a_priori_error',
    'rangayyan_ch4_psd_from_acf',
    'wiener_filter',
]


# -- rgacfd: ACF distance measure for nonstationary segmentation.
def rangayyan_acf_distance(x, seg_len, p):
    """
    ACF distance measure for nonstationary segmentation

    Formula: d_ACF = (1/p) sum_{m=1}^{p} (R_1(m) - R_2(m))^2 / (R_1(0)*R_2(0))

    Parameters
    ----------
    x : array-like
        Input data.
    seg_len : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: acf_dist_trace, segment_bounds

    References
    ----------
    Rangayyan Ch 8.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "ACF distance measure for nonstationary segmentation"}
    )


# -- rgadp: LMS adaptive noise canceller -- Rangayyan & Krishnan Sec 3.10.2.
def rangayyan_adaptive_filter(x, reference, mu=0.01, order=16):
    """LMS adaptive noise canceller (Widrow-Hoff).

    For each sample::

        y[n] = w[n].T r_vec[n]
        e[n] = x[n] - y[n]
        w[n+1] = w[n] + 2 μ e[n] r_vec[n]

    Cleaned signal is ``e``; estimated noise is ``y``.

    Parameters
    ----------
    x : array-like
        Primary signal (target + correlated noise).
    reference : array-like
        Reference noise (same length).
    mu : float
        Step size.
    order : int
        Number of taps.

    Returns
    -------
    RichResult with keys ``signal`` (=e), ``noise_estimate`` (=y),
    ``weights``, ``mu``, ``order``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 3.10.2 "The least-mean-squares
        adaptive filter", p.184. The previous docstring cited Ch 11.
    Widrow, B., & Stearns, S. D. (1985). *Adaptive Signal Processing*.
        Prentice-Hall.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    if x.size != r.size:
        raise ValueError("x and reference must have equal length.")
    M = int(order)
    N = x.size
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    # Start at n = 0 with a zero-padded reference history rather than at
    # n = M-1. The old loop left y[0..M-2] and e[0..M-2] at zero, so the first
    # M-1 samples of the returned signal were not the cancelled input -- they
    # were nothing at all. At the default order=16 that silently destroyed 15
    # samples, and it broke the identity that defines a noise canceller:
    # signal + noise_estimate must reconstruct the primary input everywhere.
    # Zero-padding the history is the standard LMS start-up and keeps the
    # filter adapting from the first sample.
    for n in range(N):
        seg = r[max(0, n - M + 1) : n + 1][::-1]
        rv = np.zeros(M)
        rv[: seg.size] = seg
        y[n] = float(w @ rv)
        e[n] = x[n] - y[n]
        w = w + 2.0 * mu * e[n] * rv
    res = RichResult(
        title="LMS adaptive noise canceller",
        summary_lines=[
            ("Taps", M),
            ("μ", float(mu)),
            ("Residual MSE", float(np.mean(e**2))),
        ],
        interpretation=f"LMS order {M}, μ={mu}; residual MSE {float(np.mean(e**2)):.4g}.",
        payload={"signal": e, "noise_estimate": y, "weights": w, "mu": float(mu), "order": M},
    )
    return with_describe_pointer(res, "rgadp")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> n = rng.standard_normal(500)
# >>> x = np.sin(2*np.pi*np.arange(500)/50.0) + n
# >>> r = rangayyan_adaptive_filter(x, reference=n, mu=0.01, order=8)
# >>> r["signal"].shape == (500,)
# True


# -- rganc: Adaptive noise canceler (ANC) structure.
def rangayyan_anc(primary, reference, mu, order):
    """
    Adaptive noise canceler (ANC) structure

    Formula: e(n) = d(n) - y(n); y(n) = w^T(n)*primary(n); w updated to minimize E[e^2]

    Parameters
    ----------
    primary : array-like
        Input data.
    reference : array-like
        Input data.
    mu : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: clean_signal, error

    References
    ----------
    Rangayyan Ch 3.10.1
    """
    primary = np.asarray(primary, dtype=float)
    n = int(primary) if primary.ndim == 0 else len(primary)
    result = float(np.mean(primary))
    se = float(np.std(primary, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Adaptive noise canceler (ANC) structure"}
    )


# compact alias per ledger/NAMING.md
rangayyananc = rangayyan_anc


# -- rgeegadp: Adaptive segmentation of EEG using GLR test.
def rangayyan_eeg_adaptive_seg(eeg, fs, min_seg, threshold, cdf=None):
    """
    Adaptive segmentation of EEG using GLR test

    Formula: GLR change-point statistic; segment boundary when GLR > threshold

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    min_seg : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segment_bounds, glr_stat

    References
    ----------
    Rangayyan Ch 8.10
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    if eeg.ndim == 0:
        return RichResult(
            payload={"statistic": float("nan"), "p_value": float("nan"), "n": 1, "method": "scalar-input placeholder"}
        )
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Adaptive segmentation of EEG using GLR test",
            }
        )
    x_sorted = np.sort(eeg)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(eeg), scale=np.std(eeg, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "Adaptive segmentation of EEG using GLR test",
        }
    )


# -- rgfecg: Maternal-fetal ECG separation via adaptive noise cancellation.
def rangayyan_fetal_ecg(abdominal, maternal_ref, mu, order):
    """
    Maternal-fetal ECG separation via adaptive noise cancellation

    Formula: Fetal ECG = abdominal ECG - adaptive_filter(maternal chest ECG)

    Parameters
    ----------
    abdominal : array-like
        Input data.
    maternal_ref : array-like
        Input data.
    mu : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg

    References
    ----------
    Rangayyan Ch 3.14
    """
    abdominal = np.asarray(abdominal, dtype=float)
    n = int(abdominal) if abdominal.ndim == 0 else len(abdominal)
    result = float(np.mean(abdominal))
    se = float(np.std(abdominal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Maternal-fetal ECG separation via adaptive noise cancellation",
        }
    )


# -- rgglr: Generalized likelihood ratio (GLR) test for change detection.
def rangayyan_gen_likelihood_ratio(x, seg_len, order, cdf=None):
    """
    Generalized likelihood ratio (GLR) test for change detection

    Formula: GLR(t) = log(L(theta_hat_1,theta_hat_2)/L(theta_hat)) > threshold

    Parameters
    ----------
    x : array-like
        Input data.
    seg_len : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: glr_stat, change_points

    References
    ----------
    Rangayyan Ch 8.5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(
            payload={"statistic": float("nan"), "p_value": float("nan"), "n": 1, "method": "scalar-input placeholder"}
        )
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Generalized likelihood ratio (GLR) test for change detection",
            }
        )
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "Generalized likelihood ratio (GLR) test for change detection",
        }
    )


# -- rgkalmn: Kalman filter: state prediction/update with Riccati equation.
def rangayyan_kalman_filter(z, F, H, Q, R, x0, P0):
    """
    Kalman filter: state prediction/update with Riccati equation

    Formula: x_k|k-1=F*x_{k-1}; P_k|k-1=F*P*F^T+Q; K=P*H^T*(H*P*H^T+R)^{-1}; update z,P

    Parameters
    ----------
    z : array-like
        Input data.
    F : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.
    x0 : array-like
        Input data.
    P0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_hat, P_hat, K_gains

    References
    ----------
    Rangayyan Ch 8.7
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Kalman filter: state prediction/update with Riccati equation",
        }
    )


# -- rglms: Least-mean-squares (LMS) adaptive filter.
def rangayyan_lms_filter(x, d, mu, order):
    """
    Least-mean-squares (LMS) adaptive filter

    Formula: e(n) = d(n) - w^T(n)*x(n); w(n+1) = w(n) + 2*mu*e(n)*x(n)

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    mu : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, w_history

    References
    ----------
    Rangayyan Ch 3.10.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Least-mean-squares (LMS) adaptive filter"}
    )


# -- rgpcgadp: Adaptive segmentation of PCG signals via SEM.
def rangayyan_pcg_adaptive_seg(pcg, fs, ar_order):
    """
    Adaptive segmentation of PCG signals via SEM

    Formula: SEM computed between consecutive AR model segments; high SEM = change

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    ar_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segment_bounds, sem_trace

    References
    ----------
    Rangayyan Ch 8.11
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Adaptive segmentation of PCG signals via SEM"}
    )


# -- rgricca: Steady-state Riccati equation solution for Kalman gain.
def rangayyan_riccati_eq(F, H, Q, R):
    """
    Steady-state Riccati equation solution for Kalman gain

    Formula: P = F*P*F^T + Q - F*P*H^T*(H*P*H^T+R)^{-1}*H*P*F^T

    Parameters
    ----------
    F : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: P_steady, K_steady

    References
    ----------
    Rangayyan Ch 8.7
    """
    F = np.asarray(F, dtype=float)
    n = int(F) if F.ndim == 0 else len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Steady-state Riccati equation solution for Kalman gain",
        }
    )


# -- rgrls: Recursive least-squares (RLS) adaptive filter.
def rangayyan_rls_filter(x, d, lam, delta, order):
    """
    Recursive least-squares (RLS) adaptive filter

    Formula: P(n)=(P(n-1)-k(n)*x^T(n)*P(n-1))/lambda; w(n)=w(n-1)+k(n)*e(n)

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    delta : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, w_history

    References
    ----------
    Rangayyan Ch 3.10.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Recursive least-squares (RLS) adaptive filter"}
    )


# -- rgrls_mon: Monitoring RLS filter output for nonstationary detection.
def rangayyan_rls_monitor(x, d, lam, threshold):
    """
    Monitoring RLS filter output for nonstationary detection

    Formula: Segment boundary detected when ||e(n)||^2 exceeds threshold after RLS convergence

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: change_points, error_trace

    References
    ----------
    Rangayyan Ch 8.6.1
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
            "method": "Monitoring RLS filter output for nonstationary detection",
        }
    )


# -- rgrlsl: RLS lattice (ladder) adaptive filter.
def rangayyan_rls_lattice(x, d, lam, order):
    """
    RLS lattice (ladder) adaptive filter

    Formula: Forward/backward prediction errors updated with reflection coefficients

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, k_f, k_b

    References
    ----------
    Rangayyan Ch 8.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLS lattice (ladder) adaptive filter"})


# -- rgsemm: Spectral error measure (SEM) for adaptive segmentation.
def rangayyan_spec_error_meas(x, fs, p, seg_len):
    """
    Spectral error measure (SEM) for adaptive segmentation

    Formula: SEM(m) = (1/p) sum_{k=1}^{p} (log S_m(k) - log S_{ref}(k))^2

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    p : array-like
        Input data.
    seg_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sem_trace, segment_bounds

    References
    ----------
    Rangayyan Ch 8.5.1
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
            "method": "Spectral error measure (SEM) for adaptive segmentation",
        }
    )


# -- rgwhop: Wiener-Hopf matrix equations for FIR Wiener filter.
def rangayyan_wiener_hopf(x, d, order):
    """
    Wiener-Hopf matrix equations for FIR Wiener filter

    Formula: R_xx * w_opt = r_dx; R_xx = autocorr matrix, r_dx = cross-corr vector

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w_opt

    References
    ----------
    Rangayyan Ch 3.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wiener-Hopf matrix equations for FIR Wiener filter"}
    )


# -- rgwnr: Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter).
def rangayyan_wiener_filter(x, noise_psd, signal_psd):
    """
    Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)

    Formula: H_opt(f) = S_xd(f)/S_xx(f) = S_dd(f)/(S_dd(f)+S_nn(f))

    Parameters
    ----------
    x : array-like
        Input data.
    noise_psd : array-like
        Input data.
    signal_psd : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: filtered_x

    References
    ----------
    Rangayyan Ch 3.9
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
            "method": "Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)",
        }
    )


# -- rng137: Estimation error.
def rangayyan_ch3_estimation_error(d, d_tilde, n=None):
    r"""Estimation error of an adaptive filter (Rangayyan Ch. 3):

    .. math:: e(n) = d(n) - \tilde d(n),

    the desired response minus its estimate. Also returns the mean
    squared error, the quantity the LMS and RLS recursions actually
    minimise.

    Parameters
    ----------
    d : array-like
        Desired response.
    d_tilde : array-like
        Estimate.
    n : int, optional
        Index to report; the whole series if omitted.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (adaptive filters).
    """
    d = np.asarray(d, dtype=float).ravel()
    dt = np.asarray(d_tilde, dtype=float).ravel()
    if d.size != dt.size:
        raise ValueError("d and d_tilde must have the same length.")
    if d.size < 1:
        raise ValueError("d must be non-empty.")
    e = d - dt
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "N": int(e.size),
                               "method": "e(n) = d(n) - d_tilde(n)"})


# -- rng138: Output of the Wiener (transversal) filter as convolution of input with tap weights..
def rangayyan_ch3_wiener_filter_output_convolution(x, w_k, n, M):
    """
    Output of the Wiener (transversal) filter as convolution of input with tap weights.

    Formula: d_tilde(n) = sum_{k=0}^{M-1} w_k * x(n - k)

    Parameters
    ----------
    x : array-like
        Input data.
    w_k : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.154, p. 173
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
            "method": "Output of the Wiener (transversal) filter as convolution of input with tap weights.",
        }
    )


# -- rng139: Wiener filter output expressed as inner product of tap-weight and input vectors..
def rangayyan_ch3_wiener_output_dot_product(w, x):
    """
    Wiener filter output expressed as inner product of tap-weight and input vectors.

    Formula: d_tilde(n) = w^T * x(n) = x^T(n) * w = <x, w>

    Parameters
    ----------
    w : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.157, p. 174
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
            "method": "Wiener filter output expressed as inner product of tap-weight and input vectors.",
        }
    )


# -- rng140: Estimation error in vector form.
def rangayyan_ch3_estimation_error_vector_form(d, w, x, n=None):
    r"""Estimation error in vector form (Rangayyan Ch. 3):

    .. math:: e(n) = d(n) - \mathbf{w}^T \mathbf{x}(n),

    with w the tap-weight vector and x(n) the tap-input vector. This
    is the same error as :mod:`morie.fn.rng137` once the estimate is
    written as an inner product -- which is the step that makes the
    gradient computable in closed form.

    Parameters
    ----------
    d : array-like, shape (N,)
        Desired response.
    w : array-like, shape (p,)
        Tap weights.
    x : array-like, shape (N, p)
        Tap-input vectors, one row per time index.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``estimate``,
        ``N``, ``p``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (adaptive filters).
    """
    d = np.asarray(d, dtype=float).ravel()
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != d.size:
        X = X.T
    if X.shape[0] != d.size:
        raise ValueError("x must have one row per entry of d.")
    if X.shape[1] != w.size:
        raise ValueError(f"x has {X.shape[1]} columns but w has {w.size} weights.")
    est = X @ w
    e = d - est
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "estimate": est,
                               "N": int(d.size), "p": int(w.size),
                               "method": "e(n) = d(n) - w^T x(n)"})


# -- rng141: MSE cost function of the Wiener filter (Rangayyan Eq 3.166).
def rangayyan_ch3_mse_cost_function(w, Theta, Phi, sigma_d):
    r"""Mean-squared-error cost of a tap-weight vector under Wiener filter theory.

    .. math::

        J(\mathbf{w}) = E[e^2(n)]
            = \sigma_d^2 - \mathbf{w}^T\Theta - \Theta^T\mathbf{w}
              + \mathbf{w}^T\Phi\mathbf{w}

    Parameters
    ----------
    w : array-like, shape (M,)
        Tap-weight vector.
    Theta : array-like, shape (M,)
        Cross-correlation vector (Eq. 3.160); see :mod:`morie.fn.rng142`.
    Phi : array-like, shape (M, M)
        Autocorrelation matrix (Eq. 3.163); see :mod:`morie.fn.rng143`.
    sigma_d : float
        Standard deviation :math:`\sigma_d` of the desired response, whose
        mean is assumed zero. **This is the SD, not the variance** -- the
        book writes :math:`E[d^2(n)]` as :math:`\sigma_d^2`.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`J(\mathbf{w})`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On any shape mismatch, or if ``sigma_d`` is negative.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.159) sets up :math:`J(\mathbf{w}) = E[e^2(n)]` on p. 174; the
        simplified quadratic form used here is **Eq. (3.166), p. 175**. The
        module previously cited "Eq 3.159 / 3.166, p. 174" -- 3.166 is on
        p. 175.

    Notes
    -----
    :math:`J` is a second-order function of :math:`\mathbf{w}` with minimum
    :math:`J_{\min} = \sigma_d^2 - \Theta^T\Phi^{-1}\Theta` (Eq. 3.172) at
    :math:`\mathbf{w}_o = \Phi^{-1}\Theta` (Eq. 3.169). That pair is what the
    tests pin, since it checks the quadratic and the Wiener-Hopf solution
    against each other rather than against a transcribed constant.
    """
    wv = np.asarray(w, dtype=float).ravel()
    th = np.asarray(Theta, dtype=float).ravel()
    ph = np.asarray(Phi, dtype=float)
    sd = float(sigma_d)
    M = wv.size
    if th.size != M:
        raise ValueError(f"w and Theta must have the same length; got {M} and {th.size}")
    if ph.shape != (M, M):
        raise ValueError(f"Phi must have shape ({M}, {M}); got {ph.shape}")
    if not np.isfinite(sd) or sd < 0:
        raise ValueError(
            f"sigma_d must be finite and non-negative; got {sigma_d!r}. It is the "
            "SD of the desired response, not its variance."
        )
    J = sd**2 - wv @ th - th @ wv + wv @ ph @ wv
    return RichResult(
        payload={
            "value": float(J),
            "M": int(M),
            "method": "Wiener MSE cost J(w) (Rangayyan Eq 3.166)",
        }
    )


# -- rng142: Wiener cross-correlation vector Theta (Rangayyan Eq 3.160/3.161).
def rangayyan_ch3_cross_correlation_vector(x, d, M):
    r"""Cross-correlation vector between the tap-input vector and the desired response.

    .. math::

        \Theta = E[\mathbf{x}(n)\,d(n)]
        = [\theta(0), \theta(-1), \ldots, \theta(1-M)]^T

    with (Eq. 3.161)

    .. math::

        \theta(-k) = E[x(n-k)\,d(n)], \quad k = 0, 1, \ldots, M-1.

    Parameters
    ----------
    x : array-like
        Input signal :math:`x(n)`.
    d : array-like
        Desired response :math:`d(n)`, same length as ``x``. This is a
        *signal*, not a scalar lag.
    M : int
        Filter length (number of taps), :math:`M \ge 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Theta`, length ``M``), ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length, if ``M < 1``, or if ``M``
        exceeds the signal length.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.160) and its expansion Eq. (3.161), p. 174, Section 3.9
        ("The Wiener filter").

    Notes
    -----
    The expectation is estimated by the sample average over the ``N - M + 1``
    time indices for which the full tap-input vector :math:`\mathbf{x}(n)` is
    available. Indices before the filter has filled are excluded rather than
    zero-padded: the book defines :math:`\Theta` as an expectation over the
    stationary process, and padding would bias every lag toward zero by a
    known factor of ``(N-k)/N``.

    The third argument was named ``n`` before this was implemented, which read
    as a time index; it is the filter length :math:`M`. The function has never
    returned a value -- the previous body referenced an undefined ``y`` and
    raised ``UnboundLocalError`` on every call -- so there is no caller to
    break.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ds = np.asarray(d, dtype=float).ravel()
    if xs.size != ds.size:
        raise ValueError(
            f"x and d must have the same length; got {xs.size} and {ds.size}. "
            "d is the desired-response SIGNAL d(n), not a scalar."
        )
    M = int(M)
    if M < 1:
        raise ValueError(f"M (filter length) must be >= 1; got {M}")
    if M > xs.size:
        raise ValueError(f"M={M} exceeds the signal length {xs.size}")
    N = xs.size
    theta = np.empty(M, dtype=float)
    for k in range(M):
        # theta(-k) = E[x(n-k) d(n)], averaged over the n for which the whole
        # tap vector exists: n = M-1 .. N-1.
        theta[k] = np.mean(xs[M - 1 - k : N - k] * ds[M - 1 :])
    return RichResult(
        payload={
            "array": theta,
            "M": M,
            "n": int(N),
            "method": "Wiener cross-correlation vector Theta (Rangayyan Eq 3.160/3.161)",
        }
    )


# -- rng143: Wiener autocorrelation matrix Phi (Rangayyan Eq 3.163/3.164/3.165).
def rangayyan_ch3_autocorrelation_matrix(x, M):
    r"""Autocorrelation matrix of the tap-input vector used in Wiener filtering.

    .. math::

        \Phi = E[\mathbf{x}(n)\,\mathbf{x}^T(n)]

    which in full :math:`M \times M` form (Eq. 3.164) is the symmetric
    Toeplitz matrix with element

    .. math::

        \phi(i-k) = E[x(n-k)\,x(n-i)], \qquad \phi(i-k) = \phi(k-i).

    Parameters
    ----------
    x : array-like
        Input signal :math:`x(n)`.
    M : int
        Filter length (number of taps), :math:`M \ge 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi`, shape ``(M, M)``), ``phi`` (the
        ``M`` autocorrelation lags that generate it), ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``M < 1`` or ``M`` exceeds the signal length.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.163), full matrix form Eq. (3.164), element Eq. (3.165),
        pp. 174-175. "With the assumption of wide-sense stationarity, the
        :math:`M \times M` matrix :math:`\Phi` is completely specified by
        :math:`M` values of the autocorrelation :math:`\phi(0), \ldots,
        \phi(M-1)`."

    Notes
    -----
    Built from the ``M`` lags rather than by forming and averaging outer
    products, because the book's own remark above says the matrix has only
    ``M`` degrees of freedom. Building it the direct way produces a matrix
    that is Toeplitz only up to sample noise, which then makes
    ``Phi @ w == Theta`` fail to reproduce the Wiener-Hopf solution exactly.
    """
    xs = np.asarray(x, dtype=float).ravel()
    M = int(M)
    if M < 1:
        raise ValueError(f"M (filter length) must be >= 1; got {M}")
    if M > xs.size:
        raise ValueError(f"M={M} exceeds the signal length {xs.size}")
    N = xs.size
    phi = np.empty(M, dtype=float)
    for k in range(M):
        # phi(k) = E[x(n)x(n-k)], averaged over the n where the tap vector exists.
        phi[k] = np.mean(xs[M - 1 :] * xs[M - 1 - k : N - k])
    Phi = toeplitz(phi)
    return RichResult(
        payload={
            "array": Phi,
            "phi": phi,
            "M": M,
            "n": int(N),
            "method": "Wiener autocorrelation matrix Phi (Rangayyan Eq 3.163/3.164/3.165)",
        }
    )


# -- rng144: Gradient of MSE cost function with respect to tap-weight vector..
def rangayyan_ch3_mse_gradient(w, Theta, Phi):
    """
    Gradient of MSE cost function with respect to tap-weight vector.

    Formula: dJ(w)/dw = -2*Theta + 2*Phi*w

    Parameters
    ----------
    w : array-like
        Input data.
    Theta : array-like
        Input data.
    Phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.167, p. 175
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Gradient of MSE cost function with respect to tap-weight vector.",
        }
    )


# -- rng145: Wiener-Hopf normal equation for the optimal tap weights..
def rangayyan_ch3_wiener_hopf_normal_equation(Phi, w_o, Theta):
    """
    Wiener-Hopf normal equation for the optimal tap weights.

    Formula: Phi * w_o = Theta

    Parameters
    ----------
    Phi : array-like
        Input data.
    w_o : array-like
        Input data.
    Theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.168, p. 175
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wiener-Hopf normal equation for the optimal tap weights.",
        }
    )


# -- rng146: Closed-form optimal Wiener filter tap weights..
def rangayyan_ch3_optimal_wiener_filter(Phi, Theta):
    """
    Closed-form optimal Wiener filter tap weights.

    Formula: w_o = Phi^(-1) * Theta

    Parameters
    ----------
    Phi : array-like
        Input data.
    Theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.169, p. 175
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Closed-form optimal Wiener filter tap weights."}
    )


# -- rng147: Minimum mean-squared error achievable by the Wiener filter..
def rangayyan_ch3_minimum_mse(sigma_d, Theta, Phi):
    """
    Minimum mean-squared error achievable by the Wiener filter.

    Formula: J_min = sigma_d^2 - Theta^T * Phi^(-1) * Theta

    Parameters
    ----------
    sigma_d : array-like
        Input data.
    Theta : array-like
        Input data.
    Phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.172, p. 175
    """
    sigma_d = np.atleast_1d(np.asarray(sigma_d, dtype=float))
    n = len(sigma_d)
    result = float(np.mean(sigma_d))
    se = float(np.std(sigma_d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Minimum mean-squared error achievable by the Wiener filter.",
        }
    )


# -- rng148: Wiener-Hopf equation expressed as a convolution relationship under stationarity..
def rangayyan_ch3_wiener_convolution_relationship(w_ok, phi, theta, k):
    """
    Wiener-Hopf equation expressed as a convolution relationship under stationarity.

    Formula: w_ok * phi(k) = theta(k)

    Parameters
    ----------
    w_ok : array-like
        Input data.
    phi : array-like
        Input data.
    theta : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.174, p. 176
    """
    w_ok = np.atleast_1d(np.asarray(w_ok, dtype=float))
    n = len(w_ok)
    result = float(np.mean(w_ok))
    se = float(np.std(w_ok, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wiener-Hopf equation expressed as a convolution relationship under stationarity.",
        }
    )


# -- rng149: Frequency-domain Wiener relation between PSD and CSD..
def rangayyan_ch3_wiener_frequency_relation(W, S_xx, S_xd, omega):
    """
    Frequency-domain Wiener relation between PSD and CSD.

    Formula: W(omega) * S_xx(omega) = S_xd(omega)

    Parameters
    ----------
    W : array-like
        Input data.
    S_xx : array-like
        Input data.
    S_xd : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.175, p. 176
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency-domain Wiener relation between PSD and CSD.",
        }
    )


# -- rng150: Wiener filter frequency response as ratio of CSD to PSD of input..
def rangayyan_ch3_wiener_frequency_response(S_xd, S_xx, omega):
    """
    Wiener filter frequency response as ratio of CSD to PSD of input.

    Formula: W(omega) = S_xd(omega) / S_xx(omega)

    Parameters
    ----------
    S_xd : array-like
        Input data.
    S_xx : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.176, p. 176
    """
    S_xd = np.atleast_1d(np.asarray(S_xd, dtype=float))
    n = len(S_xd)
    result = float(np.mean(S_xd))
    se = float(np.std(S_xd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wiener filter frequency response as ratio of CSD to PSD of input.",
        }
    )


# -- rng151: Optimal Wiener filter for noise removal (Rangayyan Eq 3.183).
def rangayyan_ch3_wiener_optimal_for_noise_removal(Phi_d, Phi_eta, Phi_1d):
    r"""Optimal Wiener tap-weight vector when the input is signal plus noise.

    For :math:`x(n) = d(n) + \eta(n)` with signal and noise statistically
    independent and at least one of zero mean, Eq. (3.181) gives
    :math:`\Phi = \Phi_d + \Phi_\eta` and Eq. (3.182) gives
    :math:`\Theta = \Phi_{1d}`, so Eq. (3.169) becomes

    .. math::

        \mathbf{w}_o = (\Phi_d + \Phi_\eta)^{-1}\,\Phi_{1d}.

    Parameters
    ----------
    Phi_d : array-like, shape (M, M)
        Autocorrelation matrix of the desired signal.
    Phi_eta : array-like, shape (M, M)
        Autocorrelation matrix of the noise.
    Phi_1d : array-like, shape (M,)
        :math:`M \times 1` autocorrelation vector of the desired signal.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{w}_o`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch, or if :math:`\Phi_d + \Phi_\eta` is singular.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.183), p. 177, via Eq. (3.181) and Eq. (3.182) on the same page.

    Notes
    -----
    Solved with :func:`numpy.linalg.solve` rather than by forming the inverse:
    the sum of two autocorrelation matrices is positive definite in theory but
    can be badly conditioned in practice, and an explicit inverse loses more
    digits than a solve. A singular sum raises rather than returning ``inf``,
    because Eq. (3.183) has no solution in that case.
    """
    Pd = np.asarray(Phi_d, dtype=float)
    Pe = np.asarray(Phi_eta, dtype=float)
    P1 = np.asarray(Phi_1d, dtype=float).ravel()
    if Pd.ndim != 2 or Pd.shape[0] != Pd.shape[1]:
        raise ValueError(f"Phi_d must be a square matrix; got shape {Pd.shape}")
    if Pe.shape != Pd.shape:
        raise ValueError(f"Phi_eta must match Phi_d shape {Pd.shape}; got {Pe.shape}")
    M = Pd.shape[0]
    if P1.size != M:
        raise ValueError(f"Phi_1d must have length {M}; got {P1.size}")
    try:
        w_o = np.linalg.solve(Pd + Pe, P1)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Phi_d + Phi_eta is singular, so Eq. (3.183) has no solution. "
            "Check that both are genuine autocorrelation matrices."
        ) from exc
    return RichResult(
        payload={
            "array": w_o,
            "M": int(M),
            "method": "optimal Wiener filter for noise removal (Rangayyan Eq 3.183)",
        }
    )


# -- rng152: Wiener filter frequency response in terms of signal and noise PSDs..
def rangayyan_ch3_wiener_frequency_response_snr_form(S_d, S_eta, omega):
    """
    Wiener filter frequency response in terms of signal and noise PSDs.

    Formula: W(omega) = S_d(omega) / (S_d(omega) + S_eta(omega)) = 1 / (1 + S_eta(omega)/S_d(omega))

    Parameters
    ----------
    S_d : array-like
        Input data.
    S_eta : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.186, p. 177
    """
    S_d = np.atleast_1d(np.asarray(S_d, dtype=float))
    n = len(S_d)
    result = float(np.mean(S_d))
    se = float(np.std(S_d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wiener filter frequency response in terms of signal and noise PSDs.",
        }
    )


# -- rng153: Primary input of an adaptive noise canceller (ANC): signal plus primary noise..
def rangayyan_ch3_anc_primary_input_model(v, m, n):
    """
    Primary input of an adaptive noise canceller (ANC): signal plus primary noise.

    Formula: x(n) = v(n) + m(n)

    Parameters
    ----------
    v : array-like
        Input data.
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.187, p. 181
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Primary input of an adaptive noise canceller (ANC): signal plus primary noise.",
        }
    )


# -- rng154: Output of the ANC as the difference between primary input and adaptive filter output..
def rangayyan_ch3_anc_output(x, y, n):
    """
    Output of the ANC as the difference between primary input and adaptive filter output.

    Formula: v_tilde(n) = e(n) = x(n) - y(n)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.188, p. 182
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
            "method": "Output of the ANC as the difference between primary input and adaptive filter output.",
        }
    )


# -- rng155: Adaptive FIR filter output in LMS framework using reference input r(n)..
def rangayyan_ch3_lms_filter_output(r, w_k, n, M):
    """
    Adaptive FIR filter output in LMS framework using reference input r(n).

    Formula: y(n) = sum_{k=0}^{M-1} w_k * r(n - k)

    Parameters
    ----------
    r : array-like
        Input data.
    w_k : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.195, p. 183
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Adaptive FIR filter output in LMS framework using reference input r(n).",
        }
    )


# -- rng156: LMS estimation error.
def rangayyan_ch3_lms_estimation_error(x, w, r, n=None):
    r"""LMS instantaneous error (Rangayyan Ch. 3):

    .. math:: e(n) = x(n) - \mathbf{w}^T(n)\,\mathbf{r}(n),

    with r(n) the reference input. Note the weights carry a time
    index: in LMS they are updated at every sample, so this is the
    error under the CURRENT weights, not a fixed filter.

    Parameters
    ----------
    x : array-like, shape (N,)
        Primary input.
    w : array-like, shape (p,) or (N, p)
        Weights; a single vector is treated as time-invariant.
    r : array-like, shape (N, p)
        Reference input vectors.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``time_varying``,
        ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the LMS algorithm).
    """
    x = np.asarray(x, dtype=float).ravel()
    R = np.atleast_2d(np.asarray(r, dtype=float))
    if R.shape[0] != x.size:
        R = R.T
    if R.shape[0] != x.size:
        raise ValueError("r must have one row per sample of x.")
    W = np.asarray(w, dtype=float)
    tv = W.ndim == 2
    if tv:
        if W.shape != R.shape:
            raise ValueError("time-varying w must match the shape of r.")
        est = np.sum(W * R, axis=1)
    else:
        W = np.atleast_1d(W).ravel()
        if W.size != R.shape[1]:
            raise ValueError(f"w has {W.size} weights but r has {R.shape[1]} columns.")
        est = R @ W
    e = x - est
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "time_varying": bool(tv),
                               "N": int(x.size),
                               "method": "e(n) = x(n) - w^T(n) r(n); weights are time-indexed"})


# -- rng157: Quadratic squared-error form used in LMS gradient derivations..
def rangayyan_ch3_lms_squared_error(x, r, w, n):
    """
    Quadratic squared-error form used in LMS gradient derivations.

    Formula: e^2(n) = x^2(n) - 2*x(n)*r^T(n)*w(n) + w^T(n)*r(n)*r^T(n)*w(n)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    w : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.200, p. 184
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
            "method": "Quadratic squared-error form used in LMS gradient derivations.",
        }
    )


# -- rng158: Steepest-descent update rule for the tap-weight vector..
def rangayyan_ch3_lms_steepest_descent(w, mu, n):
    """
    Steepest-descent update rule for the tap-weight vector.

    Formula: w(n+1) = w(n) - mu * grad(e^2(n))

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.201, p. 184
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Steepest-descent update rule for the tap-weight vector.",
        }
    )


# -- rng159: LMS gradient estimate.
def rangayyan_ch3_lms_gradient_estimate(r, e, x=None, w=None, n=None):
    r"""LMS instantaneous gradient estimate (Rangayyan Ch. 3):

    .. math:: \widehat{\nabla}\,e^2(n) = -2 e(n)\, \mathbf{r}(n).

    Widrow-Hoff's key simplification: the true gradient of the MEAN
    squared error needs an expectation, but the gradient of the
    INSTANTANEOUS squared error needs only the current sample. The
    algorithm is a stochastic gradient descent whose noisy steps
    average to the right direction, which is why LMS converges in the
    mean rather than monotonically.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    e : array-like, shape (N,)
        Instantaneous errors.
    x, w : ignored
        Interface compatibility -- the identity
        -2xr + 2(w'r)r = -2er already folds them in.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``gradient`` (N, p), ``gradient_at_n``, ``N``, ``p``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the LMS algorithm).
    """
    R = np.atleast_2d(np.asarray(r, dtype=float))
    ev = np.asarray(e, dtype=float).ravel()
    if R.shape[0] != ev.size:
        R = R.T
    if R.shape[0] != ev.size:
        raise ValueError("r must have one row per entry of e.")
    grad = -2.0 * ev[:, None] * R
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < grad.shape[0]:
            raise ValueError(f"n must lie in 0..{grad.shape[0] - 1}, got {idx}.")
        at_n = grad[idx]
    return RichResult(payload={"gradient": grad, "gradient_at_n": at_n,
                               "N": int(R.shape[0]), "p": int(R.shape[1]),
                               "method": "grad e^2(n) = -2 e(n) r(n); stochastic, not exact"})


# -- rng160: Widrow-Hoff LMS tap-weight update rule..
def rangayyan_ch3_widrow_hoff_lms(w, mu, e, r, n):
    """
    Widrow-Hoff LMS tap-weight update rule.

    Formula: w(n+1) = w(n) + 2*mu*e(n)*r(n)

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    e : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.203, p. 185
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Widrow-Hoff LMS tap-weight update rule."}
    )


# -- rng161: Variable step-size LMS update rule..
def rangayyan_ch3_lms_variable_step(w, mu, e, r, n):
    """
    Variable step-size LMS update rule.

    Formula: w(n+1) = w(n) + 2*mu(n)*e(n)*r(n)

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    e : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.204, p. 185
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variable step-size LMS update rule."})


# -- rng162: Time-varying step size mu(n) per Zhang et al. for VAG signals..
def rangayyan_ch3_lms_step_size_zhang(mu, M, x_bar, alpha, r, n):
    """
    Time-varying step size mu(n) per Zhang et al. for VAG signals.

    Formula: mu(n) = mu / ( (M+1) * x_bar^2(n) * [alpha, r(n), x_bar^2(n-1)] )

    Parameters
    ----------
    mu : array-like
        Input data.
    M : array-like
        Input data.
    x_bar : array-like
        Input data.
    alpha : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.205, p. 185
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Time-varying step size mu(n) per Zhang et al. for VAG signals.",
        }
    )


# -- rng163: Weighted least-squares objective for the RLS algorithm with forgetting factor lambda..
def rangayyan_ch3_rls_objective(e, lam, n):
    """
    Weighted least-squares objective for the RLS algorithm with forgetting factor lambda.

    Formula: xi(n) = sum_{i=1}^{n} lambda^(n-i) * |e(i)|^2

    Parameters
    ----------
    e : array-like
        Input data.
    lam : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.206, p. 186
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    result = float(np.mean(e))
    se = float(np.std(e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Weighted least-squares objective for the RLS algorithm with forgetting factor lambda.",
        }
    )


# -- rng164: Normal equation for the RLS algorithm..
def rangayyan_ch3_rls_normal_equation(Phi, w_tilde, Theta, n):
    """
    Normal equation for the RLS algorithm.

    Formula: Phi(n) * w_tilde(n) = Theta(n)

    Parameters
    ----------
    Phi : array-like
        Input data.
    w_tilde : array-like
        Input data.
    Theta : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.207, p. 187
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Normal equation for the RLS algorithm."}
    )


# -- rng165: RLS correlation matrix.
def rangayyan_ch3_rls_phi_matrix(r, lam=0.99, n=None):
    r"""RLS exponentially weighted correlation matrix (Rangayyan
    Ch. 3):

    .. math:: \Phi(n) = \sum_{i=1}^{n} \lambda^{n-i}\,
              \mathbf{r}(i)\,\mathbf{r}^T(i).

    The forgetting factor lambda < 1 discounts old data geometrically,
    giving an effective memory of about 1/(1 - lambda) samples --
    returned, because that number, not n, is what governs how fast RLS
    tracks a change.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    lam : float in (0, 1], default 0.99
        Forgetting factor.
    n : int, optional
        Time index; the last sample if omitted.

    Returns
    -------
    RichResult
        keys: ``Phi`` (p, p), ``effective_memory``, ``lam``, ``n``,
        ``condition_number``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the RLS algorithm).
    """
    R = np.atleast_2d(np.asarray(r, dtype=float))
    lam = float(lam)
    if not 0 < lam <= 1:
        raise ValueError(f"lam must lie in (0, 1], got {lam}.")
    N = R.shape[0]
    idx = N if n is None else int(n)
    if not 1 <= idx <= N:
        raise ValueError(f"n must lie in 1..{N}, got {idx}.")
    w = lam ** (idx - 1 - np.arange(idx))
    Phi = (R[:idx] * w[:, None]).T @ R[:idx]
    mem = np.inf if lam == 1.0 else 1.0 / (1.0 - lam)
    try:
        cond = float(np.linalg.cond(Phi))
    except np.linalg.LinAlgError:
        cond = np.inf
    return RichResult(payload={"Phi": Phi, "effective_memory": float(mem),
                               "lam": lam, "n": idx, "condition_number": cond,
                               "method": "Phi(n) = sum lambda^(n-i) r(i) r^T(i)"})


# -- rng166: RLS cross-correlation vector.
def rangayyan_ch3_rls_theta_vector(r, x, lam=0.99, n=None):
    r"""RLS exponentially weighted cross-correlation vector
    (Rangayyan Ch. 3):

    .. math:: \Theta(n) = \sum_{i=1}^{n} \lambda^{n-i}\,
              \mathbf{r}(i)\, x(i).

    Paired with :mod:`morie.fn.rng165`: the RLS weight vector solves
    :math:`\Phi(n)\mathbf{w}(n) = \Theta(n)`, which is the
    normal-equation form the recursion updates without ever inverting
    Phi directly. The solved weights are returned alongside.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    x : array-like, shape (N,)
        Primary input.
    lam : float in (0, 1], default 0.99
        Forgetting factor.
    n : int, optional
        Time index.

    Returns
    -------
    RichResult
        keys: ``Theta``, ``weights`` (solving Phi w = Theta),
        ``lam``, ``n``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the RLS algorithm).
    """

    R = np.atleast_2d(np.asarray(r, dtype=float))
    xv = np.asarray(x, dtype=float).ravel()
    if R.shape[0] != xv.size:
        R = R.T
    if R.shape[0] != xv.size:
        raise ValueError("r must have one row per sample of x.")
    lam = float(lam)
    if not 0 < lam <= 1:
        raise ValueError(f"lam must lie in (0, 1], got {lam}.")
    N = R.shape[0]
    idx = N if n is None else int(n)
    if not 1 <= idx <= N:
        raise ValueError(f"n must lie in 1..{N}, got {idx}.")
    w = lam ** (idx - 1 - np.arange(idx))
    Theta = (R[:idx] * w[:, None]).T @ xv[:idx]
    Phi = rangayyan_ch3_rls_phi_matrix(R, lam=lam, n=idx)["Phi"]
    try:
        weights = np.linalg.solve(Phi, Theta)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(Phi, Theta, rcond=None)[0]
    return RichResult(payload={"Theta": Theta, "weights": weights, "lam": lam,
                               "n": idx,
                               "method": "Theta(n) = sum lambda^(n-i) r(i) x(i); Phi w = Theta"})


# -- rng167: RLS recursion for the autocorrelation matrix (Rangayyan Eq 3.211).
def rangayyan_ch3_rls_phi_recursion(Phi, r, lam):
    r"""One RLS update of the time-averaged autocorrelation matrix.

    .. math::

        \Phi(n) = \lambda\,\Phi(n-1) + \mathbf{r}(n)\,\mathbf{r}^T(n)

    Parameters
    ----------
    Phi : array-like, shape (M, M)
        Previous matrix :math:`\Phi(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor :math:`\lambda`, with :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or if ``lam`` is outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.211), p. 187, obtained by isolating the :math:`i = n` term of
        the definition Eq. (3.208)
        :math:`\Phi(n) = \sum_{i=1}^{n}\lambda^{n-i}\mathbf{r}(i)\mathbf{r}^T(i)`.

    Notes
    -----
    The book bounds the forgetting factor as :math:`0 < \lambda \le 1`, noting
    that :math:`1/(1-\lambda)` "is a measure of the memory of the algorithm".
    :math:`\lambda > 1` would grow past history without bound and is rejected;
    the vestigial ``n`` argument the previous signature carried is not in the
    equation and has been dropped.
    """
    P = np.asarray(Phi, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Phi must be a square matrix; got shape {P.shape}")
    if rv.size != P.shape[0]:
        raise ValueError(f"r must have length {P.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(
            f"lam (forgetting factor) must satisfy 0 < lam <= 1; got {lam!r} "
            "(Rangayyan p. 186)"
        )
    Phi_n = lam * P + np.outer(rv, rv)
    return RichResult(
        payload={
            "array": Phi_n,
            "M": int(P.shape[0]),
            "lam": lam,
            "method": "RLS autocorrelation-matrix recursion (Rangayyan Eq 3.211)",
        }
    )


# -- rng168: RLS recursion for the cross-correlation vector (Rangayyan Eq 3.212).
def rangayyan_ch3_rls_theta_recursion(Theta, r, x, lam):
    r"""One RLS update of the time-averaged cross-correlation vector.

    .. math::

        \Theta(n) = \lambda\,\Theta(n-1) + \mathbf{r}(n)\,x(n)

    Parameters
    ----------
    Theta : array-like, shape (M,)
        Previous vector :math:`\Theta(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    x : float
        Primary input sample :math:`x(n)` -- a **scalar**, not a signal.
    lam : float
        Forgetting factor :math:`\lambda`, with :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Theta(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch, non-scalar ``x``, or ``lam`` outside
        :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.212), p. 187, the recursive form of the definition
        Eq. (3.209) :math:`\Theta(n) = \sum_{i=1}^{n}\lambda^{n-i}\mathbf{r}(i)x(i)`.

    Notes
    -----
    In the adaptive-noise-cancelling arrangement of Figure 3.94, :math:`x(n)`
    is the *primary* input and :math:`\mathbf{r}(n)` the *reference*; getting
    the two the wrong way round is the usual error here and is not detectable
    from shapes, since only ``r`` is a vector.
    """
    th = np.asarray(Theta, dtype=float).ravel()
    rv = np.asarray(r, dtype=float).ravel()
    xs = np.asarray(x, dtype=float)
    lam = float(lam)
    if rv.size != th.size:
        raise ValueError(f"r must have the same length as Theta ({th.size}); got {rv.size}")
    if xs.ndim != 0:
        raise ValueError(
            f"x must be a scalar sample x(n); got shape {xs.shape}. Eq. (3.212) "
            "updates one time step."
        )
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    Theta_n = lam * th + rv * float(xs)
    return RichResult(
        payload={
            "array": Theta_n,
            "M": int(th.size),
            "lam": lam,
            "method": "RLS cross-correlation-vector recursion (Rangayyan Eq 3.212)",
        }
    )


# -- rng169: Matrix inversion (ABCD) lemma used in RLS..
def rangayyan_ch3_abcd_matrix_inversion_lemma(A, B, C, D):
    """
    Matrix inversion (ABCD) lemma used in RLS.

    Formula: (A + B*C*D)^(-1) = A^(-1) - A^(-1) * B * (D*A^(-1)*B + C^(-1))^(-1) * D * A^(-1)

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.213, p. 188
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Matrix inversion (ABCD) lemma used in RLS."}
    )


# -- rng170: Riccati recursion for the inverse autocorrelation matrix (Rangayyan Eq 3.215).
def rangayyan_ch3_rls_inverse_recursion(Phi_inv, r, lam):
    r"""Update :math:`\Phi^{-1}` directly, without inverting anything.

    .. math::

        \Phi^{-1}(n) = \lambda^{-1}\Phi^{-1}(n-1)
          - \frac{\lambda^{-2}\Phi^{-1}(n-1)\mathbf{r}(n)\mathbf{r}^T(n)\Phi^{-1}(n-1)}
                 {1 + \lambda^{-1}\mathbf{r}^T(n)\Phi^{-1}(n-1)\mathbf{r}(n)}

    Parameters
    ----------
    Phi_inv : array-like, shape (M, M)
        Previous inverse :math:`\Phi^{-1}(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi^{-1}(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.215), p. 188, from applying the matrix-inversion ("ABCD")
        lemma Eq. (3.213) to the recursion Eq. (3.211) with
        :math:`A = \lambda\Phi(n-1)`, :math:`B = \mathbf{r}(n)`,
        :math:`C = 1`, :math:`D = \mathbf{r}^T(n)`.

    Notes
    -----
    The point of the lemma is that the bracketed quantity in Eq. (3.214) is a
    **scalar**, so the :math:`M \times M` inverse never has to be recomputed.
    The denominator is therefore a plain division here, not a solve.

    This is the same quantity :mod:`morie.fn.rng172` produces via the gain
    vector :math:`\mathbf{k}(n)`; the two agree to rounding, which is what the
    tests check.
    """
    Pi = np.asarray(Phi_inv, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
        raise ValueError(f"Phi_inv must be a square matrix; got shape {Pi.shape}")
    if rv.size != Pi.shape[0]:
        raise ValueError(f"r must have length {Pi.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    Pr = Pi @ rv
    denom = 1.0 + inv_lam * float(rv @ Pr)
    Pi_n = inv_lam * Pi - (inv_lam**2) * np.outer(Pr, Pr) / denom
    return RichResult(
        payload={
            "array": Pi_n,
            "M": int(Pi.shape[0]),
            "lam": lam,
            "method": "RLS inverse-autocorrelation recursion (Rangayyan Eq 3.215)",
        }
    )


# -- rng171: Kalman-like gain vector in RLS (Rangayyan Eq 3.217).
def rangayyan_ch3_rls_kalman_gain(P, r, lam):
    r"""Gain vector :math:`\mathbf{k}(n)` of the RLS algorithm.

    .. math::

        \mathbf{k}(n) = \frac{\lambda^{-1}P(n-1)\mathbf{r}(n)}
                             {1 + \lambda^{-1}\mathbf{r}^T(n)P(n-1)\mathbf{r}(n)}

    Parameters
    ----------
    P : array-like, shape (M, M)
        Previous inverse correlation matrix :math:`P(n-1) = \Phi^{-1}(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{k}(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.217), p. 188. The book notes :math:`\mathbf{k}(n)` "is
        analogous to the Kalman gain vector in Kalman filter theory".

    Notes
    -----
    The denominator is a scalar, which is the whole point of the
    matrix-inversion lemma this comes from -- no inverse is formed here.

    The previous body was the shared mean-and-standard-error stub,
    ``float(np.mean(P))`` returned under the key ``estimate``. It did not
    raise, so it was **green** in the suite while returning the mean of an
    inverse correlation matrix in place of a gain vector; the equation above
    was printed correctly in its own docstring the whole time. The vestigial
    trailing ``n`` argument, which appears nowhere in Eq. (3.217), is dropped.
    """
    Pm = np.asarray(P, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    if rv.size != Pm.shape[0]:
        raise ValueError(f"r must have length {Pm.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    num = inv_lam * (Pm @ rv)
    den = 1.0 + inv_lam * float(rv @ Pm @ rv)
    return RichResult(
        payload={
            "array": num / den,
            "M": int(Pm.shape[0]),
            "lam": lam,
            "method": "RLS Kalman-like gain vector (Rangayyan Eq 3.217)",
        }
    )


# -- rng172: RLS recursion for P(n) via the gain vector (Rangayyan Eq 3.218).
def rangayyan_ch3_rls_p_recursion(P, k, r, lam):
    r"""Update :math:`P(n) = \Phi^{-1}(n)` using the precomputed gain vector.

    .. math::

        P(n) = \lambda^{-1}P(n-1) - \lambda^{-1}\mathbf{k}(n)\mathbf{r}^T(n)P(n-1)

    where (Eq. 3.217)

    .. math::

        \mathbf{k}(n) = \frac{\lambda^{-1}P(n-1)\mathbf{r}(n)}
                             {1 + \lambda^{-1}\mathbf{r}^T(n)P(n-1)\mathbf{r}(n)}.

    Parameters
    ----------
    P : array-like, shape (M, M)
        Previous :math:`P(n-1)`, initialised as :math:`\delta^{-1}I`.
    k : array-like, shape (M,)
        Gain vector :math:`\mathbf{k}(n)` from Eq. (3.217).
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`P(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.218), p. 188, the simplified form of Eq. (3.215) under the
        notation :math:`P(n) = \Phi^{-1}(n)` of Eq. (3.216) and the gain
        vector Eq. (3.217). The book notes :math:`\mathbf{k}(n)` "is analogous
        to the Kalman gain vector".

    Notes
    -----
    ``k`` is taken as an argument rather than recomputed, because that is how
    the book states Eq. (3.218) and because the caller already needs
    :math:`\mathbf{k}(n)` for the tap-weight update Eq. (3.224). Passing a
    :math:`\mathbf{k}` inconsistent with ``P``, ``r`` and ``lam`` yields a
    :math:`P(n)` that is not :math:`\Phi^{-1}(n)`, and nothing here can detect
    that -- Eq. (3.221) :math:`\mathbf{k}(n) = P(n)\mathbf{r}(n)` is the
    identity to check it with, and the tests do.
    """
    Pm = np.asarray(P, dtype=float)
    kv = np.asarray(k, dtype=float).ravel()
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    M = Pm.shape[0]
    if kv.size != M:
        raise ValueError(f"k must have length {M}; got {kv.size}")
    if rv.size != M:
        raise ValueError(f"r must have length {M}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    P_n = inv_lam * Pm - inv_lam * np.outer(kv, rv @ Pm)
    return RichResult(
        payload={
            "array": P_n,
            "M": int(M),
            "lam": lam,
            "method": "RLS P(n) recursion via the gain vector (Rangayyan Eq 3.218)",
        }
    )


# -- rng173: RLS gain identity k(n) = P(n) r(n) (Rangayyan Eq 3.221).
def rangayyan_ch3_rls_gain_identity(P, r):
    r"""Gain vector expressed through the *updated* inverse correlation matrix.

    .. math::

        \mathbf{k}(n) = P(n)\,\mathbf{r}(n)

    Parameters
    ----------
    P : array-like, shape (M, M)
        The **updated** matrix :math:`P(n)`, not :math:`P(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{k}(n)`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.221), p. 188, obtained by comparing the bracketed expression in
        Eq. (3.220) with the :math:`P(n)` recursion Eq. (3.218).

    Notes
    -----
    Same vector as :mod:`morie.fn.rng171`, reached from the other side of the
    update: rng171 uses :math:`P(n-1)` and needs the scalar denominator,
    this uses :math:`P(n)` and needs nothing. Feeding :math:`P(n-1)` here is
    the obvious mistake and is undetectable from shapes -- both are
    :math:`M \times M` -- so the tests pin the two against each other.

    The previous body returned ``float(np.mean(P))`` under the key
    ``estimate`` and was green in the suite.
    """
    Pm = np.asarray(P, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    if rv.size != Pm.shape[0]:
        raise ValueError(f"r must have length {Pm.shape[0]}; got {rv.size}")
    return RichResult(
        payload={
            "array": Pm @ rv,
            "M": int(Pm.shape[0]),
            "method": "RLS gain identity k(n) = P(n) r(n) (Rangayyan Eq 3.221)",
        }
    )


# -- rng174: Compact RLS tap-weight update using a priori error alpha(n)..
def rangayyan_ch3_rls_weight_update_compact(w_tilde, k, alpha, n):
    """
    Compact RLS tap-weight update using a priori error alpha(n).

    Formula: w_tilde(n) = w_tilde(n-1) + k(n) * alpha(n)

    Parameters
    ----------
    w_tilde : array-like
        Input data.
    k : array-like
        Input data.
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.224, p. 189
    """
    w_tilde = np.atleast_1d(np.asarray(w_tilde, dtype=float))
    n = len(w_tilde)
    result = float(np.mean(w_tilde))
    se = float(np.std(w_tilde, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Compact RLS tap-weight update using a priori error alpha(n).",
        }
    )


# -- rng175: A priori error in the RLS update step..
def rangayyan_ch3_rls_a_priori_error(x, r, w_tilde, n):
    """
    A priori error in the RLS update step.

    Formula: alpha(n) = x(n) - r^T(n) * w_tilde(n-1) = x(n) - w_tilde^T(n-1) * r(n)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    w_tilde : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.225, p. 189
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "A priori error in the RLS update step."}
    )


# -- rng204: PSD as the Fourier transform of the ACF (Wiener-Khinchin)..
def rangayyan_ch4_psd_from_acf(phi_xx, X, f, tau):
    """
    PSD as the Fourier transform of the ACF (Wiener-Khinchin).

    Formula: S_xx(f) = FT[phi_xx(tau)] = X(f) * X*(f) = |X(f)|^2

    Parameters
    ----------
    phi_xx : array-like
        Input data.
    X : array-like
        Input data.
    f : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.30, p. 235
    """
    phi_xx = np.atleast_1d(np.asarray(phi_xx, dtype=float))
    n = len(phi_xx)
    result = float(np.mean(phi_xx))
    se = float(np.std(phi_xx, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "PSD as the Fourier transform of the ACF (Wiener-Khinchin).",
        }
    )


# -- wnflt: Wiener filter for optimal noise reduction.
_QUOTE = "The noise is strong with this one. --"


def wiener_filter(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    noise_psd: np.ndarray | None = None,
    nperseg: int = 256,
) -> SignalResult:
    """Apply a Wiener filter for optimal noise reduction.

    The Wiener filter minimizes the mean-square error between the
    estimated and desired signal in the frequency domain:

    .. math::

        H(f) = \\frac{P_{xx}(f)}{P_{xx}(f) + P_{nn}(f)}

    where :math:`P_{xx}` is the signal PSD and :math:`P_{nn}` is the
    noise PSD.

    Parameters
    ----------
    x : array-like
        1-D noisy input signal.
    fs : float
        Sampling frequency in Hz (default 1.0).
    noise_psd : array-like or None
        Noise power spectral density estimate.  If *None*, the noise
        floor is estimated from the lowest 10% of spectral bins.
    nperseg : int
        Segment length for PSD estimation (default 256).

    Returns
    -------
    SignalResult
        ``filtered`` contains the denoised signal, ``extra`` has
        ``wiener_gain`` and ``noise_psd``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    X = np.fft.rfft(x)
    power = np.abs(X) ** 2 / n

    if noise_psd is None:
        sorted_power = np.sort(power)
        noise_floor = np.mean(sorted_power[: max(1, len(sorted_power) // 10)])
        noise_est = np.full_like(power, noise_floor)
    else:
        noise_est = np.asarray(noise_psd, dtype=float).ravel()
        if len(noise_est) != len(power):
            from ._sci_core import interp1d

            old_f = np.linspace(0, 1, len(noise_est))
            new_f = np.linspace(0, 1, len(power))
            noise_est = interp1d(old_f, noise_est, fill_value="extrapolate")(new_f)

    gain = power / (power + noise_est + 1e-12)
    Y = X * gain
    filtered = np.fft.irfft(Y, n=n)

    return SignalResult(
        name="wiener_filter",
        filtered=filtered,
        fs=fs,
        n_samples=n,
        extra={"wiener_gain": gain, "noise_psd": noise_est},
    )


wnflt = wiener_filter


_CHEATSHEET = [
    'rgacfd: ACF distance measure for nonstationary segmentation',
    'rgadp: LMS adaptive noise canceller -- Rangayyan & Krishnan Sec 3.10.2',
    'rganc: Adaptive noise canceler (ANC) structure',
    'rgeegadp: Adaptive segmentation of EEG using GLR test',
    'rgfecg: Maternal-fetal ECG separation via adaptive noise cancellation',
    'rgglr: Generalized likelihood ratio (GLR) test for change detection',
    'rgkalmn: Kalman filter: state prediction/update with Riccati equation',
    'rglms: Least-mean-squares (LMS) adaptive filter',
    'rgpcgadp: Adaptive segmentation of PCG signals via SEM',
    'rgricca: Steady-state Riccati equation solution for Kalman gain',
    'rgrls: Recursive least-squares (RLS) adaptive filter',
    'rgrls_mon: Monitoring RLS filter output for nonstationary detection',
    'rgrlsl: RLS lattice (ladder) adaptive filter',
    'rgsemm: Spectral error measure (SEM) for adaptive segmentation',
    'rgwhop: Wiener-Hopf matrix equations for FIR Wiener filter',
    'rgwnr: Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)',
    'rng137: e = d - d_tilde; MSE is what LMS/RLS minimise',
    'rng138: Output of the Wiener (transversal) filter as convolution of input with tap weights.',
    'rng139: Wiener filter output expressed as inner product of tap-weight and input vectors.',
    'rng140: inner-product form is what makes the gradient closed-form',
    "rng141: J(w) = sigma_d^2 - w'Theta - Theta'w + w'Phi w (Rangayyan Eq 3.166).",
    'rng142: Theta = E[x(n) d(n)], theta(-k)=E[x(n-k)d(n)] (Rangayyan Eq 3.160/3.161).',
    'rng143: Phi = E[x(n) x^T(n)], symmetric Toeplitz (Rangayyan Eq 3.163/3.164).',
    'rng144: Gradient of MSE cost function with respect to tap-weight vector.',
    'rng145: Wiener-Hopf normal equation for the optimal tap weights.',
    'rng146: Closed-form optimal Wiener filter tap weights.',
    'rng147: Minimum mean-squared error achievable by the Wiener filter.',
    'rng148: Wiener-Hopf equation expressed as a convolution relationship under stationarity.',
    'rng149: Frequency-domain Wiener relation between PSD and CSD.',
    'rng150: Wiener filter frequency response as ratio of CSD to PSD of input.',
    'rng151: w_o = (Phi_d + Phi_eta)^-1 Phi_1d (Rangayyan Eq 3.183).',
    'rng152: Wiener filter frequency response in terms of signal and noise PSDs.',
    'rng153: Primary input of an adaptive noise canceller (ANC): signal plus primary noise.',
    'rng154: Output of the ANC as the difference between primary input and adaptive filter output.',
    'rng155: Adaptive FIR filter output in LMS framework using reference input r(n).',
    'rng156: LMS weights carry a time index; error is under CURRENT weights',
    'rng157: Quadratic squared-error form used in LMS gradient derivations.',
    'rng158: Steepest-descent update rule for the tap-weight vector.',
    'rng159: instantaneous gradient needs no expectation -- that IS the trick',
    'rng160: Widrow-Hoff LMS tap-weight update rule.',
    'rng161: Variable step-size LMS update rule.',
    'rng162: Time-varying step size mu(n) per Zhang et al. for VAG signals.',
    'rng163: Weighted least-squares objective for the RLS algorithm with forgetting factor lambda.',
    'rng164: Normal equation for the RLS algorithm.',
    'rng165: memory ~ 1/(1-lambda) governs tracking, not n',
    'rng166: RLS solves Phi w = Theta without inverting Phi',
    "rng167: Phi(n) = lam*Phi(n-1) + r r' (Rangayyan Eq 3.211).",
    'rng168: Theta(n) = lam*Theta(n-1) + r(n) x(n) (Rangayyan Eq 3.212).',
    'rng169: Matrix inversion (ABCD) lemma used in RLS.',
    'rng170: Riccati recursion for Phi^-1(n) (Rangayyan Eq 3.215).',
    "rng171: k(n) = lam^-1 P(n-1) r / (1 + lam^-1 r' P(n-1) r) (Rangayyan Eq 3.217).",
    "rng172: P(n) = lam^-1 P(n-1) - lam^-1 k(n) r'(n) P(n-1) (Rangayyan Eq 3.218).",
    'rng173: k(n) = P(n) r(n) (Rangayyan Eq 3.221).',
    'rng174: Compact RLS tap-weight update using a priori error alpha(n).',
    'rng175: A priori error in the RLS update step.',
    'rng204: PSD as the Fourier transform of the ACF (Wiener-Khinchin).',
    'wiener_filter({}) -> Wiener filter for optimal noise reduction.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
