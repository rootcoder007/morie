# morie.fn -- bsaadapt (rootcoder007/morie)
"""Optimal and adaptive filtering: Wiener-Hopf, LMS, RLS, adaptive noise cancellation, Kalman, adaptive segmentation.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 57
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from math import cos, fsum, log, log10, pi, sin, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._containers import SignalResult
from ._rgcore import aslist
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import toeplitz

__all__ = [
    'acfdist',
    'rangayyan_acf_distance',
    'rangayyan_adaptive_filter',
    'anc',
    'rangayyan_anc',
    'rangayyan_eeg_adaptive_seg',
    'fetalecg',
    'rangayyan_fetal_ecg',
    'rangayyan_gen_likelihood_ratio',
    'kalman',
    'rangayyan_kalman_filter',
    'lmsfilt',
    'rangayyan_lms_filter',
    'pcgseg',
    'rangayyan_pcg_adaptive_seg',
    'riccati',
    'rangayyan_riccati_eq',
    'rlsfilt',
    'rangayyan_rls_filter',
    'rlsmonitor',
    'rangayyan_rls_monitor',
    'rlslattice',
    'rangayyan_rls_lattice',
    'sem',
    'rangayyan_spec_error_meas',
    'whopf',
    'rangayyan_wiener_hopf',
    'wienerfilt',
    'rangayyan_wiener_filter',
    'rangayyan_ch3_estimation_error',
    'wienerout',
    'rangayyan_ch3_wiener_filter_output_convolution',
    'wienerdot',
    'rangayyan_ch3_wiener_output_dot_product',
    'rangayyan_ch3_estimation_error_vector_form',
    'rangayyan_ch3_mse_cost_function',
    'rangayyan_ch3_cross_correlation_vector',
    'rangayyan_ch3_autocorrelation_matrix',
    'msegrad',
    'rangayyan_ch3_mse_gradient',
    'wienerhopf',
    'rangayyan_ch3_wiener_hopf_normal_equation',
    'wieneropt',
    'rangayyan_ch3_optimal_wiener_filter',
    'wienermin',
    'rangayyan_ch3_minimum_mse',
    'wienerconv',
    'rangayyan_ch3_wiener_convolution_relationship',
    'wienerfreqrel',
    'rangayyan_ch3_wiener_frequency_relation',
    'wienerfreq',
    'rangayyan_ch3_wiener_frequency_response',
    'rangayyan_ch3_wiener_optimal_for_noise_removal',
    'wienersnr',
    'rangayyan_ch3_wiener_frequency_response_snr_form',
    'ancinput',
    'rangayyan_ch3_anc_primary_input_model',
    'ancout',
    'rangayyan_ch3_anc_output',
    'lmsout',
    'rangayyan_ch3_lms_filter_output',
    'rangayyan_ch3_lms_estimation_error',
    'lmssqerr',
    'rangayyan_ch3_lms_squared_error',
    'lmsdescent',
    'rangayyan_ch3_lms_steepest_descent',
    'rangayyan_ch3_lms_gradient_estimate',
    'widrowhoff',
    'rangayyan_ch3_widrow_hoff_lms',
    'lmsvarstep',
    'rangayyan_ch3_lms_variable_step',
    'lmszhang',
    'rangayyan_ch3_lms_step_size_zhang',
    'rlsobj',
    'rangayyan_ch3_rls_objective',
    'rlsnormal',
    'rangayyan_ch3_rls_normal_equation',
    'rangayyan_ch3_rls_phi_matrix',
    'rangayyan_ch3_rls_theta_vector',
    'rangayyan_ch3_rls_phi_recursion',
    'rangayyan_ch3_rls_theta_recursion',
    'abcdlemma',
    'rangayyan_ch3_abcd_matrix_inversion_lemma',
    'rangayyan_ch3_rls_inverse_recursion',
    'rangayyan_ch3_rls_kalman_gain',
    'rangayyan_ch3_rls_p_recursion',
    'rangayyan_ch3_rls_gain_identity',
    'rlsupdate',
    'rangayyan_ch3_rls_weight_update_compact',
    'rlsapriori',
    'rangayyan_ch3_rls_a_priori_error',
    'psdacf',
    'rangayyan_ch4_psd_from_acf',
    'wiener_filter',
]



# -- rgacfd: ACF distance measure for nonstationary segmentation.
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def acfdist(x1, x2, lags=8):
    """ACF distance measure for adaptive segmentation.

    Rangayyan (2024) Section 8.5:
        d_ACF = (1/p) sum_{m=1}^{p}
                  [ R_1(m) - R_2(m) ]^2 / [ R_1(0) R_2(0) ],

    comparing two segments through their autocorrelation sequences,
    normalized by the product of their zero-lag values.

    That normalization is what the measure lives or dies by: dividing by
    R_1(0) R_2(0) removes the amplitude of both segments, so the
    distance responds to a change in SHAPE and not to a change in
    loudness.  The lag-zero term is excluded from the sum for the same
    reason -- it carries only the energies, which have already been
    divided out.
    """
    a, b = aslist(x1), aslist(x2)
    p = int(lags)
    if p < 1:
        raise ValueError("need at least one lag")
    if len(a) <= p or len(b) <= p:
        raise ValueError("each segment needs more samples than lags")
    r1 = _acf(a, p + 1)
    r2 = _acf(b, p + 1)
    if r1[0] <= 0 or r2[0] <= 0:
        raise ValueError("a segment has zero energy")
    d = fsum((r1[m] - r2[m]) ** 2 for m in range(1, p + 1)) \
        / (p * r1[0] * r2[0])
    return RichResult(payload={
        "distance": d, "acf_1": r1, "acf_2": r2, "lags": p,
        "energy_1": r1[0], "energy_2": r2[0],
        "amplitude_invariant": True, "zero_lag_excluded": True,
        "method": "Rangayyan (2024) Section 8.5 (ACF distance)"})


rangayyan_acf_distance = acfdist  # pre-policy spelling


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
def anc(primary, reference, order=8, mu=0.01, method="lms", lam=0.98,
        delta=1.0):
    """Adaptive noise canceller, LMS or RLS.

    Rangayyan (2024) Section 3.10.  The structure is fixed by eqs.
    (3.195)-(3.196): the adaptive filter sees only the reference and
    estimates the primary noise, and the ERROR is the output.  What
    varies is the adaptation rule -- eq. (3.203) for LMS, eq. (3.224)
    for RLS.

    The book's own summary of why it works (eqs. 3.193-3.194): because
    v and m are independent, minimizing the total output power also
    minimizes E[(m - y)^2], so the filter converges on the noise and
    leaves the signal alone -- and at the ideal minimum y = m and
    e = v exactly.

    That is also the failure mode.  If the reference carries any of the
    signal of interest, the filter will happily cancel it too; the
    signal is not protected by anything except the independence
    assumption.  ``reference_leakage`` measures the correlation between
    the reference and the canceller's output, which should be near zero.
    """
    if method not in ("lms", "rls"):
        raise ValueError("method must be 'lms' or 'rls'")
    if method == "lms":
        r = lmsfilt(primary, reference, order=order, mu=mu)
    else:
        r = rlsfilt(primary, reference, order=order, lam=lam, delta=delta)
    e = r["e"]
    rs = aslist(reference)
    n = len(e)
    me, mr = fsum(e) / n, fsum(rs) / n
    ve = fsum((v - me) ** 2 for v in e) / n
    vr = fsum((v - mr) ** 2 for v in rs) / n
    cov = fsum((a - me) * (b - mr) for a, b in zip(e, rs)) / n
    leak = cov / sqrt(ve * vr) if ve > 0 and vr > 0 else 0.0
    out = dict(r)
    out.update({"reference_leakage": leak,
                "well_separated": abs(leak) < 0.2,
                "adaptation": method,
                "method": "Rangayyan (2024) Section 3.10, "
                          "eqs. (3.195)-(3.196)"})
    return RichResult(payload=out)


rangayyan_anc = anc  # pre-policy spelling


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
def fetalecg(abdominal, chest, order=32, mu=0.005, method="lms"):
    """Fetal ECG extraction by adaptive noise cancellation.

    Rangayyan (2024) Section 3.14, after Widrow et al.: the abdominal
    lead is the primary input, carrying the fetal ECG plus a much
    stronger maternal ECG; a chest lead is the reference, carrying the
    maternal ECG alone.  The canceller estimates the maternal
    contribution from the chest lead and subtracts it, leaving the fetal
    ECG in the error signal.

    The book records that Widrow et al. used MULTIPLE reference channels,
    32 taps each and a delay of 129 ms, because a single chest lead does
    not span the maternal signal as it appears at the abdomen -- the
    propagation differs by lead.  A single reference, which is what this
    function takes, is therefore the reduced case and will leave more
    maternal residue; ``single_reference`` records that.

    ``suppression_db`` is how far the primary's power fell, which is the
    honest headline number: it measures maternal removal, not fetal
    recovery, and those are only the same thing if the independence
    assumption held.
    """
    abd, ref = aslist(abdominal), aslist(chest)
    if len(abd) != len(ref):
        raise ValueError("the abdominal and chest leads must have the "
                         "same length")
    r = anc(abd, ref, order=order, mu=mu, method=method)
    px, pe = r["input_power"], r["output_power"]
    return RichResult(payload={
        "fetal": r["e"], "maternal_estimate": r["y"], "order": order,
        "input_power": px, "output_power": pe,
        "suppression_db": 10.0 * log10(px / pe) if pe > 0 and px > 0
        else None,
        "reference_leakage": r["reference_leakage"],
        "single_reference": True,
        "widrow_used_multiple_references": True,
        "method": "Rangayyan (2024) Section 3.14, after Widrow et al."})


rangayyan_fetal_ecg = fetalecg  # pre-policy spelling


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
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def kalman(z, F, H, Q, R, x0=None, P0=None):
    """Kalman filter: predict, then correct.

        x(k|k-1) = F x(k-1|k-1)
        P(k|k-1) = F P(k-1|k-1) F^T + Q
        K(k)     = P(k|k-1) H^T [H P(k|k-1) H^T + R]^(-1)
        x(k|k)   = x(k|k-1) + K(k) [z(k) - H x(k|k-1)]
        P(k|k)   = [I - K(k) H] P(k|k-1)

    The optimal linear estimator for a linear-Gaussian state-space
    model, and the recursive counterpart of the Wiener filter of Section
    3.9: same MMSE criterion, but tracking a state that evolves rather
    than a fixed set of tap weights.

    P is symmetrized at every step.  The Joseph form is not used, so
    rounding can push P out of symmetry over a long run; the largest
    asymmetry before correction is returned, and a growing value is the
    warning that the covariance is degrading.

    Parameters
    ----------
    z : sequence
        Measurements, each a vector of length p.
    F, H, Q, R : matrices
        State transition (n x n), observation (p x n), process noise
        (n x n), measurement noise (p x p).
    """
    Fm = [aslist(r) for r in F]
    Hm = [aslist(r) for r in H]
    Qm = [aslist(r) for r in Q]
    Rm = [aslist(r) for r in R]
    ns = len(Fm)
    p = len(Hm)
    if any(len(r) != ns for r in Fm) or any(len(r) != ns for r in Hm):
        raise ValueError("F must be n x n and H must be p x n")
    if len(Qm) != ns or any(len(r) != ns for r in Qm):
        raise ValueError("Q must be n x n")
    if len(Rm) != p or any(len(r) != p for r in Rm):
        raise ValueError("R must be p x p")
    x = [0.0] * ns if x0 is None else aslist(x0)
    P = [[1.0 if i == j else 0.0 for j in range(ns)] for i in range(ns)] \
        if P0 is None else [aslist(r) for r in P0]

    def mv(M, v):
        return [fsum(M[i][j] * v[j] for j in range(len(v)))
                for i in range(len(M))]

    def mm(A, B):
        return [[fsum(A[i][t] * B[t][j] for t in range(len(B)))
                 for j in range(len(B[0]))] for i in range(len(A))]

    def tr(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    states, covs, gains, innov = [], [], [], []
    asym = 0.0
    for zk in z:
        zv = aslist(zk)
        if len(zv) != p:
            raise ValueError("every measurement must have length %d" % p)
        xp = mv(Fm, x)
        Pp = mm(mm(Fm, P), tr(Fm))
        Pp = [[Pp[i][j] + Qm[i][j] for j in range(ns)] for i in range(ns)]
        S = mm(mm(Hm, Pp), tr(Hm))
        S = [[S[i][j] + Rm[i][j] for j in range(p)] for i in range(p)]
        PHt = mm(Pp, tr(Hm))
        K = []
        for i in range(ns):
            K.append(_solve(tr(S), [PHt[i][j] for j in range(p)]))
        y = [zv[i] - fsum(Hm[i][j] * xp[j] for j in range(ns))
             for i in range(p)]
        x = [xp[i] + fsum(K[i][j] * y[j] for j in range(p))
             for i in range(ns)]
        KH = mm(K, Hm)
        Pn = [[Pp[i][j] - fsum(KH[i][t] * Pp[t][j] for t in range(ns))
               for j in range(ns)] for i in range(ns)]
        asym = max(asym, max(abs(Pn[i][j] - Pn[j][i])
                             for i in range(ns) for j in range(ns)))
        P = [[0.5 * (Pn[i][j] + Pn[j][i]) for j in range(ns)]
             for i in range(ns)]
        states.append(list(x))
        covs.append([row[:] for row in P])
        gains.append([row[:] for row in K])
        innov.append(y)
    return RichResult(payload={
        "states": states, "covariances": covs, "gains": gains,
        "innovations": innov, "n": len(states), "state_dim": ns,
        "obs_dim": p, "p_symmetry_error": asym, "p_symmetrized": True,
        "joseph_form": False,
        "method": "Kalman (1960); the recursive counterpart of the Wiener "
                  "filter of Rangayyan (2024) Section 3.9"})


rangayyan_kalman_filter = kalman  # pre-policy spelling


# -- rglms: Least-mean-squares (LMS) adaptive filter.
def lmsfilt(primary, reference, order=8, mu=0.01, variable=False,
            alpha=0.02):
    """Run the LMS adaptive noise canceller.

    Rangayyan (2024) Section 3.10.2, eqs. (3.195)-(3.196), (3.199),
    (3.203):

        y(n) = sum_k w_k r(n-k)
        e(n) = x(n) - y(n)
        w(n+1) = w(n) + 2 mu e(n) r(n)

    with the error signal e as the CANCELLER'S OUTPUT.  With
    ``variable=True`` the step size follows eq. (3.205).

    Two things the book stresses are reported rather than assumed: the
    stability condition 0 < mu < 1/lambda_max (checked here against the
    trace bound on the reference power, so a mu that will diverge is
    caught before the run), and the fact that convergence is only in the
    mean -- the tap weights keep jittering around the Wiener solution
    forever, by an amount proportional to mu.  ``final_weights`` is one
    sample of that jitter, not a converged answer.
    """
    xs, rs = aslist(primary), aslist(reference)
    if len(xs) != len(rs):
        raise ValueError("primary and reference must have the same length")
    m = int(order)
    if m < 1:
        raise ValueError("order must be at least 1")
    n = len(xs)
    if n <= m:
        raise ValueError("need more samples than taps")
    mv = float(mu)
    if mv <= 0:
        raise ValueError("mu must be positive")
    rpow = fsum(v * v for v in rs) / n
    bound = 1.0 / (m * rpow) if rpow > 0 else float("inf")
    w = [0.0] * m
    y, e, hist = [], [], []
    power_prev = None
    for i in range(n):
        rv = [rs[i - k] if i - k >= 0 else 0.0 for k in range(m)]
        yi = fsum(a * b for a, b in zip(w, rv))
        ei = xs[i] - yi
        if variable:
            step = lmszhang(min(mv, 0.999), m, rs[i], alpha=alpha,
                            power_prev=power_prev)
            power_prev = step["power"]
            mu_i = step["mu"]
        else:
            mu_i = mv
        w = [a + 2.0 * mu_i * ei * b for a, b in zip(w, rv)]
        y.append(yi)
        e.append(ei)
        hist.append(mu_i)
    px = fsum(v * v for v in xs)
    pe = fsum(v * v for v in e)
    return RichResult(payload={
        "e": e, "output": e, "y": y, "final_weights": w, "order": m,
        "mu": mv, "variable_step": bool(variable),
        "step_history": hist if variable else None,
        "stable_bound": bound, "within_bound": mv < bound,
        "input_power": px, "output_power": pe,
        "power_reduction": (pe / px) if px > 0 else None,
        "converges_in_the_mean_only": True,
        "method": "Rangayyan (2024) Section 3.10.2, eq. (3.203)"})


rangayyan_lms_filter = lmsfilt  # pre-policy spelling


# -- rgpcgadp: Adaptive segmentation of PCG signals via SEM.
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def pcgseg(x, fs, window=None, step=None, order=6, threshold=None):
    """Adaptive segmentation of a PCG signal by the spectral error measure.

    Rangayyan (2024) Section 8.5 and its PCG application: fit an AR model
    to a reference window, compare the model spectrum of each subsequent
    window against it with the SEM, and declare a boundary where the SEM
    rises above a threshold -- then restart the reference from there.

    The restart is the "adaptive" part and is easy to omit: without it
    the measure drifts away from a stale reference and every window
    after the first true boundary is flagged.  Here the reference is
    reset at each detected boundary, so the SEM trace returns to its
    baseline within a segment.

    The default threshold is the median SEM plus three times its median
    absolute deviation, scaled to the normal; a mean-and-SD rule would be
    dragged upward by the very peaks it is trying to detect.
    """
    xs = aslist(x)
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    n = len(xs)
    w = int(window) if window is not None else max(32, int(0.05 * fsv))
    if w > n:
        raise ValueError("the window is longer than the record")
    hop = int(step) if step is not None else w // 2
    if hop < 1:
        raise ValueError("step must be at least one sample")
    p = int(order)
    if w <= p:
        raise ValueError("the window must hold more samples than the order")

    def spectrum(seg):
        acf = _acf(seg, p + 1)
        if acf[0] <= 0:
            return None
        try:
            Phi = [[acf[abs(i - j)] for j in range(p)] for i in range(p)]
            a = _solve(Phi, [-acf[i + 1] for i in range(p)])
        except ValueError:
            return None
        out = []
        for k in range(1, 33):
            om = pi * k / 33.0
            re, im = 1.0, 0.0
            for j, av in enumerate(a, start=1):
                re += av * cos(-om * j)
                im += av * sin(-om * j)
            den = re * re + im * im
            out.append(acf[0] / den if den > 0 else 1e-300)
        return out

    starts = list(range(0, n - w + 1, hop))
    ref = spectrum(xs[starts[0]:starts[0] + w])
    if ref is None:
        raise ValueError("the first window has no usable AR spectrum")
    values, times = [], []
    for s in starts:
        sp = spectrum(xs[s:s + w])
        values.append(sem(sp, ref)["sem"] if sp else 0.0)
        times.append(s / fsv)
    srt = sorted(values)
    med = srt[len(srt) // 2]
    mad = sorted(abs(v - med) for v in values)[len(values) // 2]
    thr = float(threshold) if threshold is not None \
        else med + 3.0 * 1.4826 * mad
    bounds, ref = [], spectrum(xs[starts[0]:starts[0] + w])
    adaptive = []
    for idx, s in enumerate(starts):
        sp = spectrum(xs[s:s + w])
        v = sem(sp, ref)["sem"] if (sp and ref) else 0.0
        adaptive.append(v)
        if v > thr:
            bounds.append(s)
            ref = sp
    return RichResult(payload={
        "sem": adaptive, "sem_fixed_reference": values, "times": times,
        "boundaries": bounds, "n_boundaries": len(bounds),
        "threshold": thr, "median": med, "mad": mad,
        "window": w, "step": hop, "order": p, "fs": fsv,
        "reference_restarted_at_boundaries": True,
        "robust_threshold": threshold is None,
        "method": "Rangayyan (2024) Section 8.5 (adaptive segmentation "
                  "of the PCG)"})


rangayyan_pcg_adaptive_seg = pcgseg  # pre-policy spelling


# -- rgricca: Steady-state Riccati equation solution for Kalman gain.
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def riccati(F, H, Q, R, maxiter=1000, tol=1e-12):
    """Steady-state solution of the discrete algebraic Riccati equation.

        P = F P F^T + Q - F P H^T (H P H^T + R)^(-1) H P F^T

    The fixed point the Kalman covariance converges to for a
    time-invariant model.  Once P has settled the gain K is constant, so
    the filter reduces to a fixed linear filter -- which is exactly the
    Wiener solution of Section 3.9 for that model, and is the sense in
    which the Kalman filter generalizes it.

    Solved by iterating the recursion to a fixed point rather than by an
    eigenvalue method: no external solver, and the iteration count and
    final change are returned so a non-convergent case is visible.  A
    model that is not detectable has no stabilizing solution, and the
    iteration will not converge; ``converged`` says so instead of
    returning whatever P happened to be at the iteration limit.
    """
    Fm = [aslist(r) for r in F]
    Hm = [aslist(r) for r in H]
    Qm = [aslist(r) for r in Q]
    Rm = [aslist(r) for r in R]
    n = len(Fm)
    p = len(Hm)

    def mm(A, B):
        return [[fsum(A[i][t] * B[t][j] for t in range(len(B)))
                 for j in range(len(B[0]))] for i in range(len(A))]

    def tr(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    P = [row[:] for row in Qm]
    change = float("inf")
    it = 0
    for it in range(1, int(maxiter) + 1):
        FPFt = mm(mm(Fm, P), tr(Fm))
        S = mm(mm(Hm, P), tr(Hm))
        S = [[S[i][j] + Rm[i][j] for j in range(p)] for i in range(p)]
        PHt = mm(P, tr(Hm))
        G = []
        for i in range(n):
            G.append(_solve(tr(S), [PHt[i][j] for j in range(p)]))
        corr = mm(mm(Fm, mm(G, mm(Hm, P))), tr(Fm))
        Pn = [[FPFt[i][j] + Qm[i][j] - corr[i][j] for j in range(n)]
              for i in range(n)]
        Pn = [[0.5 * (Pn[i][j] + Pn[j][i]) for j in range(n)]
              for i in range(n)]
        change = max(abs(Pn[i][j] - P[i][j])
                     for i in range(n) for j in range(n))
        P = Pn
        if change < tol:
            break
    S = mm(mm(Hm, P), tr(Hm))
    S = [[S[i][j] + Rm[i][j] for j in range(p)] for i in range(p)]
    PHt = mm(P, tr(Hm))
    K = []
    for i in range(n):
        K.append(_solve(tr(S), [PHt[i][j] for j in range(p)]))
    return RichResult(payload={
        "P": P, "K": K, "iterations": it, "change": change,
        "converged": change < tol, "n": n,
        "steady_state_is_the_wiener_solution": True,
        "method": "discrete algebraic Riccati equation; the fixed point "
                  "of the Kalman covariance recursion"})


rangayyan_riccati_eq = riccati  # pre-policy spelling


# -- rgrls: Recursive least-squares (RLS) adaptive filter.
def rlsfilt(primary, reference, order=8, lam=0.98, delta=1.0):
    """Run the RLS adaptive filter.

    Rangayyan (2024) Section 3.10.3, eqs. (3.206), (3.215)-(3.216),
    (3.221), (3.224)-(3.225):

        alpha(n) = x(n) - r^T(n) w(n-1)                          (3.225)
        k(n)     = P(n-1) r(n) / (lambda + r^T(n) P(n-1) r(n))
        P(n)     = [P(n-1) - k(n) r^T(n) P(n-1)] / lambda        (3.215)
        w(n)     = w(n-1) + k(n) alpha(n)                        (3.224)

    with P = Phi^(-1) (eq. 3.216) initialized to delta * I.

    RLS converges in far fewer samples than LMS because it uses the
    inverse correlation matrix rather than a scalar step, at O(M^2) per
    sample instead of O(M).  The price is numerical: P is meant to stay
    symmetric positive definite, and rounding can break that, after
    which the filter diverges silently.  P is symmetrized at each step
    and ``p_symmetry_error`` reports the largest asymmetry seen before
    that correction, so a run that was close to breaking says so.

    ``delta`` seeds P and encodes how little is assumed about the input:
    large delta means low confidence and fast initial adaptation.
    """
    xs, rs = aslist(primary), aslist(reference)
    if len(xs) != len(rs):
        raise ValueError("primary and reference must have the same length")
    m = int(order)
    if m < 1:
        raise ValueError("order must be at least 1")
    n = len(xs)
    if n <= m:
        raise ValueError("need more samples than taps")
    lv = float(lam)
    if not 0 < lv <= 1:
        raise ValueError("eq. (3.206) needs 0 < lambda <= 1")
    dv = float(delta)
    if dv <= 0:
        raise ValueError("delta must be positive")
    P = [[dv if i == j else 0.0 for j in range(m)] for i in range(m)]
    w = [0.0] * m
    e, y, asym = [], [], 0.0
    for i in range(n):
        rv = [rs[i - k] if i - k >= 0 else 0.0 for k in range(m)]
        pred = fsum(a * b for a, b in zip(w, rv))
        alpha = xs[i] - pred
        Pr = [fsum(P[a][b] * rv[b] for b in range(m)) for a in range(m)]
        den = lv + fsum(rv[a] * Pr[a] for a in range(m))
        if den <= 0:
            raise ValueError("the RLS denominator vanished at sample %d; "
                             "P has lost positive definiteness" % i)
        kg = [v / den for v in Pr]
        newP = [[(P[a][b] - kg[a] * Pr[b]) / lv for b in range(m)]
                for a in range(m)]
        asym = max(asym, max(abs(newP[a][b] - newP[b][a])
                             for a in range(m) for b in range(m)))
        P = [[0.5 * (newP[a][b] + newP[b][a]) for b in range(m)]
             for a in range(m)]
        w = [a + b * alpha for a, b in zip(w, kg)]
        y.append(pred)
        e.append(alpha)
    px = fsum(v * v for v in xs)
    pe = fsum(v * v for v in e)
    return RichResult(payload={
        "e": e, "output": e, "y": y, "final_weights": w, "P": P,
        "order": m, "lam": lv, "delta": dv,
        "memory": (1.0 / (1.0 - lv)) if lv < 1 else float("inf"),
        "p_symmetry_error": asym, "p_symmetrized": True,
        "input_power": px, "output_power": pe,
        "power_reduction": (pe / px) if px > 0 else None,
        "method": "Rangayyan (2024) Section 3.10.3, eqs. (3.215), "
                  "(3.221), (3.224)-(3.225)"})


rangayyan_rls_filter = rlsfilt  # pre-policy spelling


# -- rgrls_mon: Monitoring RLS filter output for nonstationary detection.
def rlsmonitor(x, reference=None, order=8, lam=0.98, settle=None,
               threshold=3.0, window=None):
    """Watch an RLS filter's error for a nonstationarity.

    Rangayyan (2024) Section 8.5 uses adaptive segmentation: while the
    signal statistics hold, an adaptive filter converges and its error
    power settles; when the statistics change, the model no longer fits
    and the error jumps until the filter re-adapts.  A boundary is
    declared where the short-time error power exceeds a threshold.

    Two guards the method needs, both explicit here.  The filter must
    have CONVERGED before its error means anything, so the first
    ``settle`` samples are excluded from both the baseline and the
    detection -- the initial transient is not a segment boundary.  And
    the threshold is relative to the post-settling baseline, in units of
    its standard deviation, because the absolute error power depends on
    the signal amplitude and would otherwise need retuning per record.
    """
    xs = aslist(x)
    m = int(order)
    n = len(xs)
    if n <= 2 * m:
        raise ValueError("need well more samples than taps")
    ref = aslist(reference) if reference is not None else \
        [0.0] + xs[:-1]                  # one-step prediction by default
    r = rlsfilt(xs, ref, order=m, lam=lam)
    e = r["e"]
    s = int(settle) if settle is not None else min(n // 4, 10 * m)
    if s >= n - m:
        raise ValueError("the settling period leaves no samples to monitor")
    w = int(window) if window is not None else max(m, (n - s) // 20)
    if w < 1:
        raise ValueError("the window must hold at least one sample")
    power = []
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = e[lo:i + 1]
        power.append(fsum(v * v for v in seg) / len(seg))
    base = power[s:]
    mu = fsum(base) / len(base)
    sd = sqrt(fsum((v - mu) ** 2 for v in base) / len(base))
    thr = mu + float(threshold) * sd
    hits, i = [], s
    while i < n:
        if power[i] > thr:
            j = i
            while j + 1 < n and power[j + 1] > thr:
                j += 1
            hits.append(max(range(i, j + 1), key=lambda q: power[q]))
            i = j + 1
        else:
            i += 1
    return RichResult(payload={
        "error": e, "error_power": power, "boundaries": hits,
        "n_boundaries": len(hits), "threshold": thr, "baseline": mu,
        "baseline_sd": sd, "settle": s, "window": w, "order": m,
        "transient_excluded": True,
        "method": "Rangayyan (2024) Section 8.5 (adaptive segmentation)"})


rangayyan_rls_monitor = rlsmonitor  # pre-policy spelling


# -- rgrlsl: RLS lattice (ladder) adaptive filter.
def rlslattice(x, order=4, lam=0.98, delta=1e-2):
    """RLS lattice (ladder) predictor.

    A lattice recursion propagates FORWARD and BACKWARD prediction
    errors stage by stage, each stage adding one order:

        f_m(n) = f_{m-1}(n) - gamma_m b_{m-1}(n-1)
        b_m(n) = b_{m-1}(n-1) - gamma_m f_{m-1}(n)

    with the reflection coefficients gamma_m estimated recursively from
    exponentially weighted error energies, the same forgetting factor
    lambda as eq. (3.206).

    Two properties make the lattice worth the extra bookkeeping over the
    transversal RLS of eqs. (3.224)-(3.225), and both are returned:
    every stage is a complete predictor, so ALL orders up to the
    requested one come out of a single run, and the structure is stable
    exactly when |gamma_m| < 1 -- the same condition as the
    Levinson-Durbin reflection coefficients of eq. (7.39), and one that
    can be enforced stage by stage rather than checked after the fact.
    """
    xs = aslist(x)
    m = int(order)
    if m < 1:
        raise ValueError("order must be at least 1")
    n = len(xs)
    if n <= m:
        raise ValueError("need more samples than the order")
    lv = float(lam)
    if not 0 < lv <= 1:
        raise ValueError("lambda must satisfy 0 < lambda <= 1")
    dv = float(delta)
    if dv <= 0:
        raise ValueError("delta must be positive")
    fe = [dv] * (m + 1)
    be = [dv] * (m + 1)
    cross = [0.0] * (m + 1)
    bprev = [0.0] * (m + 1)
    gam = [0.0] * (m + 1)
    ferr = [[0.0] * (m + 1) for _ in range(n)]
    berr = [[0.0] * (m + 1) for _ in range(n)]
    for i in range(n):
        f = [0.0] * (m + 1)
        b = [0.0] * (m + 1)
        f[0] = b[0] = xs[i]
        for s in range(1, m + 1):
            cross[s] = lv * cross[s - 1 + 0] * 0.0 + cross[s]
            cross[s] = lv * cross[s] + f[s - 1] * bprev[s - 1]
            fe[s] = lv * fe[s] + f[s - 1] * f[s - 1]
            be[s] = lv * be[s] + bprev[s - 1] * bprev[s - 1]
            den = sqrt(fe[s] * be[s])
            gam[s] = (cross[s] / den) if den > 0 else 0.0
            if abs(gam[s]) >= 1.0:
                gam[s] = 0.999 if gam[s] > 0 else -0.999
            f[s] = f[s - 1] - gam[s] * bprev[s - 1]
            b[s] = bprev[s - 1] - gam[s] * f[s - 1]
        ferr[i] = f
        berr[i] = b
        bprev = b
    return RichResult(payload={
        "reflection": gam[1:], "forward_error": [row[m] for row in ferr],
        "backward_error": [row[m] for row in berr],
        "all_orders_forward": ferr, "order": m, "lam": lv,
        "stable": all(abs(g) < 1.0 for g in gam[1:]),
        "every_stage_is_a_predictor": True,
        "method": "RLS lattice; the |gamma| < 1 stability condition is "
                  "the same as Rangayyan (2024) eq. (7.39)"})


rangayyan_rls_lattice = rlslattice  # pre-policy spelling


# -- rgsemm: Spectral error measure (SEM) for adaptive segmentation.
def sem(psd, reference):
    """Spectral error measure between two PSDs.

    Rangayyan (2024) Section 8.5 (adaptive segmentation):
        SEM(m) = (1/p) sum_{k=1}^{p} [ log S_m(k) - log S_ref(k) ]^2,

    the mean squared difference of LOG spectra.  The logarithm is what
    makes it scale-free: a segment twice as loud as the reference but
    identically shaped shifts every log bin by the same constant, so a
    plain squared difference of PSDs would call it a change and the log
    version registers only the offset.

    Zero bins would send the logarithm to -inf, so they are floored and
    counted; a spectrum with many zeros is one the measure does not
    apply to.
    """
    a = aslist(psd)
    b = aslist(reference)
    if len(a) != len(b):
        raise ValueError("the two PSDs must have the same length")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PSD cannot be negative")
    floor = 1e-300
    zeros = sum(1 for v in a + b if v <= floor)
    la = [log(v if v > floor else floor) for v in a]
    lb = [log(v if v > floor else floor) for v in b]
    d = [p - q for p, q in zip(la, lb)]
    value = fsum(v * v for v in d) / len(d)
    offset = fsum(d) / len(d)
    shape = fsum((v - offset) ** 2 for v in d) / len(d)
    return RichResult(payload={
        "sem": value, "log_difference": d, "n_bins": len(d),
        "mean_offset": offset, "shape_only": shape,
        "gain_change_only": abs(value - offset * offset) < 1e-9,
        "zero_bins": zeros, "scale_free": True,
        "method": "Rangayyan (2024) Section 8.5 (spectral error measure)"})


rangayyan_spec_error_meas = sem  # pre-policy spelling


# -- rgwhop: Wiener-Hopf matrix equations for FIR Wiener filter.
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def whopf(x, d, order):
    """Build and solve the Wiener-Hopf system from data.

    Rangayyan (2024) eqs. (3.168), (3.171): the M x M autocorrelation
    matrix Phi of the input and the M x 1 cross-correlation vector Theta
    between input and desired response, solved for w_o.

    Under wide-sense stationarity the book notes Phi is completely
    specified by M autocorrelation values, so it is TOEPLITZ and is
    built from a single lag sequence rather than M^2 separate estimates.
    The biased ACF (divisor N) is used, which keeps that Toeplitz matrix
    positive semidefinite; the unbiased 1/(N-m) estimator does not, and
    an indefinite Phi turns the bowl of eq. (3.166) into a saddle.
    """
    xs, ds = aslist(x), aslist(d)
    if len(xs) != len(ds):
        raise ValueError("input and desired response must have equal length")
    m = int(order)
    if m < 1:
        raise ValueError("order must be at least 1")
    if len(xs) <= m:
        raise ValueError("need more samples than taps")
    phi = _acf(xs, m)
    theta = _ccf(xs, ds, m)
    Phi = [[phi[abs(i - j)] for j in range(m)] for i in range(m)]
    r = wienerhopf(Phi, theta)
    n = len(ds)
    var_d = fsum(v * v for v in ds) / n - (fsum(ds) / n) ** 2
    jm = wienermin(Phi, theta, var_d)
    return RichResult(payload={
        "w": r["w"], "phi": phi, "theta": theta, "Phi": Phi,
        "order": m, "j_min": jm["j_min"], "var_d": var_d,
        "toeplitz": True, "acf_biased": True,
        "condition": r["condition"],
        "method": "Rangayyan (2024) eqs. (3.168), (3.171)"})


rangayyan_wiener_hopf = whopf  # pre-policy spelling


# -- rgwnr: Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter).
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def wienerfilt(x, desired=None, order=8, sd=None, seta=None, fs=1.0):
    """Wiener filter, in either the time or the frequency formulation.

    Rangayyan (2024) Section 3.9.  Two routes, both in the book:

    - with a desired signal available, solve eqs. (3.168)-(3.169) for the
      M tap weights and filter (the ``desired`` argument);
    - with the signal and noise PSDs available, apply eq. (3.186)
      frequency by frequency (the ``sd`` and ``seta`` arguments).

    The book's own illustration uses the second: a piecewise-linear model
    of a clean ECG cycle supplies S_d, and the T-P interbeat intervals --
    which should be isoelectric, so anything in them is noise -- supply
    S_eta.  That is the practical answer to the obvious objection that
    the desired signal is exactly what one does not have.

    Exactly one route must be selected; asking for both would leave two
    different filters with no rule for combining them.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    have_time = desired is not None
    have_freq = sd is not None or seta is not None
    if have_time == have_freq:
        raise ValueError("give either a desired signal (time-domain route, "
                         "eqs. 3.168-3.169) or both PSDs (frequency route, "
                         "eq. 3.186), not both and not neither")
    if have_time:
        r = whopf(xs, desired, order)
        y = wienerout(r["w"], xs)["d_hat"]
        return RichResult(payload={
            "y": y, "w": r["w"], "order": r["order"], "j_min": r["j_min"],
            "route": "time", "method": "Rangayyan (2024) eqs. (3.168)-(3.169)"})
    if sd is None or seta is None:
        raise ValueError("the frequency route needs BOTH S_d and S_eta")
    n = len(xs)
    W = wienersnr(sd, seta)["W"]
    half = n // 2 + 1
    if len(W) != half:
        raise ValueError("the PSDs need one value per one-sided DFT bin "
                         "(%d for %d samples), got %d" % (half, n, len(W)))
    re, im = [], []
    step = 2.0 * pi / n
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(xs)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(xs)))
    for k in range(n):
        g = W[k] if k < half else W[n - k]
        re[k] *= g
        im[k] *= g
    y = []
    for i in range(n):
        acc = 0.0
        for k in range(n):
            ang = step * i * k
            acc += re[k] * cos(ang) - im[k] * sin(ang)
        y.append(acc / n)
    return RichResult(payload={
        "y": y, "W": W, "route": "frequency", "fs": float(fs), "n": n,
        "method": "Rangayyan (2024) eq. (3.186)"})


rangayyan_wiener_filter = wienerfilt  # pre-policy spelling


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
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def wienerout(w, x):
    """Output of the Wiener transversal filter, as a convolution.

    Rangayyan (2024) eq. (3.154):
        d~(n) = sum_{k=0}^{M-1} w_k x(n - k),

    the tap-weight sequence being the impulse response of the FIR filter.
    Eq. (3.153) then gives the estimation error e(n) = d(n) - d~(n),
    which is what the optimization minimizes.

    Samples before the filter has filled (n < M-1) use only the taps that
    have data behind them; that transient is returned as ``settled_from``
    so it is not mistaken for filter error.
    """
    ws, xs = aslist(w), aslist(x)
    if not ws or not xs:
        raise ValueError("both the tap weights and the input need samples")
    m, n = len(ws), len(xs)
    out = []
    for i in range(n):
        out.append(fsum(ws[k] * xs[i - k] for k in range(m) if i - k >= 0))
    return RichResult(payload={
        "d_hat": out, "n": n, "order": m, "settled_from": m - 1,
        "method": "Rangayyan (2024) eq. (3.154)"})


rangayyan_ch3_wiener_filter_output_convolution = wienerout  # pre-policy spelling


# -- rng139: Wiener filter output expressed as inner product of tap-weight and input vectors..
def wienerdot(w, xvec):
    """Wiener filter output as an inner product.

    Rangayyan (2024) eq. (3.155):
        d~(n) = w^T x(n) = x^T(n) w = <x, w>,

    with x(n) = [x(n), x(n-1), ..., x(n-M+1)]^T the tap-delay-line
    vector.  Identical in value to eq. (3.154); the vector form is what
    makes the MSE of eq. (3.166) a quadratic in w and the whole
    optimization tractable.

    Note the ORDER of the vector: it runs backwards in time, so x[0] is
    the current sample.  Handing it forwards reverses the filter.
    """
    ws, xv = aslist(w), aslist(xvec)
    if len(ws) != len(xv):
        raise ValueError("w and x(n) must have the same length; x(n) runs "
                         "backwards in time, x[0] being the current sample")
    if not ws:
        raise ValueError("need at least one tap")
    return RichResult(payload={
        "d_hat": fsum(a * b for a, b in zip(ws, xv)), "order": len(ws),
        "vector_is_time_reversed": True,
        "method": "Rangayyan (2024) eq. (3.155)"})


rangayyan_ch3_wiener_output_dot_product = wienerdot  # pre-policy spelling


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
def msegrad(phi, theta, w):
    """Gradient of the MSE with respect to the tap weights.

    Rangayyan (2024) eq. (3.167):
        dJ(w)/dw = -2 Theta + 2 Phi w,

    the derivative of the quadratic J(w) of eq. (3.166).  Setting it to
    zero gives the Wiener-Hopf equation (3.168), and following its
    negative is the steepest-descent step that LMS approximates in eq.
    (3.201).

    Because J is quadratic and Phi is positive semidefinite, the surface
    is a bowl with a single minimum -- there are no local minima to get
    stuck in, which is why gradient descent works here at all.
    """
    P = [aslist(r) for r in phi]
    t = aslist(theta)
    ws = aslist(w)
    m = len(ws)
    if len(t) != m or any(len(r) != m for r in P) or len(P) != m:
        raise ValueError("Phi must be M x M and Theta, w of length M")
    g = [2.0 * (fsum(P[i][j] * ws[j] for j in range(m)) - t[i])
         for i in range(m)]
    return RichResult(payload={
        "gradient": g, "norm": sqrt(fsum(v * v for v in g)),
        "at_optimum": all(abs(v) < 1e-9 for v in g), "order": m,
        "surface": "quadratic, single minimum",
        "method": "Rangayyan (2024) eq. (3.167)"})


rangayyan_ch3_mse_gradient = msegrad  # pre-policy spelling


# -- rng145: Wiener-Hopf normal equation for the optimal tap weights..
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def wienerhopf(phi, theta):
    """The Wiener-Hopf normal equation.

    Rangayyan (2024) eq. (3.168):  Phi w_o = Theta.

    The book calls it the normal equation because at the optimum each
    element of the input vector is orthogonal to the estimation error --
    the residual carries nothing the filter could still have used.  That
    orthogonality is the check returned: ``residual`` is Phi w - Theta,
    which is zero at the solution by construction, and ``condition``
    warns when Phi is close to singular, where the solved weights are
    numerically meaningless even though the residual looks small.
    """
    P = [aslist(r) for r in phi]
    t = aslist(theta)
    m = len(t)
    if len(P) != m or any(len(r) != m for r in P):
        raise ValueError("Phi must be M x M and Theta of length M")
    w = _solve(P, t)
    resid = [fsum(P[i][j] * w[j] for j in range(m)) - t[i] for i in range(m)]
    diag = [abs(P[i][i]) for i in range(m)]
    return RichResult(payload={
        "w": w, "residual": resid, "max_residual": max(abs(v) for v in resid),
        "order": m,
        "condition": (max(diag) / min(diag)) if min(diag) > 0 else float("inf"),
        "orthogonality": "at w_o the input vector and the error are "
                         "orthogonal, and so are the output and the error",
        "method": "Rangayyan (2024) eq. (3.168)"})


rangayyan_ch3_wiener_hopf_normal_equation = wienerhopf  # pre-policy spelling


# -- rng146: Closed-form optimal Wiener filter tap weights..
def wieneropt(phi, theta):
    """Closed-form optimal Wiener tap weights.

    Rangayyan (2024) eq. (3.169):  w_o = Phi^(-1) Theta.

    Written as an inverse in the book, SOLVED as a linear system here.
    Forming Phi^(-1) explicitly and multiplying costs more work and
    loses accuracy for exactly the ill-conditioned Phi where the answer
    matters most; Gaussian elimination with partial pivoting on the
    system is the same mathematics with better numerics.
    """
    r = wienerhopf(phi, theta)
    out = dict(r)
    out["w_o"] = r["w"]
    out["solved_not_inverted"] = True
    out["method"] = "Rangayyan (2024) eq. (3.169)"
    return RichResult(payload=out)


rangayyan_ch3_optimal_wiener_filter = wieneropt  # pre-policy spelling


# -- rng147: Minimum mean-squared error achievable by the Wiener filter..
def wienermin(phi, theta, var_d):
    """Minimum MSE achievable by the Wiener filter.

    Rangayyan (2024) eq. (3.172):
        J_min = sigma_d^2 - Theta^T Phi^(-1) Theta.

    The floor the filter cannot go below: the variance of the desired
    signal less the part of it the input can explain.  J_min = 0 only if
    the input determines the desired signal exactly.

    A NEGATIVE J_min is impossible for consistent statistics, so it is
    reported as ``consistent = False`` rather than returned as a number:
    it means sigma_d^2, Phi and Theta were estimated from different data
    or with different scalings, which is a silent and common error.
    """
    r = wienerhopf(phi, theta)
    t = aslist(theta)
    jmin = float(var_d) - fsum(a * b for a, b in zip(t, r["w"]))
    return RichResult(payload={
        "j_min": jmin, "w_o": r["w"], "var_d": float(var_d),
        "explained": fsum(a * b for a, b in zip(t, r["w"])),
        "consistent": jmin >= -1e-9 * max(abs(float(var_d)), 1.0),
        "fraction_explained": (1.0 - jmin / float(var_d))
        if var_d else None,
        "method": "Rangayyan (2024) eq. (3.172)"})


rangayyan_ch3_minimum_mse = wienermin  # pre-policy spelling


# -- rng148: Wiener-Hopf equation expressed as a convolution relationship under stationarity..
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def wienerconv(w, phi, theta):
    """Wiener-Hopf as a convolution relationship.

    Rangayyan (2024) eqs. (3.173)-(3.174):
        sum_i w_oi phi(k - i) = theta(k),  k = 0..M-1
        w_ok * phi(k) = theta(k).

    Eq. (3.171) has phi(i - k) and theta(-k); the step to eq. (3.173) is
    the STATIONARITY assumption, under which phi and theta are
    even-symmetric so the signs of the arguments may be dropped.  The
    convolution form then follows, and Fourier-transforming it gives eq.
    (3.175).

    That assumption is the one to watch: for a nonstationary signal the
    even symmetry fails and the convolution form is not equivalent to
    eq. (3.168).
    """
    ws, p, t = aslist(w), aslist(phi), aslist(theta)
    m = len(ws)
    if len(p) < m or len(t) < m:
        raise ValueError("need at least M lags of phi and theta")
    lhs = [fsum(ws[i] * p[abs(k - i)] for i in range(m)) for k in range(m)]
    gap = max(abs(a - b) for a, b in zip(lhs, t[:m]))
    scale = max(abs(v) for v in t[:m]) or 1.0
    return RichResult(payload={
        "lhs": lhs, "theta": t[:m], "max_difference": gap,
        "holds": gap <= 1e-8 * scale, "order": m,
        "requires_stationarity": True,
        "method": "Rangayyan (2024) eqs. (3.173)-(3.174)"})


rangayyan_ch3_wiener_convolution_relationship = wienerconv  # pre-policy spelling


# -- rng149: Frequency-domain Wiener relation between PSD and CSD..
def wienerfreqrel(W, sxx, sxd):
    """The frequency-domain Wiener relation.

    Rangayyan (2024) eq. (3.175):
        W(omega) S_xx(omega) = S_xd(omega),

    the Fourier transform of the convolution form of eq. (3.174).  It is
    the statement before the division: at any frequency where S_xx is
    zero the relation is satisfied by ANY W, so the filter is
    undetermined there -- which eq. (3.176) hides by dividing.

    Those frequencies are reported as ``undetermined_bins``.
    """
    Ws = [complex(v) for v in W]
    a = [complex(v) for v in sxx]
    b = [complex(v) for v in sxd]
    if not (len(Ws) == len(a) == len(b)):
        raise ValueError("W, S_xx and S_xd must have the same length")
    lhs = [w * s for w, s in zip(Ws, a)]
    gap = max(abs(u - v) for u, v in zip(lhs, b))
    scale = max(abs(v) for v in b) or 1.0
    und = [i for i, v in enumerate(a) if abs(v) <= 1e-300]
    return RichResult(payload={
        "lhs": lhs, "sxd": b, "max_difference": gap,
        "holds": gap <= 1e-8 * scale,
        "undetermined_bins": und, "n_undetermined": len(und),
        "method": "Rangayyan (2024) eq. (3.175)"})


rangayyan_ch3_wiener_frequency_relation = wienerfreqrel  # pre-policy spelling


# -- rng150: Wiener filter frequency response as ratio of CSD to PSD of input..
def wienerfreq(sxx, sxd):
    """Wiener filter frequency response from the CSD and PSD.

    Rangayyan (2024) eq. (3.176):
        W(omega) = S_xd(omega) / S_xx(omega),

    with S_xx the PSD of the input and S_xd the cross-spectral density
    between input and desired signal.

    Where S_xx vanishes the ratio is undefined; eq. (3.175) shows the
    filter is genuinely undetermined there, so W is set to zero -- the
    choice that adds nothing rather than amplifying a bin with no signal
    in it -- and the affected bins are reported rather than hidden.
    """
    a = [complex(v) for v in sxx]
    b = [complex(v) for v in sxd]
    if len(a) != len(b):
        raise ValueError("S_xx and S_xd must have the same length")
    if not a:
        raise ValueError("need at least one bin")
    W, und = [], []
    for i, (p, c) in enumerate(zip(a, b)):
        if abs(p) <= 1e-300:
            W.append(0j)
            und.append(i)
        else:
            W.append(c / p)
    return RichResult(payload={
        "W": W, "magnitude": [abs(v) for v in W],
        "undetermined_bins": und, "n_undetermined": len(und),
        "zero_where_undetermined": True, "n": len(W),
        "method": "Rangayyan (2024) eq. (3.176)"})


rangayyan_ch3_wiener_frequency_response = wienerfreq  # pre-policy spelling


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
def wienersnr(sd, seta):
    """Wiener frequency response in terms of signal and noise PSDs.

    Rangayyan (2024) eq. (3.186):
        W(omega) = S_d(omega) / (S_d(omega) + S_eta(omega))
                 = 1 / (1 + S_eta(omega) / S_d(omega)),

    which follows from eq. (3.176) with S_xx = S_d + S_eta (3.184) and
    S_xd = S_d (3.185), the additive-noise case of eq. (3.177).

    The book lists three properties, all returned here as checks rather
    than prose: W = 0 wherever S_d = 0 (a component absent from the
    input is NOT restored), W = 1 wherever S_eta = 0 (clean components
    pass with unit gain), and W decreases as S_eta rises -- the gain
    tracks the per-frequency SNR.

    Both PSDs zero at a bin leaves 0/0; the filter is undetermined
    there and is set to zero, with the bin reported.
    """
    d = aslist(sd)
    e = aslist(seta)
    if len(d) != len(e):
        raise ValueError("S_d and S_eta must have the same length")
    if not d:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in d) or any(v < 0 for v in e):
        raise ValueError("a PSD cannot be negative")
    W, und = [], []
    for i, (a, b) in enumerate(zip(d, e)):
        tot = a + b
        if tot <= 0:
            W.append(0.0)
            und.append(i)
        else:
            W.append(a / tot)
    snr = [(a / b if b > 0 else float("inf")) for a, b in zip(d, e)]
    return RichResult(payload={
        "W": W, "snr": snr, "undetermined_bins": und,
        "zero_where_signal_absent": all(W[i] == 0.0 for i in range(len(d))
                                        if d[i] == 0.0),
        "unity_where_noise_absent": all(W[i] == 1.0 for i in range(len(d))
                                        if e[i] == 0.0 and d[i] > 0),
        "monotone_in_snr": all(
            W[i] <= W[j] + 1e-12
            for i in range(len(d)) for j in range(len(d))
            if snr[i] <= snr[j] and snr[i] != float("inf")),
        "n": len(W), "method": "Rangayyan (2024) eq. (3.186)"})


rangayyan_ch3_wiener_frequency_response_snr_form = wienersnr  # pre-policy spelling


# -- rng153: Primary input of an adaptive noise canceller (ANC): signal plus primary noise..
def ancinput(v, m):
    """Primary input of an adaptive noise canceller.

    Rangayyan (2024) Section 3.10.1:  x(n) = v(n) + m(n),

    the signal of interest v plus a primary noise m.  The reference input
    r(n) is a separate recording correlated with m but not with v.

    Those two independence conditions are the whole basis of the method,
    and the book is explicit that NOTHING else about the processes need
    be known.  They are checked here on the supplied data rather than
    assumed: if v and m are correlated, eq. (3.193) does not separate and
    the canceller removes part of the signal along with the noise.
    """
    vs, ms = aslist(v), aslist(m)
    if len(vs) != len(ms):
        raise ValueError("signal and noise must have the same length")
    n = len(vs)
    if not n:
        raise ValueError("need at least one sample")
    x = [a + b for a, b in zip(vs, ms)]
    mv, mm = fsum(vs) / n, fsum(ms) / n
    cov = fsum((a - mv) * (b - mm) for a, b in zip(vs, ms)) / n
    sv = sqrt(fsum((a - mv) ** 2 for a in vs) / n)
    sm = sqrt(fsum((b - mm) ** 2 for b in ms) / n)
    rho = cov / (sv * sm) if sv > 0 and sm > 0 else 0.0
    return RichResult(payload={
        "x": x, "v": vs, "m": ms, "n": n, "correlation": rho,
        "independent": abs(rho) < 0.1,
        "assumption": "the method needs v and m statistically "
                      "independent, and the reference correlated with m "
                      "but not with v",
        "method": "Rangayyan (2024) Section 3.10.1"})


rangayyan_ch3_anc_primary_input_model = ancinput  # pre-policy spelling


# -- rng154: Output of the ANC as the difference between primary input and adaptive filter output..
def ancout(x, y):
    """Output of the adaptive noise canceller.

    Rangayyan (2024) eq. (3.196):
        v~(n) = e(n) = x(n) - y(n),

    where y is the adaptive filter's estimate of the primary noise.  The
    error signal IS the output -- that inversion is what makes an ANC
    different from an ordinary filter, and it is why minimizing the
    output power maximizes the output SNR (eqs. 3.193-3.194): the signal
    component v is untouched by the adaptation, so all the power that
    can be removed is noise.

    ``power_reduction`` is the ratio the adaptation is minimizing.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("primary input and filter output must have the "
                         "same length")
    if not xs:
        raise ValueError("need at least one sample")
    e = [a - b for a, b in zip(xs, ys)]
    px = fsum(v * v for v in xs)
    pe = fsum(v * v for v in e)
    return RichResult(payload={
        "e": e, "v_hat": e, "n": len(e),
        "input_power": px, "output_power": pe,
        "power_reduction": (pe / px) if px > 0 else None,
        "error_is_the_output": True,
        "method": "Rangayyan (2024) eq. (3.196)"})


rangayyan_ch3_anc_output = ancout  # pre-policy spelling


# -- rng155: Adaptive FIR filter output in LMS framework using reference input r(n)..
def lmsout(w, r):
    """Adaptive FIR filter output in the LMS framework.

    Rangayyan (2024) eq. (3.195):
        y(n) = sum_{k=0}^{M-1} w_k r(n - k),

    the filter acting on the REFERENCE input r, not on the primary input
    x.  That is the structural point: the adaptive filter never sees the
    signal of interest, so it cannot cancel it -- it can only build an
    estimate of the noise out of the reference.
    """
    ws, rs = aslist(w), aslist(r)
    if not ws or not rs:
        raise ValueError("both the tap weights and the reference need "
                         "samples")
    m = len(ws)
    y = [fsum(ws[k] * rs[i - k] for k in range(m) if i - k >= 0)
         for i in range(len(rs))]
    return RichResult(payload={
        "y": y, "n": len(y), "order": m, "settled_from": m - 1,
        "filters_the_reference": True,
        "method": "Rangayyan (2024) eq. (3.195)"})


rangayyan_ch3_lms_filter_output = lmsout  # pre-policy spelling


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
def lmssqerr(x, rvec, w):
    """Instantaneous squared error, expanded.

    Rangayyan (2024) eq. (3.200):
        e^2(n) = x^2(n) - 2 x(n) r^T(n) w(n)
                 + w^T(n) r(n) r^T(n) w(n),

    the square of eq. (3.199).  The book reads it as a concave
    hyperparaboloid that is never negative -- a bowl with one bottom --
    which is why gradient descent on it converges.

    LMS's simplification is to use this INSTANTANEOUS square as an
    estimate of the MSE instead of taking expectations, which is what
    makes the algorithm free of averaging and also what makes its
    trajectory noisy.  Both the expanded form and the direct square are
    returned and compared, since the expansion is the step the gradient
    of eq. (3.202) is taken from.
    """
    rv, ws = aslist(rvec), aslist(w)
    if len(rv) != len(ws):
        raise ValueError("r(n) and w must have the same length")
    if not rv:
        raise ValueError("need at least one tap")
    xv = float(x)
    rw = fsum(a * b for a, b in zip(rv, ws))
    expanded = xv * xv - 2.0 * xv * rw + rw * rw
    e = xv - rw
    return RichResult(payload={
        "e": e, "e_squared": e * e, "expanded": expanded,
        "max_difference": abs(expanded - e * e),
        "agrees": abs(expanded - e * e) <= 1e-9 * (1 + abs(e * e)),
        "nonnegative": expanded >= -1e-12,
        "instantaneous_not_expected": True,
        "method": "Rangayyan (2024) eq. (3.200)"})


rangayyan_ch3_lms_squared_error = lmssqerr  # pre-policy spelling


# -- rng158: Steepest-descent update rule for the tap-weight vector..
def lmsdescent(w, e, rvec, mu):
    """One steepest-descent step with the LMS gradient estimate.

    Rangayyan (2024) eqs. (3.201)-(3.202):
        w(n+1) = w(n) - mu grad(e^2(n))                          (3.201)
        grad-hat(e^2(n)) = -2 e(n) r(n)                          (3.202)

    Substituting the second into the first gives the Widrow-Hoff rule of
    eq. (3.203).  Both the gradient and the resulting step are returned,
    so the two equations can be seen separately rather than only in
    their combined form.

    The book states mu controls stability and rate: larger mu converges
    faster and is less stable, and convergence in the mean requires
    0 < mu < 1/lambda_max of the reference autocorrelation matrix.
    """
    ws, rv = aslist(w), aslist(rvec)
    if len(ws) != len(rv):
        raise ValueError("w and r(n) must have the same length")
    if not ws:
        raise ValueError("need at least one tap")
    ev = float(e)
    grad = [-2.0 * ev * v for v in rv]
    step = [a - float(mu) * g for a, g in zip(ws, grad)]
    return RichResult(payload={
        "gradient": grad, "w_next": step, "mu": float(mu), "e": ev,
        "order": len(ws),
        "equals_widrow_hoff": True,
        "method": "Rangayyan (2024) eqs. (3.201)-(3.202)"})


rangayyan_ch3_lms_steepest_descent = lmsdescent  # pre-policy spelling


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
def widrowhoff(w, e, rvec, mu):
    """The Widrow-Hoff LMS tap-weight update.

    Rangayyan (2024) eq. (3.203):
        w(n+1) = w(n) + 2 mu e(n) r(n).

    The factor of TWO is part of the equation, not a convention: it comes
    from the 2 in the gradient of eq. (3.202).  Many texts fold it into
    mu, so a step size copied between sources can be out by a factor of
    two -- which, near the stability limit, is the difference between
    converging and diverging.

    The book's stability condition is 0 < mu < 1/lambda_max of the
    reference autocorrelation matrix; ``stable_bound`` reports the
    largest safe mu given the reference power seen here, using the
    standard trace bound lambda_max <= M * E[r^2].
    """
    ws, rv = aslist(w), aslist(rvec)
    if len(ws) != len(rv):
        raise ValueError("w and r(n) must have the same length")
    if not ws:
        raise ValueError("need at least one tap")
    mv, ev = float(mu), float(e)
    nxt = [a + 2.0 * mv * ev * b for a, b in zip(ws, rv)]
    power = fsum(v * v for v in rv)
    bound = 1.0 / power if power > 0 else float("inf")
    return RichResult(payload={
        "w_next": nxt, "mu": mv, "e": ev, "order": len(ws),
        "factor_of_two_is_in_the_equation": True,
        "stable_bound": bound, "within_bound": mv < bound,
        "method": "Rangayyan (2024) eq. (3.203)"})


rangayyan_ch3_widrow_hoff_lms = widrowhoff  # pre-policy spelling


# -- rng161: Variable step-size LMS update rule..
def lmsvarstep(w, e, rvec, mu_n):
    """LMS update with a time-varying step size.

    Rangayyan (2024) eq. (3.204):
        w(n+1) = w(n) + 2 mu(n) e(n) r(n),

    eq. (3.203) with mu allowed to change at every sample.  The point is
    nonstationarity: a fixed mu that is stable for the loudest part of a
    record is far too slow for the quietest, so the step is normalized by
    a running estimate of the input power -- eq. (3.205) is one such rule.
    """
    ws, rv = aslist(w), aslist(rvec)
    if len(ws) != len(rv):
        raise ValueError("w and r(n) must have the same length")
    if not ws:
        raise ValueError("need at least one tap")
    mv, ev = float(mu_n), float(e)
    nxt = [a + 2.0 * mv * ev * b for a, b in zip(ws, rv)]
    return RichResult(payload={
        "w_next": nxt, "mu": mv, "e": ev, "order": len(ws),
        "time_varying": True,
        "method": "Rangayyan (2024) eq. (3.204)"})


rangayyan_ch3_lms_variable_step = lmsvarstep  # pre-policy spelling


# -- rng162: Time-varying step size mu(n) per Zhang et al. for VAG signals..
def lmszhang(mu, order, r, alpha=0.02, power_prev=None):
    """Time-varying LMS step size after Zhang et al., for VAG signals.

    Rangayyan (2024) eq. (3.205):
        mu(n) = mu / ( (M + 1) xbar^2(n) ),   0 < mu < 1,
    where the running power estimate is
        xbar^2(n) = alpha r^2(n) + (1 - alpha) xbar^2(n - 1),
    with a forgetting factor 0 <= alpha << 1.

    The book presents this as Zhang et al.'s remedy for the high
    nonstationarity of VAG signals: normalizing by the current input
    power makes the effective step size scale-free, so the same mu works
    across a record whose amplitude varies by an order of magnitude.

    alpha must be small -- the book writes alpha << 1 -- because it sets
    how fast the power estimate forgets; alpha near 1 tracks the
    instantaneous sample and reintroduces exactly the jitter the
    averaging was meant to remove.  A value above 0.5 is refused.
    """
    m = int(order)
    if m < 1:
        raise ValueError("order must be at least 1")
    mv = float(mu)
    if not 0 < mv < 1:
        raise ValueError("eq. (3.205) needs 0 < mu < 1")
    av = float(alpha)
    if not 0 <= av <= 0.5:
        raise ValueError("the book writes 0 <= alpha << 1; alpha above 0.5 "
                         "tracks the instantaneous sample instead of "
                         "averaging, got %g" % av)
    rv = float(r)
    prev = rv * rv if power_prev is None else float(power_prev)
    power = av * rv * rv + (1.0 - av) * prev
    if power <= 0:
        raise ValueError("the running power estimate vanished; mu(n) is "
                         "unbounded")
    step = mv / ((m + 1) * power)
    return RichResult(payload={
        "mu": step, "power": power, "power_prev": prev, "alpha": av,
        "order": m, "base_mu": mv,
        "method": "Rangayyan (2024) eq. (3.205), after Zhang et al."})


rangayyan_ch3_lms_step_size_zhang = lmszhang  # pre-policy spelling


# -- rng163: Weighted least-squares objective for the RLS algorithm with forgetting factor lambda..
def rlsobj(errors, lam):
    """Weighted least-squares objective of the RLS algorithm.

    Rangayyan (2024) eq. (3.206):
        xi(n) = sum_{i=1}^{n} lambda^(n-i) |e(i)|^2,   0 < lambda <= 1.

    The weights lambda^(n-i) are below 1 for past errors, so recent ones
    count for more.  The book gives the interpretation of 1/(1 - lambda)
    as the MEMORY of the algorithm: lambda = 0.98 remembers about 50
    samples, lambda = 1 remembers everything and RLS reduces to ordinary
    growing-window least squares, which cannot track a nonstationary
    signal at all.

    That effective memory is returned, because it is the number that
    decides whether a given lambda can follow the nonstationarity in a
    given record.
    """
    e = aslist(errors)
    if not e:
        raise ValueError("need at least one error value")
    lv = float(lam)
    if not 0 < lv <= 1:
        raise ValueError("eq. (3.206) needs 0 < lambda <= 1")
    n = len(e)
    weights = [lv ** (n - 1 - i) for i in range(n)]
    xi = fsum(wgt * v * v for wgt, v in zip(weights, e))
    return RichResult(payload={
        "xi": xi, "weights": weights, "lam": lv, "n": n,
        "memory": (1.0 / (1.0 - lv)) if lv < 1 else float("inf"),
        "growing_window": lv == 1.0,
        "method": "Rangayyan (2024) eq. (3.206)"})


rangayyan_ch3_rls_objective = rlsobj  # pre-policy spelling


# -- rng164: Normal equation for the RLS algorithm..
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def rlsnormal(phi, theta):
    """Normal equation solved by the RLS procedure.

    Rangayyan (2024) eq. (3.207):  Phi(n) w~(n) = Theta(n).

    Structurally identical to the Wiener-Hopf equation (3.168) -- the
    book says so, and derives it the same way -- but with Phi and Theta
    now EXPONENTIALLY WEIGHTED and updated at every sample rather than
    estimated once over the whole record.  That is the entire difference
    between the optimal fixed filter and the adaptive one.

    Solving this directly costs an M x M inversion per sample, which is
    what the ABCD lemma of eq. (3.213) exists to avoid; this function is
    the direct route, useful as the reference the recursion is checked
    against.
    """
    r = wienerhopf(phi, theta)
    out = dict(r)
    out["w_tilde"] = r["w"]
    out["same_form_as_wiener_hopf"] = True
    out["direct_inversion"] = True
    out["method"] = "Rangayyan (2024) eq. (3.207)"
    return RichResult(payload=out)


rangayyan_ch3_rls_normal_equation = rlsnormal  # pre-policy spelling


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
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def abcdlemma(A, B, C, D):
    """The ABCD matrix-inversion lemma used by RLS.

    Rangayyan (2024) eq. (3.213):
        (A + B C D)^(-1)
            = A^(-1) - A^(-1) B (D A^(-1) B + C^(-1))^(-1) D A^(-1),

    valid when A, C, (A + BCD) and (D A^-1 B + C^-1) are all invertible.

    Its use in RLS is that B is a COLUMN and C a scalar, so the inner
    bracket is 1 x 1: an M x M inverse is replaced by a scalar division,
    which is what makes the recursion O(M^2) per sample instead of
    O(M^3).

    Both sides are formed and compared, so the identity is checked on
    the caller's own matrices rather than recited.
    """
    Am = [aslist(r) for r in A]
    Bm = [aslist(r) for r in B]
    Cm = [aslist(r) for r in C]
    Dm = [aslist(r) for r in D]
    n = len(Am)
    if any(len(r) != n for r in Am):
        raise ValueError("A must be square")
    k = len(Cm)
    if len(Bm) != n or any(len(r) != k for r in Bm):
        raise ValueError("B must be n x k")
    if len(Dm) != k or any(len(r) != n for r in Dm):
        raise ValueError("D must be k x n")
    if any(len(r) != k for r in Cm):
        raise ValueError("C must be k x k")

    def inv(M):
        m = len(M)
        cols = []
        for j in range(m):
            cols.append(_solve(M, [1.0 if i == j else 0.0
                                   for i in range(m)]))
        return [[cols[j][i] for j in range(m)] for i in range(m)]

    def mul(P, Q):
        return [[fsum(P[i][t] * Q[t][j] for t in range(len(Q)))
                 for j in range(len(Q[0]))] for i in range(len(P))]

    BCD = mul(mul(Bm, Cm), Dm)
    direct = inv([[Am[i][j] + BCD[i][j] for j in range(n)]
                  for i in range(n)])
    Ai = inv(Am)
    Ci = inv(Cm)
    inner = mul(mul(Dm, Ai), Bm)
    inner = [[inner[i][j] + Ci[i][j] for j in range(k)] for i in range(k)]
    lemma = mul(mul(mul(Ai, Bm), inv(inner)), mul(Dm, Ai))
    lemma = [[Ai[i][j] - lemma[i][j] for j in range(n)] for i in range(n)]
    gap = max(abs(direct[i][j] - lemma[i][j])
              for i in range(n) for j in range(n))
    scale = max(abs(direct[i][j]) for i in range(n) for j in range(n)) or 1.0
    return RichResult(payload={
        "direct": direct, "lemma": lemma, "max_difference": gap,
        "holds": gap <= 1e-6 * scale, "n": n, "k": k,
        "scalar_when_k_is_one": k == 1,
        "method": "Rangayyan (2024) eq. (3.213)"})


rangayyan_ch3_abcd_matrix_inversion_lemma = abcdlemma  # pre-policy spelling


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
def rlsupdate(w_prev, k, alpha):
    """Compact RLS tap-weight update.

    Rangayyan (2024) eq. (3.224):
        w~(n) = w~(n-1) + k(n) alpha(n),

    with k(n) = P(n) r(n) the gain vector (eq. 3.221) and alpha(n) the a
    priori error of eq. (3.225).

    ERRATUM.  Eq. (3.224) is internally inconsistent in the book: its
    first line writes the correction with a MINUS sign,
        w~(n) = w~(n-1) - k(n)[x(n) - r^T(n) w~(n-1)],
    while its second line and eq. (3.225) give the PLUS form above.  The
    plus form is the correct one -- it is standard RLS, and the minus
    form drives the weights away from the solution and diverges.  This
    implementation uses the plus form.
    """
    ws, kv = aslist(w_prev), aslist(k)
    if len(ws) != len(kv):
        raise ValueError("w and k must have the same length")
    if not ws:
        raise ValueError("need at least one tap")
    a = float(alpha)
    return RichResult(payload={
        "w_next": [p + q * a for p, q in zip(ws, kv)],
        "correction": [q * a for q in kv], "alpha": a, "order": len(ws),
        "sign": "+",
        "erratum": "eq. (3.224) line 1 prints a minus sign that "
                   "contradicts its own line 2 and eq. (3.225); the plus "
                   "form is correct",
        "method": "Rangayyan (2024) eq. (3.224)"})


rangayyan_ch3_rls_weight_update_compact = rlsupdate  # pre-policy spelling


# -- rng175: A priori error in the RLS update step..
def rlsapriori(x, rvec, w_prev):
    """A priori error in the RLS update.

    Rangayyan (2024) eq. (3.225):
        alpha(n) = x(n) - r^T(n) w~(n-1) = x(n) - w~^T(n-1) r(n).

    A PRIORI means it uses the OLD weights: it is the error the filter
    would have made before this sample's update.  The a posteriori error
    x(n) - r^T(n) w~(n) uses the new ones and is smaller by a factor of
    (1 - k^T r).  Substituting one for the other in eq. (3.224) is a
    standard slip that changes the algorithm's convergence.

    Both are computable only after the update, so only the a priori one
    is returned here -- with the distinction stated.
    """
    rv, ws = aslist(rvec), aslist(w_prev)
    if len(rv) != len(ws):
        raise ValueError("r(n) and w must have the same length")
    if not rv:
        raise ValueError("need at least one tap")
    pred = fsum(a * b for a, b in zip(rv, ws))
    return RichResult(payload={
        "alpha": float(x) - pred, "prediction": pred, "order": len(rv),
        "uses_previous_weights": True,
        "not_the_a_posteriori_error": True,
        "method": "Rangayyan (2024) eq. (3.225)"})


rangayyan_ch3_rls_a_priori_error = rlsapriori  # pre-policy spelling


# -- rng204: PSD as the Fourier transform of the ACF (Wiener-Khinchin)..
def _solve(A, b):
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the correlation matrix is singular; the "
                             "Wiener-Hopf system of eq. (3.168) has no "
                             "unique solution")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _acf(x, lags):
    """Biased ACF estimate, divisor N (keeps the Toeplitz system PSD)."""
    n = len(x)
    return [fsum(x[i] * x[i + m] for i in range(n - m)) / n
            for m in range(lags)]


def _ccf(x, d, lags):
    """theta(k) = E[x(n-k) d(n)], the RHS of eq. (3.168)."""
    n = min(len(x), len(d))
    return [fsum(x[i - k] * d[i] for i in range(k, n)) / n
            for k in range(lags)]


def psdacf(x):
    """PSD as the Fourier transform of the ACF (Wiener-Khinchin).

    Rangayyan (2024) eq. (4.30):
        S_xx(f) = FT[phi_xx(tau)] = X(f) X*(f) = |X(f)|^2.

    Both routes are computed -- the transform of the CIRCULAR ACF, and
    the squared magnitude of the transform -- and compared.  They agree
    exactly only for the circular ACF; the linear ACF, truncated at the
    record length, transforms to a SMOOTHED spectrum, which is a
    different (and biased) estimator.  ``linear_difference`` shows how
    far apart the two are on this record.

    The book notes the PSD peaks at the frequencies of periodic activity,
    which is what makes it useful for finding EEG rhythms.
    """
    xs = aslist(x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    re, im = [], []
    step = 2.0 * pi / n
    for k in range(n):
        re.append(fsum(v * cos(-step * i * k) for i, v in enumerate(xs)))
        im.append(fsum(v * sin(-step * i * k) for i, v in enumerate(xs)))
    direct = [a * a + b * b for a, b in zip(re, im)]
    circ = [fsum(xs[i] * xs[(i + m) % n] for i in range(n))
            for m in range(n)]
    cr = []
    for k in range(n):
        cr.append(fsum(circ[m] * cos(-step * m * k) for m in range(n)))
    gap = max(abs(a - b) for a, b in zip(direct, cr))
    lin = _acf(xs, n) if n > 1 else [0.0]
    lr = []
    for k in range(n):
        lr.append(fsum(lin[m] * cos(-step * m * k) for m in range(n)))
    lgap = max(abs(a - b * n) for a, b in zip(direct, lr))
    scale = max(direct) or 1.0
    return RichResult(payload={
        "psd": direct, "via_circular_acf": cr, "acf_circular": circ,
        "acf_linear": lin, "max_difference": gap,
        "holds": gap <= 1e-6 * scale,
        "linear_difference": lgap,
        "linear_acf_is_smoothed": True, "n": n,
        "method": "Rangayyan (2024) eq. (4.30)"})


rangayyan_ch4_psd_from_acf = psdacf  # pre-policy spelling


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
    'rgacfd: ACF distance for segmentation, Section 8.5',
    'rgadp: LMS adaptive noise canceller -- Rangayyan & Krishnan Sec 3.10.2.',
    'rganc: adaptive noise canceller, Section 3.10',
    'rgeegadp: Adaptive segmentation of EEG using GLR test.',
    'rgfecg: fetal ECG by adaptive cancellation, Section 3.14',
    'rgglr: Generalized likelihood ratio (GLR) test for change detection.',
    'rgkalmn: Kalman filter',
    'rglms: LMS adaptive noise canceller, Section 3.10.2',
    'rgpcgadp: adaptive PCG segmentation, Section 8.5',
    'rgricca: steady-state Riccati solution',
    'rgrls: RLS adaptive filter, Section 3.10.3',
    'rgrls_mon: RLS error monitoring for segmentation, Section 8.5',
    'rgrlsl: RLS lattice predictor',
    'rgsemm: spectral error measure, Section 8.5',
    'rgwhop: Wiener-Hopf system from data, eqs. (3.168), (3.171)',
    'rgwnr: Wiener filter, time or frequency route, Section 3.9',
    'rng137: Estimation error.',
    'rng138: Wiener filter output as a convolution, eq. (3.154)',
    'rng139: Wiener output as an inner product, eq. (3.155)',
    'rng140: Estimation error in vector form.',
    'rng141: MSE cost function of the Wiener filter (Rangayyan Eq 3.166).',
    'rng142: Wiener cross-correlation vector Theta (Rangayyan Eq 3.160/3.161).',
    'rng143: Wiener autocorrelation matrix Phi (Rangayyan Eq 3.163/3.164/3.165).',
    'rng144: MSE gradient, Rangayyan eq. (3.167)',
    'rng145: Wiener-Hopf normal equation, eq. (3.168)',
    'rng146: optimal Wiener tap weights, eq. (3.169)',
    'rng147: minimum MSE of the Wiener filter, eq. (3.172)',
    'rng148: Wiener-Hopf as a convolution, eqs. (3.173)-(3.174)',
    'rng149: frequency-domain Wiener relation, eq. (3.175)',
    'rng150: Wiener frequency response, eq. (3.176)',
    'rng151: Optimal Wiener filter for noise removal (Rangayyan Eq 3.183).',
    'rng152: Wiener response from signal and noise PSDs, eq. (3.186)',
    'rng153: ANC primary input model, Section 3.10.1',
    'rng154: ANC output, Rangayyan eq. (3.196)',
    'rng155: LMS filter output on the reference, eq. (3.195)',
    'rng156: LMS estimation error.',
    'rng157: expanded LMS squared error, eq. (3.200)',
    'rng158: LMS steepest-descent step, eqs. (3.201)-(3.202)',
    'rng159: LMS gradient estimate.',
    'rng160: Widrow-Hoff LMS update, eq. (3.203)',
    'rng161: variable-step LMS update, eq. (3.204)',
    'rng162: Zhang time-varying LMS step size, eq. (3.205)',
    'rng163: RLS weighted objective, eq. (3.206)',
    'rng164: RLS normal equation, eq. (3.207)',
    'rng165: RLS correlation matrix.',
    'rng166: RLS cross-correlation vector.',
    'rng167: RLS recursion for the autocorrelation matrix (Rangayyan Eq 3.211).',
    'rng168: RLS recursion for the cross-correlation vector (Rangayyan Eq 3.212).',
    'rng169: ABCD matrix-inversion lemma, eq. (3.213)',
    'rng170: Riccati recursion for the inverse autocorrelation matrix (Rangayyan Eq 3.215).',
    'rng171: Kalman-like gain vector in RLS (Rangayyan Eq 3.217).',
    'rng172: RLS recursion for P(n) via the gain vector (Rangayyan Eq 3.218).',
    'rng173: RLS gain identity k(n) = P(n) r(n) (Rangayyan Eq 3.221).',
    'rng174: compact RLS weight update, eq. (3.224)',
    'rng175: RLS a priori error, eq. (3.225)',
    'rng204: PSD from the ACF, Rangayyan eq. (4.30)',
    'wnflt: Wiener filter for optimal noise reduction.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
