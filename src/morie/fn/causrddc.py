r"""Robust bias-corrected inference for regression-discontinuity designs.

Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014) "Robust Nonparametric
Confidence Intervals for Regression-Discontinuity Designs", *Econometrica*
82(6), 2295-2326.

Local polynomial RD estimators need a bandwidth, and the bandwidth selectors
in use -- cross-validation, or minimising asymptotic MSE -- deliberately
balance squared bias against variance. That makes them "large" in the sense
that :math:`n h_n^5 \not\to 0` for the local-linear estimator, so the leading
bias does **not** vanish from the distributional approximation and the
conventional interval

.. math:: I_{\mathrm{SRD}}(h_n) = \hat\tau_{\mathrm{SRD}}(h_n)
          \pm \Phi^{-1}_{1-\alpha/2}\sqrt{V_{\mathrm{SRD}}(h_n)}

undercovers. The paper's fix is two-part, and the second part is the
contribution:

1. **bias-correct**, :math:`\hat\tau^{bc} = \hat\tau_{\nu p}(h_n) -
   h_n^{p+1-\nu}\hat B(h_n, b_n)`, with the bias estimated from a
   higher-order (order :math:`q > p`) local polynomial at a pilot bandwidth
   :math:`b_n`;
2. **rescale by a variance that includes the bias estimate's own
   variability**, :math:`V^{bc} = V(h_n) + C^{bc}(h_n, b_n)`, which is what
   allows :math:`\rho_n = h_n / b_n \to \rho \in [0, \infty]` instead of the
   conventional :math:`\rho_n \to 0`. Conventional bias correction assumes
   the correction's variability vanishes; "however, :math:`h_n/b_n` is never
   zero in finite samples".

All three intervals are returned, because the comparison is the point:
``conventional``, ``bias_corrected`` (recentred, conventional variance) and
``robust`` (recentred, robust variance).

Implementation follows the estimators exactly rather than their asymptotic
approximations. Every quantity here is a linear combination of the outcomes,
:math:`\hat\tau = \sum_i w_i Y_i`, so the conditional variance is
:math:`\sum_i w_i^2 \hat\sigma_i^2` with no further approximation, and the
correction term :math:`C^{bc}` appears automatically because the
bias-corrected weights are the ones used. Two variance routes are offered, as
in section 5: nearest-neighbour (Abadie & Imbens 2006), which the paper
prefers and uses with :math:`J = 3`, and plug-in residuals from the fitted
local polynomials.

**Remark 7 is the identity this implementation is pinned to.** "If
:math:`h_n = b_n` (and the same kernel function is used), then
:math:`\hat\tau^{bc}_{\mathrm{SRD}}(h_n, h_n)` is numerically equivalent to
the (not bias-corrected) local-quadratic estimator", and the robust variance
coincides with that estimator's variance. That holds here by construction --
subtracting :math:`\hat\omega \hat\mu^{(p+1)}/(p+1)!` from the order-:math:`p`
fit is the Frisch-Waugh-Lovell decomposition of the order-:math:`(p+1)` fit --
and the anchor checks it to machine precision, for :math:`p = 1` and
:math:`p = 2`.

**Bandwidths.** Lemma 1 gives the MSE-optimal choice

.. math:: h_{\mathrm{MSE},\nu p s} = C\, n^{-1/(2p+3)}, \qquad
          C = \left[\frac{(1 + 2\nu) V_{\nu p}}
          {2(p + 1 - \nu) B^2_{\nu, p, p+1, s}}\right]^{1/(2p+3)},

with :math:`V_{\nu p} = (\sigma_-^2 + \sigma_+^2)\nu!^2
e_\nu' \Gamma_p^{-1}\Psi_p\Gamma_p^{-1}e_\nu / f` and
:math:`B_{\nu p r s} = \frac{\mu^{(r)}_+ - (-1)^{\nu+r+s}\mu^{(r)}_-}{r!}
\nu! e_\nu'\Gamma_p^{-1}\vartheta_{pr}`, where
:math:`\Gamma_p = \int_0^\infty K r_p r_p'`,
:math:`\vartheta_{pq} = \int_0^\infty K u^q r_p` and
:math:`\Psi_p = \int_0^\infty K^2 r_p r_p'`. Those kernel constants are
computed here by quadrature and are exact for the uniform kernel
(:math:`\Gamma_p[i][j] = 1/(i+j+1)`), which the anchor checks. The unknown
:math:`\mu^{(p+1)}_\pm`, :math:`\sigma^2_\pm` and :math:`f` are replaced by
preliminary global-polynomial and nearest-neighbour estimates; the paper's own
data-driven selectors live in its supplement (Section S.2.6), which is not in
this library, so those preliminary choices are documented as this module's
rather than attributed.

**Fuzzy designs.** Section 3 handles them through
:math:`\hat\varsigma_{\nu p} = \hat\tau_{Y\nu p}/\hat\tau_{T\nu p}` and its
first-order linearisation
:math:`(\hat\tau_Y - \tau_Y)/\tau_T - \tau_Y(\hat\tau_T - \tau_T)/\tau_T^2`,
which is exactly the weight vector used here when ``treatment`` is supplied,
so fuzzy RD and fuzzy kink RD come out of the same code path with
:math:`\nu = 0` and :math:`\nu = 1`.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causrddc", "causal_rdd_ccft_bw", "rdrobust", "rd_bandwidth", "kernel_constants",
           "local_poly_weights"]

_KERNELS = ("triangular", "uniform", "epanechnikov")


def _kern(u, kernel):
    a = abs(u)
    if a > 1.0:
        return 0.0
    if kernel == "uniform":
        return 1.0
    if kernel == "triangular":
        return 1.0 - a
    return 0.75 * (1.0 - a * a)


def _solve(M, b):
    return [float(v) for v in np.linalg.solve(np.asarray(M, dtype=float),
                                              np.asarray(b, dtype=float))]


def local_poly_weights(x, h, p, nu, kernel="triangular", side=1):
    r"""Weights of the local polynomial estimator of :math:`\mu^{(\nu)}`.

    Returns ``(w, omega)``. ``w`` satisfies
    :math:`\hat\mu^{(\nu)}_{p}(h) = \sum_i w_i Y_i` over the units on the
    given side of the cutoff, and ``omega`` is the weight the omitted
    :math:`x^{p+1}` term carries into that estimate,
    :math:`\nu!\, e_\nu' (R'WR)^{-1} R'W X^{p+1}` -- the finite-sample
    leading bias per unit of :math:`\mu^{(p+1)}/(p+1)!`, which is what the
    bias correction removes.
    """
    n = len(x)
    keep = [i for i in range(n)
            if (x[i] >= 0.0 if side > 0 else x[i] < 0.0) and abs(x[i]) <= h]
    d = p + 1
    M = [[0.0] * d for _ in range(d)]
    RW = [[0.0] * n for _ in range(d)]
    xp1 = [0.0] * n
    for i in keep:
        k = _kern(x[i] / h, kernel)
        if k <= 0.0:
            continue
        r = [(x[i] / h) ** t for t in range(d)]
        for a in range(d):
            RW[a][i] = k * r[a]
            for b in range(d):
                M[a][b] += k * r[a] * r[b]
        xp1[i] = (x[i] / h) ** d
    e = [1.0 if t == nu else 0.0 for t in range(d)]
    try:
        c = _solve([[M[a][b] for b in range(d)] for a in range(d)], e)
    except Exception:
        raise ValueError("causrddc: the local polynomial design is singular "
                         "at h = %g on side %+d -- too few points inside "
                         "the bandwidth" % (h, side))
    scale = math.factorial(nu) / (h ** nu)
    w = [scale * sum(c[a] * RW[a][i] for a in range(d)) for i in range(n)]
    omega = scale * sum(c[a] * sum(RW[a][i] * xp1[i] for i in range(n))
                        for a in range(d)) * (h ** (p + 1))
    return w, omega


def kernel_constants(p, q, kernel="triangular", n_grid=2001):
    r"""The constants of Lemma 1: :math:`\Gamma_p`, :math:`\vartheta_{pq}`,
    :math:`\Psi_p`.

    :math:`\Gamma_p = \int_0^\infty K(u) r_p(u) r_p(u)' du`,
    :math:`\vartheta_{pq} = \int_0^\infty K(u) u^q r_p(u) du`,
    :math:`\Psi_p = \int_0^\infty K(u)^2 r_p(u) r_p(u)' du`, by Simpson
    quadrature on the kernel's support. For the uniform kernel these are
    exactly :math:`1/(i+j+1)`, :math:`1/(q+i+1)` and :math:`1/(i+j+1)`.
    """
    d = p + 1
    G = [[0.0] * d for _ in range(d)]
    P = [[0.0] * d for _ in range(d)]
    th = [0.0] * d
    m = int(n_grid) | 1
    step = 1.0 / (m - 1)
    for g in range(m):
        u = g * step
        wq = (1.0 if g in (0, m - 1) else (4.0 if g % 2 else 2.0)) * step / 3.0
        k = _kern(u, kernel)
        for a in range(d):
            th[a] += wq * k * (u ** q) * (u ** a)
            for b in range(d):
                G[a][b] += wq * k * (u ** a) * (u ** b)
                P[a][b] += wq * k * k * (u ** a) * (u ** b)
    return G, th, P


def _global_derivative(x, y, side, order, deriv):
    """Preliminary global polynomial estimate of mu^(deriv) at the cutoff."""
    idx = [i for i in range(len(x))
           if (x[i] >= 0.0 if side > 0 else x[i] < 0.0)]
    d = order + 1
    if len(idx) <= d:
        raise ValueError("causrddc: too few observations on side %+d for a "
                         "preliminary polynomial of order %d" % (side, order))
    M = [[0.0] * d for _ in range(d)]
    v = [0.0] * d
    for i in idx:
        r = [x[i] ** t for t in range(d)]
        for a in range(d):
            v[a] += r[a] * y[i]
            for b in range(d):
                M[a][b] += r[a] * r[b]
    beta = _solve(M, v)
    fitted = [sum(beta[t] * x[i] ** t for t in range(d)) for i in idx]
    resid = [y[i] - f for i, f in zip(idx, fitted)]
    sigma2 = (sum(r * r for r in resid) / max(1, len(idx) - d))
    return beta[deriv] * math.factorial(deriv), sigma2


def _nn_sigma2(x, y, J, side_of):
    """Abadie-Imbens nearest-neighbour variance, section 5, same side only."""
    n = len(x)
    out = [0.0] * n
    idx_pos = [i for i in range(n) if side_of[i] > 0]
    idx_neg = [i for i in range(n) if side_of[i] <= 0]
    for group in (idx_pos, idx_neg):
        if len(group) < J + 1:
            continue
        order = sorted(group, key=lambda i: x[i])
        pos = dict((i, t) for t, i in enumerate(order))
        for i in group:
            t = pos[i]
            cand = []
            lo, hi = t - 1, t + 1
            while len(cand) < J and (lo >= 0 or hi < len(order)):
                if lo < 0:
                    cand.append(order[hi])
                    hi += 1
                elif hi >= len(order):
                    cand.append(order[lo])
                    lo -= 1
                elif abs(x[order[lo]] - x[i]) <= abs(x[order[hi]] - x[i]):
                    cand.append(order[lo])
                    lo -= 1
                else:
                    cand.append(order[hi])
                    hi += 1
            mean = sum(y[j] for j in cand) / float(len(cand))
            out[i] = (len(cand) / (len(cand) + 1.0)) * (y[i] - mean) ** 2
    return out


def rd_bandwidth(x, y, nu=0, p=1, kernel="triangular", s=0,
                 prelim_order=None):
    r"""The MSE-optimal bandwidth of Lemma 1.

    ``s`` selects the estimand's sign convention in
    :math:`B_{\nu p r s}`: ``s = 0`` for a difference of one-sided
    derivatives (the RD estimand itself), ``s = 2`` for the pilot bandwidth
    of the bias estimate, as in the paper's own examples
    :math:`h_{\mathrm{MSE},010}` and :math:`b_{\mathrm{MSE},222}`.

    Returns ``{"h": ..., "h_unclamped": ..., "at_bound": ..., "C": ...,
    "B": ..., "V": ...}``. ``h`` is clamped to the observed support of the
    running variable and ``at_bound`` says whether the clamp bound: an
    estimated bias constant near zero -- the two sides sharing their
    :math:`(p+1)`th derivative -- makes the unclamped formula diverge.
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    n = len(x)
    if n != len(y):
        raise ValueError("causrddc: x and y must have the same length")
    if kernel not in _KERNELS:
        raise ValueError("causrddc: kernel must be one of %r" % (_KERNELS,))
    if not 0 <= nu <= p:
        raise ValueError("causrddc: need 0 <= nu <= p")
    r = p + 1
    po = int(prelim_order) if prelim_order is not None else r + 1
    mu_p, s2p = _global_derivative(x, y, +1, po, r)
    mu_m, s2m = _global_derivative(x, y, -1, po, r)
    G, th, P = kernel_constants(p, r, kernel)
    e = [1.0 if t == nu else 0.0 for t in range(p + 1)]
    Ginv_e = _solve(G, e)
    # B_{nu,p,p+1,s}
    diff = mu_p - ((-1.0) ** (nu + r + s)) * mu_m
    B = (diff / math.factorial(r)) * math.factorial(nu) * \
        sum(Ginv_e[a] * th[a] for a in range(p + 1))
    # V_{nu p}: (sigma^2_- + sigma^2_+) nu!^2 e' G^-1 Psi G^-1 e / f
    PG = [sum(P[a][b] * Ginv_e[b] for b in range(p + 1))
          for a in range(p + 1)]
    quad = sum(Ginv_e[a] * PG[a] for a in range(p + 1))
    f = _density_at_zero(x)
    V = (s2p + s2m) * (math.factorial(nu) ** 2) * quad / f
    if abs(B) < 1e-300:
        raise ValueError("causrddc: the leading bias constant is zero, so "
                         "the MSE-optimal bandwidth is not defined; supply h")
    C = ((1.0 + 2.0 * nu) * V /
         (2.0 * (p + 1.0 - nu) * B * B)) ** (1.0 / (2.0 * p + 3.0))
    h = C * n ** (-1.0 / (2.0 * p + 3.0))
    # A near-zero estimated bias constant sends h to infinity. That happens
    # whenever the two sides share their (p+1)th derivative, which is common
    # in simulated designs and not rare in real ones. The paper regularises
    # (Remark 11) by a rule given only in its supplement, so the honest
    # substitute here is to clamp at the observed support and say so.
    span = max(max(v for v in x), -min(v for v in x))
    at_bound = h > span
    return {"h": min(h, span), "h_unclamped": h, "at_bound": at_bound,
            "C": C, "B": B, "V": V, "f": f, "mu_plus": mu_p,
            "mu_minus": mu_m}


def _density_at_zero(x, h=None):
    """A simple kernel density estimate of f(0) for the V_{nu p} constant."""
    n = len(x)
    xs = sorted(x)
    mean = sum(xs) / n
    sd = math.sqrt(max(1e-300, sum((v - mean) ** 2 for v in xs) / (n - 1)))
    if h is None:
        h = 1.06 * sd * n ** (-0.2)
    if h <= 0:
        raise ValueError("causrddc: the running variable has no spread")
    tot = 0.0
    for v in x:
        u = v / h
        if abs(u) <= 1.0:
            tot += 0.75 * (1.0 - u * u)
    return max(tot / (n * h), 1e-12)


def causrddc(y, x, treatment=None, cutoff=0.0, nu=0, p=1, q=None, h=None,
             b=None, kernel="triangular", alpha=0.05, vce="nn", J=3):
    r"""Sharp or fuzzy RD estimates with conventional, bias-corrected and
    robust confidence intervals.

    Parameters
    ----------
    y : array-like
        Outcome.
    x : array-like
        Running variable. Treatment is assigned at ``x >= cutoff``.
    treatment : array-like, optional
        Actual treatment received. Given, the design is fuzzy and the
        estimand is the ratio :math:`\tau_Y/\tau_T` (section 3.2); omitted,
        the design is sharp.
    cutoff : float
        The threshold :math:`\bar x`; the data are recentred at it.
    nu : int
        Derivative of interest: 0 for the level (sharp/fuzzy RD), 1 for the
        kink designs (section 3.1, 3.3).
    p : int
        Order of the local polynomial point estimator. The paper's local
        linear case is ``nu=0, p=1``.
    q : int, optional
        Order of the bias estimator, ``q > p``. Defaults to ``p + 1``, which
        is the case Remark 7 describes.
    h, b : float, optional
        Main and pilot bandwidths. Omitted, ``h`` is the MSE-optimal
        :math:`h_{\mathrm{MSE},\nu p 0}` of Lemma 1 and ``b`` the
        corresponding pilot :math:`h_{\mathrm{MSE},\,p+1,\,q,\,2}`.
    kernel : {"triangular", "uniform", "epanechnikov"}
        Kernel :math:`k(\cdot)`; all satisfy Assumption 2.
    alpha : float
        1 - coverage.
    vce : {"nn", "hc"}
        Variance route: nearest-neighbour (Abadie & Imbens 2006), which the
        paper proposes in section 5 and uses with ``J=3``, or plug-in
        residuals from the local polynomial fits.
    J : int
        Number of neighbours for ``vce="nn"``.

    Returns
    -------
    RichResult
        ``estimate`` is the conventional point estimate, ``bias_corrected``
        the recentred one; ``ci_conventional``, ``ci_bias_corrected`` and
        ``ci_robust`` are the three intervals; ``se_conventional`` and
        ``se_robust`` the two standard errors; ``h``, ``b``, ``rho``,
        ``n_left``, ``n_right`` describe the fit.

    Examples
    --------
    A jump of 1 at the cutoff::

        r = causrddc(y, x)
        r["estimate"], r["ci_robust"]

    With ``h == b`` the bias-corrected estimate is the local-quadratic one
    (Remark 7)::

        a = causrddc(y, x, p=1, h=0.5, b=0.5)["bias_corrected"]
        c = causrddc(y, x, p=2, h=0.5)["estimate"]        # a == c

    References
    ----------
    Calonico, S., Cattaneo, M. D. & Titiunik, R. (2014) "Robust
    Nonparametric Confidence Intervals for Regression-Discontinuity
    Designs", *Econometrica* 82(6), 2295-2326, doi:10.3982/ECTA11757:
    Theorem 1, Remarks 3 and 7, Lemma 1, section 5.

    Abadie, A. & Imbens, G. W. (2006) "Large Sample Properties of
    Matching Estimators for Average Treatment Effects", *Econometrica*
    74(1), 235-267, doi:10.1111/j.1468-0262.2006.00655.x -- the
    nearest-neighbour variance estimator of ``vce="nn"``.

    The partialling-out step that makes Remark 7 hold by construction:

    Frisch, R. & Waugh, F. V. (1933) "Partial Time Regressions as
    Compared with Individual Trends", *Econometrica* 1(4), 387-401,
    JSTOR 1907330.

    Lovell, M. C. (1963) "Seasonal Adjustment of Economic Time Series
    and Multiple Regression Analysis", *Journal of the American
    Statistical Association* 58(304), 993-1010,
    doi:10.1080/01621459.1963.10480682 -- the generalisation to
    arbitrary regressors; freely available as Cowles Foundation
    Discussion Paper No. 151 (1963).
    """
    y = [float(v) for v in y]
    x = [float(v) - float(cutoff) for v in x]
    n = len(x)
    if n != len(y):
        raise ValueError("causrddc: y and x must have the same length")
    if kernel not in _KERNELS:
        raise ValueError("causrddc: kernel must be one of %r" % (_KERNELS,))
    if vce not in ("nn", "hc"):
        raise ValueError("causrddc: vce must be 'nn' or 'hc'")
    p = int(p)
    nu = int(nu)
    if not 0 <= nu <= p:
        raise ValueError("causrddc: need 0 <= nu <= p")
    q = p + 1 if q is None else int(q)
    if q <= p:
        raise ValueError("causrddc: need q > p (the bias estimator must be "
                         "of higher order than the point estimator)")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("causrddc: alpha must lie in (0, 1)")

    if h is None:
        h = rd_bandwidth(x, y, nu, p, kernel, s=0)["h"]
    h = float(h)
    if h <= 0:
        raise ValueError("causrddc: h must be positive")
    if b is None:
        b = rd_bandwidth(x, y, p + 1, q, kernel, s=2)["h"]
    b = float(b)
    if b <= 0:
        raise ValueError("causrddc: b must be positive")

    def weights_for(vec):
        wp, om_p = local_poly_weights(x, h, p, nu, kernel, +1)
        wm, om_m = local_poly_weights(x, h, p, nu, kernel, -1)
        vp, _ = local_poly_weights(x, b, q, p + 1, kernel, +1)
        vm, _ = local_poly_weights(x, b, q, p + 1, kernel, -1)
        fac = 1.0 / math.factorial(p + 1)
        w_conv = [wp[i] - wm[i] for i in range(n)]
        w_bc = [w_conv[i] - fac * (om_p * vp[i] - om_m * vm[i])
                for i in range(n)]
        tau = sum(w_conv[i] * vec[i] for i in range(n))
        tau_bc = sum(w_bc[i] * vec[i] for i in range(n))
        return w_conv, w_bc, tau, tau_bc

    wY, wYbc, tauY, tauYbc = weights_for(y)
    if treatment is None:
        w_conv, w_bc = wY, wYbc
        tau, tau_bc = tauY, tauYbc
        resid_source = y
    else:
        t = [float(v) for v in treatment]
        if len(t) != n:
            raise ValueError("causrddc: treatment must have the same length "
                             "as y")
        wT, wTbc, tauT, tauTbc = weights_for(t)
        if abs(tauT) < 1e-12:
            raise ValueError("causrddc: the first-stage jump is zero, so the "
                             "fuzzy estimand is not identified")
        tau = tauY / tauT
        tau_bc = tauYbc / tauTbc
        # the paper's first-order linearisation (section 4.2, Lemma 2)
        w_conv = [(wY[i] - tau * wT[i]) / tauT for i in range(n)]
        w_bc = [(wYbc[i] - tau_bc * wTbc[i]) / tauTbc for i in range(n)]
        resid_source = [y[i] - tau * t[i] for i in range(n)]

    side_of = [1 if v >= 0.0 else -1 for v in x]
    if vce == "nn":
        sig2 = _nn_sigma2(x, resid_source, int(J), side_of)
    else:
        sig2 = _hc_sigma2(x, resid_source, h, p, kernel)

    v_conv = sum(w_conv[i] ** 2 * sig2[i] for i in range(n))
    v_rbc = sum(w_bc[i] ** 2 * sig2[i] for i in range(n))
    z = _norm_ppf(1.0 - float(alpha) / 2.0)
    se_c = math.sqrt(max(v_conv, 0.0))
    se_r = math.sqrt(max(v_rbc, 0.0))
    inside = [i for i in range(n) if abs(x[i]) <= h]
    return RichResult(payload={
        "estimate": tau,
        "bias_corrected": tau_bc,
        "se_conventional": se_c,
        "se_robust": se_r,
        "ci_conventional": (tau - z * se_c, tau + z * se_c),
        "ci_bias_corrected": (tau_bc - z * se_c, tau_bc + z * se_c),
        "ci_robust": (tau_bc - z * se_r, tau_bc + z * se_r),
        "pvalue_robust": 2.0 * (1.0 - _norm_cdf(abs(tau_bc) / se_r))
        if se_r > 0 else float("nan"),
        "h": h, "b": b, "rho": h / b, "p": p, "q": q, "nu": nu,
        "kernel": kernel, "vce": vce, "alpha": float(alpha),
        "n": n,
        "n_left": sum(1 for i in inside if x[i] < 0.0),
        "n_right": sum(1 for i in inside if x[i] >= 0.0),
        "weights_conventional": w_conv,
        "weights_bias_corrected": w_bc,
        "fuzzy": treatment is not None,
        "method": "robust bias-corrected RD (Calonico, Cattaneo & Titiunik "
                  "2014)",
    })


def _hc_sigma2(x, y, h, p, kernel):
    """Plug-in residual variance from the local polynomial fits."""
    n = len(x)
    out = [0.0] * n
    for side in (+1, -1):
        idx = [i for i in range(n)
               if (x[i] >= 0.0 if side > 0 else x[i] < 0.0) and
               abs(x[i]) <= h]
        d = p + 1
        if len(idx) <= d:
            continue
        M = [[0.0] * d for _ in range(d)]
        v = [0.0] * d
        for i in idx:
            k = _kern(x[i] / h, kernel)
            r = [(x[i] / h) ** t for t in range(d)]
            for a in range(d):
                v[a] += k * r[a] * y[i]
                for bb in range(d):
                    M[a][bb] += k * r[a] * r[bb]
        beta = _solve(M, v)
        for i in idx:
            fit = sum(beta[t] * (x[i] / h) ** t for t in range(d))
            out[i] = (y[i] - fit) ** 2
    return out


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(pr):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < pr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return ("causrddc: robust bias-corrected RD inference (Calonico, "
            "Cattaneo & Titiunik 2014). MSE-optimal bandwidths are 'large' "
            "on purpose, so the conventional CI carries a first-order bias "
            "and undercovers. Fix: recentre by an estimated bias from a "
            "higher-order local polynomial at pilot bandwidth b, AND "
            "rescale by V + C^bc, a variance that includes the bias "
            "estimate's own variability -- which is what lets rho = h/b "
            "stay non-zero. Remark 7: at h = b the bias-corrected "
            "estimator IS the local-quadratic estimator (Frisch-Waugh). "
            "Bandwidths from Lemma 1; variance nearest-neighbour (J=3) or "
            "plug-in residuals. Sharp, kink (nu=1) and fuzzy all from one "
            "code path.")


# compact alias per ledger/NAMING.md
rdrobust = causrddc

# name carried over from the generated stub this replaced
causal_rdd_ccft_bw = causrddc
