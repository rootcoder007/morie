# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Imbens-Kalyanaraman optimal bandwidth for sharp RDD."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causrddh", "ik_bandwidth", "causal_rdd_imbens_kalyanaraman"]

# Kernel constant C_K for the edge (triangular) kernel, printed in
# Section 6.2 of the source ("Using the edge kernel with C_K = 3.4375").
_CK_EDGE = 3.4375


def _ols(X, y):
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    b, _, _, _ = np.linalg.lstsq(Xa, ya, rcond=None)
    return b


def _median(v):
    s = sorted(float(u) for u in v)
    n = len(s)
    if n == 0:
        raise ValueError("empty side")
    m = n // 2
    return s[m] if n % 2 == 1 else 0.5 * (s[m - 1] + s[m])


def ik_bandwidth(x, y, cutoff=0.0):
    r"""Imbens-Kalyanaraman plug-in bandwidth for the sharp RD estimator.

    Implements, step by step, the "Algorithm for bandwidth selection"
    of Imbens and Kalyanaraman, NBER Working Paper 14726 (2009),
    Section 4.4 (the paper later published as Review of Economic
    Studies 79(3):933-959, 2012). Local source:
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    imbens-kalyanaraman-2009-w14726-optimal-bandwidth-rdd.pdf.

    Step 1 (density and variance at the cutoff): modified Silverman
    pilot :math:`h_1 = 1.84\, S_X N^{-1/5}`; counts, means and sample
    variances in :math:`[c-h_1, c)` and :math:`[c, c+h_1]`; then

    .. math:: \hat f(c) = \frac{N_{h_1,-}+N_{h_1,+}}{2 N h_1}, \qquad
              \hat\sigma^2(c) = \frac{(N_{h_1,-}-1)S^2_{Y,h_1,-}
              + (N_{h_1,+}-1)S^2_{Y,h_1,+}}{N_{h_1,-}+N_{h_1,+}}.

    ERRATUM (documented, followed the worked example): the displayed
    eq. (4.8) of the working paper prints
    :math:`\hat f(c)=(N_{h_1,-}+N_{h_1,+})/(N\,h_1)` WITHOUT the 2,
    but the authors' own worked example (Section 6.2, Lee data)
    computes :math:`\hat f(0) = (836+862)/(2\cdot 6558\cdot 0.1445)
    = 0.8962`, which is also the correct uniform-kernel density on
    :math:`[-1,1]`. The factor 2 is used here.

    Step 2 (curvature): medians of the running variable on each side;
    drop observations outside :math:`[\mathrm{med}_-, \mathrm{med}_+]`;
    fit the global cubic with a jump (eq. 4.10),
    :math:`Y=\gamma_0+\gamma_1 1_{X\ge c}+\gamma_2(X-c)+\gamma_3(X-c)^2
    +\gamma_4(X-c)^3`, and set :math:`\hat m^{(3)}(c) = 6\hat\gamma_4`.
    Pilot bandwidths
    :math:`h_{2,\pm} = 3.56\,(\hat\sigma^2(c) / (\hat f(c)
    \max(\hat m^{(3)}(c)^2, 0.01)))^{1/7} N_\pm^{-1/7}` with
    :math:`N_\pm` the full side counts; one-sided quadratic fits on
    :math:`[c, c+h_{2,+}]` and :math:`[c-h_{2,-}, c)` give
    :math:`\hat m^{(2)}_\pm(c)` as twice the quadratic coefficient.

    Step 3 (regularization and the bandwidth):
    :math:`\hat r_\pm = 720\,\hat\sigma^2(c)/(N_{2,\pm} h_{2,\pm}^4)`
    and

    SECOND ERRATUM (documented, followed the worked example): the
    printed :math:`\hat r_+` fraction in Section 6.2 shows the count
    1983 (the full window count) but its printed VALUE 0.2634 -- and
    the printed 0.3036 for :math:`\hat r_-`, and the final printed
    :math:`\hat h_{opt} = 0.2649` -- are reproduced only when
    :math:`N_{2,\pm}` counts the quadratic-window observations WITHIN
    the median-trimmed sample of Step 2 (1909 = 3818/2 on the right,
    1370 on the left, on the Lee data), while the quadratic fits
    themselves use the full-sample windows (the printed
    :math:`\hat m^{(2)}_\pm` match the full-sample fits, not the
    trimmed ones). This implementation follows that
    worked-example-consistent arithmetic and reports both counts.
    Then

    .. math:: \hat h_{opt} = C_K \left( \frac{2\hat\sigma^2(c)}
              {\hat f(c)\,[(\hat m^{(2)}_+(c)-\hat m^{(2)}_-(c))^2
              + \hat r_+ + \hat r_-]} \right)^{1/5} N^{-1/5},

    with the edge-kernel constant :math:`C_K = 3.4375`. NOTE: the
    published 2012 Review of Economic Studies version modifies some
    constants (e.g. a regularization constant of 2160 is quoted in
    later software); this implementation follows the NBER 2009 text
    in hand, whose Section 6.2 worked example on the Lee (2008) data
    (N = 6558, h1 = 0.1445, f = 0.8962, sigma2 = 0.1128^2,
    m3 = -5.4611, h2 = 0.3674/0.3852, m2 = -0.5233/0.4904,
    r = 0.2634/0.3036, h_opt = 0.2649) is reproduced by this code on
    the shipped lee2008_house dataset (this code: 0.26333/0.30357 and
    0.264863) and pinned in the tests.

    Parameters
    ----------
    x : array-like
        Running (forcing) variable.
    y : array-like
        Outcome.
    cutoff : float
        Threshold c.

    Returns
    -------
    RichResult
        ``estimate`` (the bandwidth), ``h1``, ``f_hat``, ``sigma2``,
        ``n_left_h1``, ``n_right_h1``, ``m3``, ``h2_left``,
        ``h2_right``, ``m2_left``, ``m2_right``, ``n2_left``,
        ``n2_right``, ``r_left``, ``r_right``, ``h_unregularized``,
        ``kernel_constant``, ``n``.

    References
    ----------
    Imbens, G. and Kalyanaraman, K. (2009), "Optimal Bandwidth Choice
    for the Regression Discontinuity Estimator", NBER Working Paper
    14726; published as Review of Economic Studies 79(3):933-959
    (2012), doi:10.1093/restud/rdr043. Algorithm: Section 4.4, eqs.
    (4.8)-(4.13); worked example: Section 6.2.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = int(len(xa))
    if n < 10:
        raise ValueError("need at least 10 observations")
    c = float(cutoff)
    d = xa - c

    # ---- Step 1 ----
    sx = float(np.std(xa, ddof=1))
    h1 = 1.84 * sx * n ** (-0.2)
    il = (d >= -h1) & (d < 0.0)
    ir = (d >= 0.0) & (d <= h1)
    nl = int(np.sum(il.astype(float)))
    nr = int(np.sum(ir.astype(float)))
    if nl < 3 or nr < 3:
        raise ValueError("fewer than 3 observations within the pilot "
                         "window on one side of the cutoff")
    yl = ya[il]
    yr = ya[ir]
    s2l = float(np.var(yl, ddof=1))
    s2r = float(np.var(yr, ddof=1))
    # eq. (4.8) with the worked-example factor 2 (see ERRATUM above)
    f_hat = (nl + nr) / (2.0 * n * h1)
    sigma2 = ((nl - 1) * s2l + (nr - 1) * s2r) / float(nl + nr)

    # ---- Step 2 ----
    left = d < 0.0
    right = d >= 0.0
    n_neg = int(np.sum(left.astype(float)))
    n_pos = int(np.sum(right.astype(float)))
    med_l = _median(d[left])
    med_r = _median(d[right])
    keep = (d >= med_l) & (d <= med_r)
    dk = d[keep]
    yk = ya[keep]
    Xc = np.column_stack(
        [np.ones(len(dk)), (dk >= 0.0).astype(float), dk, dk ** 2,
         dk ** 3])
    g = _ols(Xc, yk)
    m3 = 6.0 * float(g[4])
    base = (sigma2 / (f_hat * max(m3 * m3, 0.01))) ** (1.0 / 7.0)
    h2r = 3.56 * base * n_pos ** (-1.0 / 7.0)
    h2l = 3.56 * base * n_neg ** (-1.0 / 7.0)

    def _quad(mask, mask_trim):
        dm = d[mask]
        ym = ya[mask]
        n2 = int(len(dm))
        if n2 < 4:
            raise ValueError("fewer than 4 observations in a pilot "
                             "quadratic window")
        Xq = np.column_stack([np.ones(n2), dm, dm ** 2])
        b = _ols(Xq, ym)
        # the fit uses the full-sample window; the regularization
        # count is taken within the median-trimmed sample (see the
        # SECOND ERRATUM note in the docstring)
        n2_trim = int(np.sum((mask & mask_trim).astype(float)))
        return 2.0 * float(b[2]), n2, max(n2_trim, 1)

    m2r, n2r_full, n2r = _quad(right & (d <= h2r), d <= med_r)
    m2l, n2l_full, n2l = _quad(left & (d >= -h2l), d >= med_l)

    # ---- Step 3 ----
    rr = 720.0 * sigma2 / (n2r * h2r ** 4)
    rl = 720.0 * sigma2 / (n2l * h2l ** 4)
    curv = (m2r - m2l) ** 2
    h_opt = _CK_EDGE * (2.0 * sigma2 / (f_hat * (curv + rr + rl))) ** 0.2 \
        * n ** (-0.2)
    h_unreg = _CK_EDGE * (2.0 * sigma2 / (f_hat * curv)) ** 0.2 \
        * n ** (-0.2) if curv > 0 else np.inf

    return RichResult(payload={
        "estimate": float(h_opt),
        "h1": float(h1),
        "f_hat": float(f_hat),
        "sigma2": float(sigma2),
        "n_left_h1": nl,
        "n_right_h1": nr,
        "mean_left_h1": float(np.mean(yl)),
        "mean_right_h1": float(np.mean(yr)),
        "m3": float(m3),
        "h2_left": float(h2l),
        "h2_right": float(h2r),
        "m2_left": float(m2l),
        "m2_right": float(m2r),
        "n2_left": n2l,
        "n2_right": n2r,
        "n2_left_full": n2l_full,
        "n2_right_full": n2r_full,
        "r_left": float(rl),
        "r_right": float(rr),
        "h_unregularized": float(h_unreg),
        "kernel_constant": _CK_EDGE,
        "n": n,
        "method": ("Imbens-Kalyanaraman (2009/2012) plug-in bandwidth, "
                   "edge kernel, NBER w14726 algorithm"),
    })


# primary name = module name; descriptive name kept as the canonical one.
causrddh = ik_bandwidth
# stub-era exported name
causal_rdd_imbens_kalyanaraman = ik_bandwidth


def cheatsheet():
    return ("causrddh: Imbens-Kalyanaraman plug-in RDD bandwidth "
            "(NBER w14726 Section 4.4; eq. 4.8 erratum corrected per "
            "the Section 6.2 worked example)")
