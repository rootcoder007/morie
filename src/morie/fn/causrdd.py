# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Sharp RDD by local linear regression at the threshold."""

from . import _array_core as np

from ._richresult import RichResult
from .causrddh import ik_bandwidth

__all__ = ["causrdd", "rdd_local_linear", "causal_rdd_local_lin"]


def _kernel(name, u):
    if name == "triangular":
        return np.maximum(1.0 - np.abs(u), 0.0)
    if name == "uniform":
        return np.where(np.abs(u) <= 1.0, 0.5, 0.0)
    raise ValueError("kernel must be 'triangular' or 'uniform'")


def _llr_side(dm, ym, w):
    """Weighted local linear fit; returns intercept, slope and the HC0
    sandwich variance of the intercept.

    Solves the kernel-weighted least squares problem
    min sum_i w_i (y_i - a - b d_i)^2; the variance of (a, b) is the
    sandwich (X'WX)^{-1} X'W diag(e^2) WX (X'WX)^{-1} with e the
    weighted-fit residuals (White-form plug-in, the vce = "hc0"
    convention of the rdrobust software, which the tests anchor
    against).
    """
    n = len(dm)
    X = np.column_stack([np.ones(n), dm])
    XtW = X.T * w
    A = XtW @ X
    b = np.linalg.solve(A, XtW @ ym)
    e = ym - X @ b
    meat = (XtW * (e ** 2)) @ (X * w[:, None])
    Ainv = np.linalg.inv(A)
    V = Ainv @ meat @ Ainv
    return float(b[0]), float(b[1]), float(V[0, 0])


def rdd_local_linear(x, y, cutoff=0.0, h=None, kernel="triangular"):
    r"""Sharp regression-discontinuity estimate by local linear fits.

    The sharp RD estimand is the jump in the conditional expectation
    at the threshold,

    .. math:: \tau_{SRD} = \lim_{x \downarrow c} E[Y \mid X = x]
              - \lim_{x \uparrow c} E[Y \mid X = x],

    estimated (source, Section 3: "we focus on local linear
    regression") by two kernel-weighted linear fits, one on each side
    of the cutoff, with the estimate
    :math:`\hat\tau_{RD} = \hat\alpha_+ - \hat\alpha_-` the difference
    of the two boundary intercepts. The default bandwidth is the
    Imbens-Kalyanaraman plug-in of :func:`morie.fn.causrddh.ik_bandwidth`
    and the default kernel is the edge (triangular) kernel the IK
    constant is derived for.

    On data whose conditional mean is exactly linear on each side of
    the cutoff the estimator recovers the jump exactly, for any
    bandwidth -- that limiting case is pinned in the tests, together
    with an anchor against the rdrobust package (Calonico, Cattaneo,
    Titiunik) conventional estimate at a shared bandwidth.

    Parameters
    ----------
    x : array-like
        Running (forcing) variable.
    y : array-like
        Outcome.
    cutoff : float
        Threshold c.
    h : float, optional
        Bandwidth; Imbens-Kalyanaraman plug-in when omitted.
    kernel : {'triangular', 'uniform'}
        Weight function K((x - c)/h).

    Returns
    -------
    RichResult
        ``estimate`` (tau), ``se`` (HC0 sandwich, independent sides),
        ``ci``, ``intercept_left``, ``intercept_right``,
        ``slope_left``, ``slope_right``, ``h``, ``kernel``,
        ``n_left``, ``n_right``, ``n_used``.

    References
    ----------
    Imbens, G. and Kalyanaraman, K. (2009), "Optimal Bandwidth Choice
    for the Regression Discontinuity Estimator", NBER Working Paper
    14726 (Sections 2-3, tau_RD as the difference of one-sided local
    linear limits); published as Review of Economic Studies
    79(3):933-959 (2012). Local source:
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    imbens-kalyanaraman-2009-w14726-optimal-bandwidth-rdd.pdf.
    Standard error convention: White (1980) HC0 sandwich per side, as
    in rdrobust vce = "hc0" (Calonico, Cattaneo, Titiunik 2014,
    Econometrica 82(6):2295-2326).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    c = float(cutoff)
    if h is None:
        h = float(ik_bandwidth(xa, ya, cutoff=c)["estimate"])
    h = float(h)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    d = xa - c
    u = d / h
    w = _kernel(kernel, u)
    lm = (d < 0.0) & (w > 0.0)
    rm = (d >= 0.0) & (w > 0.0)
    n_l = int(np.sum(lm.astype(float)))
    n_r = int(np.sum(rm.astype(float)))
    if n_l < 3 or n_r < 3:
        raise ValueError("fewer than 3 observations with positive "
                         "kernel weight on one side")
    al, bl, vl = _llr_side(d[lm], ya[lm], w[lm])
    ar, br, vr = _llr_side(d[rm], ya[rm], w[rm])
    tau = ar - al
    se = float(np.sqrt(vl + vr))
    z = 1.959963984540054
    return RichResult(payload={
        "estimate": float(tau),
        "se": se,
        "ci": (tau - z * se, tau + z * se),
        "intercept_left": al,
        "intercept_right": ar,
        "slope_left": bl,
        "slope_right": br,
        "h": h,
        "kernel": kernel,
        "n_left": n_l,
        "n_right": n_r,
        "n_used": n_l + n_r,
        "se_note": ("HC0 sandwich per side, sides independent; "
                    "rdrobust vce='hc0' convention"),
        "method": "sharp RDD, one-sided local linear fits at the cutoff",
    })


# primary name = module name; stub-era long name kept as alias.
causrdd = rdd_local_linear
causal_rdd_local_lin = rdd_local_linear


def cheatsheet():
    return ("causrdd: sharp RDD tau = alpha_plus - alpha_minus from "
            "kernel-weighted one-sided linear fits; IK bandwidth by "
            "default; exact on side-linear data")
