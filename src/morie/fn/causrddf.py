# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Fuzzy RDD: Wald ratio of the outcome and treatment jumps."""

from . import _array_core as np

from ._richresult import RichResult
from .causrdd import rdd_local_linear
from .causrddh import ik_bandwidth

__all__ = ["causrddf", "rdd_fuzzy", "causal_rdd_fuzzy"]


def rdd_fuzzy(x, y, treat, cutoff=0.0, h=None, h_treat=None,
              kernel="triangular"):
    r"""Fuzzy regression-discontinuity estimate (ratio of jumps).

    In the fuzzy design the treatment probability, not the treatment
    itself, jumps at the threshold. The estimand (Hahn, Todd and
    van der Klaauw 2001; printed as the object of interest in the
    source in hand, Imbens-Kalyanaraman NBER WP 14726, Section 5.1) is

    .. math:: \tau_{FRD} = \frac{\lim_{x\downarrow c} E[Y|X=x]
              - \lim_{x\uparrow c} E[Y|X=x]}
              {\lim_{x\downarrow c} E[W|X=x]
              - \lim_{x\uparrow c} E[W|X=x]},

    the outcome jump divided by the first-stage jump, each estimated
    by one-sided local linear fits as in the sharp design
    (:func:`morie.fn.causrdd.rdd_local_linear`). Following the
    source's Section 5.1 (which adopts the Imbens-Lemieux 2008
    suggestion), separate Imbens-Kalyanaraman bandwidths are chosen
    for the outcome and treatment regressions by default. NOTE: the
    working paper's displayed denominator contains an evident typo
    ("E[Y_i | W_i = x]"); the denominator is the treatment regression
    E[W | X = x], as the surrounding text states.

    The standard error is the delta-method expansion of the ratio
    with independent numerator and denominator jumps,
    :math:`se^2 = (se_Y^2 + \tau_{FRD}^2 se_W^2)/\hat\tau_W^2`;
    the covariance term of the full Imbens-Lemieux expression is
    omitted and this is documented, not hidden -- tests anchor the
    POINT estimate exactly (the ratio of the two sharp fits) and the
    limiting case: with a sharp first stage (W jumps 0 to 1) the
    estimator equals the sharp RD estimate exactly.

    Parameters
    ----------
    x : array-like
        Running variable.
    y : array-like
        Outcome.
    treat : array-like
        Treatment received (0/1 or a probability).
    cutoff : float
        Threshold c.
    h : float, optional
        Outcome bandwidth; IK plug-in on (x, y) when omitted.
    h_treat : float, optional
        First-stage bandwidth; IK plug-in on (x, treat) when omitted.
    kernel : {'triangular', 'uniform'}
        Weight function.

    Returns
    -------
    RichResult
        ``estimate`` (tau_FRD), ``se`` (delta method), ``ci``,
        ``jump_outcome``, ``jump_treatment``, ``se_outcome``,
        ``se_treatment``, ``h_outcome``, ``h_treatment``,
        ``sharp_outcome`` and ``sharp_treatment`` (the two full
        one-sided fits).

    References
    ----------
    Hahn, J., Todd, P. and van der Klaauw, W. (2001),
    "Identification and Estimation of Treatment Effects with a
    Regression-Discontinuity Design", Econometrica 69(1):201-209
    (origin of the estimand; original text not yet in the local
    registry -- logged in wave3/NEEDED_SOURCES.md). Implemented from
    Imbens, G. and Kalyanaraman, K. (2009), NBER Working Paper 14726,
    Section 5.1 (tau_FRD display and the two-bandwidth prescription);
    local source /run/media/rootcoder/WD_BLACK/library/pdf/
    fetched-wave3/imbens-kalyanaraman-2009-w14726-optimal-bandwidth-rdd.pdf.
    Imbens, G. and Lemieux, T. (2008), "Regression discontinuity
    designs: A guide to practice", Journal of Econometrics
    142(2):615-635 (separate-bandwidth suggestion, as cited by the
    source).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    wa = np.asarray(treat, dtype=float)
    c = float(cutoff)
    if h is None:
        h = float(ik_bandwidth(xa, ya, cutoff=c)["estimate"])
    if h_treat is None:
        h_treat = float(ik_bandwidth(xa, wa, cutoff=c)["estimate"])
    fy = rdd_local_linear(xa, ya, cutoff=c, h=h, kernel=kernel)
    fw = rdd_local_linear(xa, wa, cutoff=c, h=h_treat, kernel=kernel)
    ty = float(fy["estimate"])
    tw = float(fw["estimate"])
    if abs(tw) < 1e-12:
        raise ValueError("no first-stage discontinuity: the treatment "
                         "jump at the cutoff is numerically zero")
    tau = ty / tw
    se = float(np.sqrt((fy["se"] ** 2 + tau ** 2 * fw["se"] ** 2)
                       / tw ** 2))
    z = 1.959963984540054
    return RichResult(payload={
        "estimate": float(tau),
        "se": se,
        "ci": (tau - z * se, tau + z * se),
        "jump_outcome": ty,
        "jump_treatment": tw,
        "se_outcome": float(fy["se"]),
        "se_treatment": float(fw["se"]),
        "h_outcome": float(h),
        "h_treatment": float(h_treat),
        "kernel": kernel,
        "sharp_outcome": dict(fy),
        "sharp_treatment": dict(fw),
        "se_note": ("delta method with independent jumps; the "
                    "Imbens-Lemieux covariance term is omitted and "
                    "documented"),
        "method": "fuzzy RDD, Wald ratio of local linear jumps",
    })


# primary name = module name; stub-era long name kept as alias.
causrddf = rdd_fuzzy
causal_rdd_fuzzy = rdd_fuzzy


def cheatsheet():
    return ("causrddf: fuzzy RDD tau = (outcome jump)/(treatment jump), "
            "one-sided local linear fits, separate IK bandwidths; "
            "reduces to the sharp estimate when the first stage is sharp")

# public names resolved by fn/_lazy_map.json
causalrddfuzzy = rdd_fuzzy
