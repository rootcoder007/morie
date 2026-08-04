# morie.fn -- function file (rootcoder007/morie)
"""Worst-case bias bound under local misspecification."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["misspecbd", "bound_specification"]


def misspecbd(estimate, sensitivity, c, se, conf=0.95):
    """Bias-aware interval when the model may be wrong by at most c.

    Suppose the estimator is exactly unbiased under a baseline model and
    that misspecification enters linearly through a perturbation vector
    gamma of size at most c in the Euclidean norm.  If s is the
    sensitivity of the estimate to that perturbation, then the bias is
    s' gamma, and its worst case over the neighbourhood is, by
    Cauchy-Schwarz,

        |bias| <= c ||s||_2,

    attained at gamma = c s / ||s||.  A conservative interval simply adds
    that bound to the usual sampling interval,

        estimate +/- ( c ||s|| + z_{1 - alpha/2} se ),

    which has at least the nominal coverage uniformly over the
    neighbourhood.  This is deliberately the conservative interval, not
    the shorter fixed-length interval that solves for the optimal
    critical value of a non-central folded normal; the shorter one is
    valid too but needs a root find, and this routine states plainly
    which of the two it reports.

    Parameters
    ----------
    estimate : float
        Point estimate under the baseline model.
    sensitivity : array-like
        Derivative of the estimate with respect to the perturbation.
    c : float
        Radius of the misspecification neighbourhood, non-negative.
    se : float
        Standard error under the baseline model.
    conf : float
        Nominal confidence level.

    Returns
    -------
    RichResult
        ``bias``, ``lower``, ``upper``, ``halfwidth``, ``worstgamma``,
        ``normsens``, ``z``, ``c``.

    References
    ----------
    The bias bound |s' gamma| <= c ||s|| is Cauchy-Schwarz, and the
    bias-aware interval construction is Armstrong, T. B. and Kolesar, M.
    (2021), "Sensitivity analysis using approximate moment condition
    models", Quantitative Economics 12(1), 77-108, Sect. 2, whose
    fixed-length variant replaces the normal quantile by the quantile of
    a folded non-central normal.  The k01 worklist attributed this row to
    Andrews and Kasy (2019), Amerian Economic Review 109(8), 2766-2794,
    but that article identifies and corrects *publication* bias and
    contains no misspecification-bound of this kind, so it is not cited
    as the source here.  Standard published form; the Quantitative
    Economics article was not in the local corpus and was not read for
    this implementation.
    """
    s = C.vec(sensitivity)
    c = float(c)
    if c < 0.0:
        raise ValueError("c must be non-negative")
    se = float(se)
    if se < 0.0:
        raise ValueError("se must be non-negative")
    ns = math.sqrt(sum(v * v for v in s))
    bias = c * ns
    z = C.qnorm(0.5 + 0.5 * float(conf))
    hw = bias + z * se
    est = float(estimate)
    wg = [0.0] * len(s) if ns == 0.0 else [c * v / ns for v in s]
    return RichResult(payload={
        "bias": bias, "lower": est - hw, "upper": est + hw,
        "halfwidth": hw, "worstgamma": wg, "normsens": ns, "z": z,
        "c": c,
        "method": "Conservative bias-aware interval under local misspecification"})


bound_specification = misspecbd


def cheatsheet():
    return "bdspcf: Worst-case bias bound under local misspecification."
