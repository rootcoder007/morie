# morie.fn -- function file (rootcoder007/morie)
"""Local average treatment effect."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_iv_late"]


def causal_iv_late(y, D, Z):
    r"""The local average treatment effect,

    .. math:: \mathrm{LATE} = \frac{E[Y|Z=1] - E[Y|Z=0]}
                                   {E[D|Z=1] - E[D|Z=0]} .

    Arithmetically this is the Wald ratio. What Imbens and Angrist
    established is what it *estimates*, and that is the whole content
    of the result: under independence of the instrument, exclusion,
    a non-zero first stage, and MONOTONICITY -- no defiers, nobody
    whom the instrument pushes out of treatment -- the ratio is the
    average effect among COMPLIERS alone.

    That is a different estimand from the average treatment effect,
    and the difference is not a technicality. The compliers are
    defined by their response to this particular instrument; a
    different instrument identifies a different subpopulation and so
    a different number, and neither is the population average unless
    effects are homogeneous. ``complier_share`` is the denominator,
    :math:`P(D_1 > D_0)`, and it says what fraction of the sample the
    estimate actually describes -- a LATE from a 4% complier share is
    a statement about 4% of the sample, however tight its confidence
    interval.

    Monotonicity is not testable from the data, but a NECESSARY
    consequence of it is: with no defiers the first stage must have
    the same sign in every subgroup. ``first_stage`` is reported so
    that a negative value, which means the coding of ``Z`` or ``D``
    is reversed relative to the intended direction, is visible rather
    than silently flipping the sign of the estimate.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    d : array-like of 0/1, shape (n,)
        Treatment actually taken.
    z : array-like of 0/1, shape (n,)
        Instrument.

    Returns
    -------
    RichResult
        keys: ``late``, ``se``, ``first_stage``, ``reduced_form``,
        ``complier_share``, ``n_z1``, ``n_z0``, ``weak_first_stage``,
        ``estimand``, ``monotonicity_assumed``, ``n``, ``method``.

    References
    ----------
    Imbens, G. W. and Angrist, J. D. (1994), "Identification and
    estimation of local average treatment effects", *Econometrica*
    62:467-475. Angrist, Imbens and Rubin (1996), *JASA* 91:444-455.
    """
    yv = np.asarray(y, dtype=float).ravel()
    Dv = np.asarray(D, dtype=float).ravel()
    Zv = np.asarray(Z, dtype=float).ravel()
    if not (yv.size == Dv.size == Zv.size):
        raise ValueError("y, D and Z must have the same length.")
    for nm, v in (("D", Dv), ("Z", Zv)):
        if not np.all(np.isin(v, (0.0, 1.0))):
            raise ValueError(f"{nm} must be binary 0/1 for the LATE theorem.")
    z1, z0 = Zv == 1, Zv == 0
    n1, n0 = int(z1.sum()), int(z0.sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(
            f"need at least 2 observations in each arm, got {n1} and {n0}.")
    rf = float(yv[z1].mean() - yv[z0].mean())
    fs = float(Dv[z1].mean() - Dv[z0].mean())
    if abs(fs) < 1e-12:
        raise ValueError(
            "the first stage is zero: the instrument does not move treatment, "
            "so the Wald ratio is 0/0 and nothing is identified.")
    late = rf / fs
    # delta-method standard error for a ratio of two independent
    # differences in means
    vy = yv[z1].var(ddof=1) / n1 + yv[z0].var(ddof=1) / n0
    vd = Dv[z1].var(ddof=1) / n1 + Dv[z0].var(ddof=1) / n0
    cov = (np.cov(yv[z1], Dv[z1])[0, 1] / n1
           + np.cov(yv[z0], Dv[z0])[0, 1] / n0)
    var = (vy - 2 * late * cov + late ** 2 * vd) / fs ** 2
    return RichResult(payload={
        "late": late, "se": float(np.sqrt(max(var, 0.0))),
        "first_stage": fs, "reduced_form": rf,
        "complier_share": fs,
        "n_z1": n1, "n_z0": n0,
        "weak_first_stage": bool(abs(fs) < 0.05),
        "estimand": "the average effect among COMPLIERS, not the population "
                    "average treatment effect",
        "monotonicity_assumed": "no defiers: nobody the instrument pushes "
                                "OUT of treatment. Not testable, but its "
                                "consequence -- a first stage of one sign "
                                "throughout -- is visible above",
        "interpretation_warning":
            "the compliers are defined by their response to THIS instrument; "
            "a different instrument identifies a different subpopulation and "
            "so a different number, and neither is the ATE unless effects "
            "are homogeneous",
        "n": int(yv.size),
        "method": "Imbens-Angrist LATE, the Wald ratio under monotonicity"})


def cheatsheet():
    return "causivla: LATE is the compliers' effect -- check complier_share to see who that is"
