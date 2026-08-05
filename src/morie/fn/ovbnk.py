# morie.fn -- function file (rootcoder007/morie)
"""Oster bound on bias from omitted variables."""

from ._richresult import RichResult

__all__ = ["oster_omitted_bias_bound"]


def oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long,
                             R_max=1.0, delta=1.0):
    """Bias-adjusted treatment effect under proportional selection.

    Coefficient stability alone is not evidence: a coefficient that
    barely moves when controls are added is uninformative unless the
    controls also moved the R-squared.  Oster's adjustment ties the two
    together through ``delta``, the ratio of selection on unobservables
    to selection on observables.

    Formula (Oster 2019, the approximation used in practice):

        beta* = beta_long
                - delta * (beta_short - beta_long)
                        * (R_max - R_long) / (R_long - R_short)

    with ``beta_short, R_short`` from the regression without controls
    and ``beta_long, R_long`` from the regression with them.  The
    identified set of interest is ``[beta_long, beta*]`` at
    ``delta = 1``.  Also returned is ``delta_star``, the value of
    ``delta`` at which ``beta*`` is exactly zero,

        delta_star = beta_long * (R_long - R_short)
                     / ((beta_short - beta_long) * (R_max - R_long)),

    the standard "how much stronger would selection on unobservables
    have to be to explain the result away" statistic.

    Parameters
    ----------
    beta_short : float
        Treatment coefficient from the uncontrolled regression.
    beta_long : float
        Treatment coefficient from the controlled regression.
    R_short : float
        R-squared of the uncontrolled regression.
    R_long : float
        R-squared of the controlled regression; must exceed ``R_short``.
    R_max : float, default 1.0
        R-squared of the hypothetical regression on treatment plus all
        observed and unobserved controls.
    delta : float, default 1.0
        Proportional-selection coefficient.

    Returns
    -------
    RichResult
        ``estimate`` (``beta*``), ``beta_star``, ``bias``,
        ``delta_star``, ``bound_lower``, ``bound_upper``,
        ``sign_stable`` (whether the identified set excludes zero).

    References
    ----------
    Oster, E. (2019).  Unobservable selection and coefficient stability:
    theory and evidence.  Journal of Business & Economic Statistics,
    37(2), 187--204.  doi:10.1080/07350015.2016.1227711
    """
    bs, bl = float(beta_short), float(beta_long)
    rs, rl, rm, d = float(R_short), float(R_long), float(R_max), float(delta)
    if not (rl > rs):
        raise ValueError("oster_omitted_bias_bound: R_long must exceed R_short")
    if rm < rl:
        raise ValueError("oster_omitted_bias_bound: R_max must be at least R_long")
    if not (0.0 <= rs <= 1.0 and 0.0 <= rl <= 1.0 and 0.0 <= rm <= 1.0):
        raise ValueError("oster_omitted_bias_bound: R-squared values must lie in [0, 1]")
    scale = (rm - rl) / (rl - rs)
    bias = d * (bs - bl) * scale
    beta_star = bl - bias
    denom = (bs - bl) * (rm - rl)
    delta_star = bl * (rl - rs) / denom if denom != 0.0 else float("inf")
    lo, hi = (beta_star, bl) if beta_star <= bl else (bl, beta_star)
    return RichResult(payload={
        "estimate": beta_star, "beta_star": beta_star, "bias": bias,
        "delta_star": delta_star, "bound_lower": lo, "bound_upper": hi,
        "sign_stable": 1.0 if lo * hi > 0.0 else 0.0,
        "method": "Oster (2019) proportional-selection bias bound"})


def cheatsheet():
    return "ovbnk: Oster bias bound under proportional selection"


osteromittedbiasbound = oster_omitted_bias_bound
