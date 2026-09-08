# morie.fn -- function file (rootcoder007/morie)
"""E-value with confidence interval -- front-end to evalu."""

__all__ = ["e_value_unmeasured_confounding"]


def e_value_unmeasured_confounding(estimate, ci_lower, ci_upper):
    r"""E-value for a point estimate and its confidence interval.

    Delegates to :func:`morie.fn.evalu.evalue`; this entry point exists
    under the historical name whose placeholder body averaged its three
    arguments. See that module for the formula and reference
    (VanderWeele & Ding 2017, *Ann Intern Med* 167(4), 268-274).
    """
    from .evalu import evalue

    return evalue(estimate, ci_lower=ci_lower, ci_upper=ci_upper)


def cheatsheet():
    return "evaltw: E-value with CI (front-end to evalu)"
