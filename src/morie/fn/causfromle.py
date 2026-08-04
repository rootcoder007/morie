# morie.fn -- function file (rootcoder007/morie)
"""VanderWeele-Ding E-value -- front-end to evalu."""

__all__ = ["causal_e_value"]


def causal_e_value(RR):
    r"""E-value for a point risk ratio.

    Delegates to :func:`morie.fn.evalu.evalue`; this entry point exists
    under the historical name whose placeholder body returned the mean
    of RR. See that module for the formula and reference (VanderWeele
    & Ding 2017, *Ann Intern Med* 167(4), 268-274).
    """
    from .evalu import evalue

    return evalue(RR)


def cheatsheet():
    return "causfromle: E-value for a point RR (front-end to evalu)"


# compact alias per ledger/NAMING.md
causalevalue = causal_e_value
