# morie.fn -- function file (rootcoder007/morie)
"""Hodges-Lehmann efficiency bounds."""

from . import _array_core as np

from ._gb_are import (
    ARE_TABLE,
    HL_SIGN_LOWER_BOUND,
    HL_WILCOXON_LOWER_BOUND,
    efficacy_are,
)
from ._richresult import RichResult

__all__ = ["gibbons_are_kw"]


def gibbons_are_kw(distribution="normal", cdf=None, f=None):
    r"""The Hodges-Lehmann (1956) bounds quoted in Gibbons p. 492-493:

    .. math:: \mathrm{ARE}(T^+, t) \ge 0.864

    over ALL continuous symmetric distributions (the infimum is
    attained by a parabolic density, not by any textbook family), and
    :math:`\mathrm{ARE}(K, t) \ge 1/3` over continuous unimodal
    symmetric distributions, attained at the uniform.

    Evaluates the Wilcoxon-vs-t ARE at the requested distribution (or
    a supplied density) and reports it against the bound -- every
    admissible density must land above 0.864, which the tests check
    for all four table families.

    Parameters
    ----------
    distribution : str
        One of the Table 13.3.1 families, used when f is omitted.
    cdf : ignored
        Interface compatibility.
    f : callable, optional
        A density to evaluate instead.

    Returns
    -------
    RichResult
        keys: ``are_wilcoxon_t``, ``hl_bound`` (0.864),
        ``above_bound``, ``sign_bound`` (1/3), ``distribution``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 13.3.

    Hodges, J. L. & Lehmann, E. L. (1956). The efficiency of some
    nonparametric competitors of the t-test. *The Annals of
    Mathematical Statistics*, 27(2), 324-335.
    """
    if f is not None:
        are = efficacy_are(f)["wilcoxon_vs_t"]
    else:
        if distribution not in ARE_TABLE:
            raise ValueError(
                f"distribution must be one of {sorted(ARE_TABLE)}, got "
                f"{distribution!r}."
            )
        are = ARE_TABLE[distribution]["wilcoxon_vs_t"]
    return RichResult(
        payload={
            "are_wilcoxon_t": float(are), "hl_bound": HL_WILCOXON_LOWER_BOUND,
            "above_bound": bool(are >= HL_WILCOXON_LOWER_BOUND - 1e-9),
            "sign_bound": HL_SIGN_LOWER_BOUND,
            "distribution": None if f is not None else distribution,
            "method": "Hodges-Lehmann bounds: ARE(T+, t) >= 0.864 (Gibbons p. 492)",
        }
    )


def cheatsheet():
    return "gb_are4: HL bound 0.864 for Wilcoxon-vs-t; 1/3 for the sign test"
