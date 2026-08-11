# morie.fn -- function file (rootcoder007/morie)
"""Cross-population EHH (XP-EHH), Sabeti et al. 2007."""

import math

from ._richresult import RichResult
from .ehhdec import ehh_decay
from .ihstst import _ihh_one_side

__all__ = ["xp_ehh"]


def xp_ehh(hapA, hapB, core, positions=None, min_ehh=0.05,
           standardize=None):
    """Cross-population extended haplotype homozygosity test.

    Sabeti et al. (2007, Methods p. 6): EHH between the core SNP and a
    marker X is "the probability that two randomly chosen chromosomes
    are homozygous at all SNPs between A and B, inclusive", computed
    over ALL N chromosomes of one population:
    EHH = sum_i C(n_i, 2) / C(N, 2) over the G homozygous groups.
    "EHH is calculated for all SNPs in population A between the core
    SNP and X, and the value integrated with respect to genetic
    distance, with the result defined as I_A.  I_B is defined
    analogously ...  The statistic ln(I_A/I_B) is then calculated";
    positive values suggest selection in population A, negative in B.
    "For identifying outliers, the log-ratio is normalized to have
    zero mean and unit variance" genome-wide; pass
    ``standardize=(mean, sd)`` from such a reference to obtain the
    normalized value.

    The integration mirrors :func:`morie.fn.ihstst.ihs_test`:
    trapezoid rule outward from the core in both directions, stopping
    after the first marker whose site-EHH falls below ``min_ehh``
    (both populations use the same rule, so local recombination-rate
    variation cancels in the ratio, which is the point of the test).

    Parameters
    ----------
    hapA, hapB : (N_A, L), (N_B, L) array-like of 0/1
        Phased haplotypes of populations A and B over the SAME L SNPs.
    core : int
        Core SNP index.
    positions : (L,) array-like, optional
        Genetic positions; index scale by default.
    min_ehh : float
        Integration stop threshold.
    standardize : (mean, sd) pair, optional
        Genome-wide moments for normalization.

    Returns
    -------
    RichResult
        Keys ``estimate`` (XP-EHH, normalized when moments given),
        ``xpehh_unstandardized``, ``I_A``, ``I_B``, ``truncated_a``,
        ``truncated_b``, ``standardized``, ``core``, ``method``.

    References
    ----------
    Sabeti, P. C., Varilly, P., et al. (2007). Genome-wide detection
    and characterization of positive selection in human populations.
    Nature 449(7164), 913-918; Methods pp. 5-6, displayed EHH equation
    and the XP-EHH definition ln(I_A/I_B) (fetched-wave3 PDF
    Sabeti-2007).
    """
    decA = ehh_decay(hapA, core, positions)
    decB = ehh_decay(hapB, core, positions)
    if len(decA["positions"]) != len(decB["positions"]):
        raise ValueError("populations must cover the same SNPs")
    pos = decA["positions"]
    me = float(min_ehh)
    aL, tAl = _ihh_one_side(pos, decA["ehhs"], decA["core"], -1, me)
    aR, tAr = _ihh_one_side(pos, decA["ehhs"], decA["core"], +1, me)
    bL, tBl = _ihh_one_side(pos, decB["ehhs"], decB["core"], -1, me)
    bR, tBr = _ihh_one_side(pos, decB["ehhs"], decB["core"], +1, me)
    IA = aL + aR
    IB = bL + bR
    if IA <= 0 or IB <= 0:
        raise ValueError("degenerate EHH curve: zero integrated area")
    u = math.log(IA / IB)
    if standardize is not None:
        mean, sd = float(standardize[0]), float(standardize[1])
        if sd <= 0:
            raise ValueError("standardize sd must be positive")
        est = (u - mean) / sd
        std = True
    else:
        est = u
        std = False
    return RichResult(payload={
        "estimate": est, "xpehh_unstandardized": u,
        "I_A": IA, "I_B": IB,
        "truncated_a": bool(tAl or tAr), "truncated_b": bool(tBl or tBr),
        "standardized": std, "core": int(decA["core"]),
        "method": "XP-EHH (Sabeti 2007): ln(I_A/I_B) of integrated site-EHH",
    })


def cheatsheet():
    return "xpehh1: XP-EHH = ln(I_A/I_B); I = trapezoid area of all-chromosome EHH (Sabeti 2007)."


# compact alias per ledger/NAMING.md
xpehh = xp_ehh
