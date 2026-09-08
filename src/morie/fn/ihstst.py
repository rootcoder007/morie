# morie.fn -- function file (rootcoder007/morie)
"""Integrated haplotype score (iHS), Voight et al. 2006."""

import math

from ._richresult import RichResult
from .ehhdec import ehh_decay

__all__ = ["ihs_test"]


def _ihh_one_side(pos, ehh, core, side, min_ehh):
    """Trapezoid area of the EHH curve on one side of the core.

    Voight et al. (2006, Materials and Methods, "Calculation of iHS"):
    "The EHH values at successive SNPs are joined by straight lines,
    and then we compute the total area under each curve, between the
    nearest points to the left and right of the core SNP where the EHH
    drops below 0.05."  Accordingly the integration runs outward from
    the core and INCLUDES the segment ending at the first marker whose
    EHH falls below ``min_ehh``; if the curve never falls below the
    threshold the area is truncated at the last marker and the curve
    is flagged as truncated.
    """
    L = len(pos)
    idx = range(core + 1, L) if side > 0 else range(core - 1, -1, -1)
    area = 0.0
    prev_p, prev_e = pos[core], ehh[core]
    truncated = True
    for j in idx:
        seg = abs(pos[j] - prev_p) * 0.5 * (ehh[j] + prev_e)
        area += seg
        prev_p, prev_e = pos[j], ehh[j]
        if ehh[j] < min_ehh:
            truncated = False
            break
    return area, truncated


def ihs_test(hap, core, positions=None, min_ehh=0.05, standardize=None):
    """Integrated haplotype score for one core SNP (Voight et al. 2006).

    EHH decay curves are computed separately for the chromosomes
    carrying the ancestral (0) and derived (1) core allele (see
    :func:`morie.fn.ehhdec.ehh_decay`).  Each curve is integrated by
    the trapezoid rule away from the core in both directions until it
    drops below ``min_ehh`` (0.05 in the paper), giving iHH_A and
    iHH_D, and (their eq. 1)

        unstandardized iHS = ln(iHH_A / iHH_D).

    Standardization (their eq. 2) is a z-score within bins of derived
    allele frequency, computed over many core SNPs:
    iHS = (u - E_p[u]) / SD_p[u].  Since a single call scores one
    core, pass ``standardize=(mean, sd)`` from a genome-wide (or
    simulated) reference to obtain the standardized value; otherwise
    the unstandardized score is returned with ``standardized=False``.

    Parameters
    ----------
    hap : (N, L) array-like of 0/1
        Phased haplotypes; 1 = derived allele, 0 = ancestral.
    core : int
        Core SNP index.
    positions : (L,) array-like, optional
        Genetic (or physical) positions; index scale by default.
    min_ehh : float
        EHH integration threshold (paper value 0.05).
    standardize : (mean, sd) pair, optional
        Frequency-bin moments for the final z-score.

    Returns
    -------
    RichResult
        Keys ``estimate`` (iHS, standardized when moments given),
        ``ihs_unstandardized``, ``ihh_a``, ``ihh_d``, ``daf`` (derived
        allele frequency), ``truncated_a``, ``truncated_d``,
        ``standardized``, ``core``, ``method``.

    References
    ----------
    Voight, B. F., Kudaravalli, S., Wen, X. and Pritchard, J. K.
    (2006). A map of recent positive selection in the human genome.
    PLoS Biology 4(3), e72; eq. (1) unstandardized iHS =
    ln(iHH_A/iHH_D) p. 0446, eq. (2) standardization, and Materials
    and Methods "Calculation of iHS" (trapezoid integration to EHH <
    0.05) (fetched-wave3 PDF Voight-2006).
    Sabeti, P. C., et al. (2002). Nature 419, 832-837 (EHH).
    """
    dec = ehh_decay(hap, core, positions)
    pos = dec["positions"]
    ehh0, ehh1 = dec["ehh0"], dec["ehh1"]
    if dec["n0"] < 2 or dec["n1"] < 2:
        raise ValueError("both core alleles need at least 2 chromosomes")
    aL, tAl = _ihh_one_side(pos, ehh0, dec["core"], -1, float(min_ehh))
    aR, tAr = _ihh_one_side(pos, ehh0, dec["core"], +1, float(min_ehh))
    dL, tDl = _ihh_one_side(pos, ehh1, dec["core"], -1, float(min_ehh))
    dR, tDr = _ihh_one_side(pos, ehh1, dec["core"], +1, float(min_ehh))
    ihh_a = aL + aR
    ihh_d = dL + dR
    if ihh_a <= 0 or ihh_d <= 0:
        raise ValueError("degenerate EHH curve: zero integrated area")
    u = math.log(ihh_a / ihh_d)
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
        "estimate": est, "ihs_unstandardized": u,
        "ihh_a": ihh_a, "ihh_d": ihh_d,
        "daf": dec["n1"] / dec["n"],
        "truncated_a": bool(tAl or tAr), "truncated_d": bool(tDl or tDr),
        "standardized": std, "core": dec["core"],
        "method": "iHS (Voight 2006 eq. 1): ln(iHH_A/iHH_D), trapezoid EHH to < min_ehh",
    })


def cheatsheet():
    return "ihstst: iHS = ln(iHH_A/iHH_D); iHH = trapezoid EHH area to EHH<0.05 (Voight 2006)."


# compact alias per ledger/NAMING.md
ihstest = ihs_test
ihstst = ihs_test
