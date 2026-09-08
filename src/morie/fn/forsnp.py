"""Forensic DNA match probability + likelihood ratio (Buckleton et al. 2005)."""

from ._richresult import RichResult

__all__ = ["forsnp", "forensic_lr"]


def _locus_prob(a1, a2, freqs, theta):
    p1 = float(freqs[a1])
    if a1 == a2:
        # NRC II Recommendation 4.10a (homozygote), Buckleton Eq.:
        # [2 theta + (1-theta) p][3 theta + (1-theta) p]
        #   / [(1+theta)(1+2theta)]
        num = (2 * theta + (1 - theta) * p1) * \
              (3 * theta + (1 - theta) * p1)
        return num / ((1 + theta) * (1 + 2 * theta))
    p2 = float(freqs[a2])
    # NRC II 4.10b (heterozygote):
    # 2 [theta + (1-theta) p1][theta + (1-theta) p2]
    #   / [(1+theta)(1+2theta)]
    num = 2 * (theta + (1 - theta) * p1) * (theta + (1 - theta) * p2)
    return num / ((1 + theta) * (1 + 2 * theta))


def forsnp(genotype, freqs, theta=0.0):
    """
    Random-match probability and likelihood ratio for a DNA profile.

    Buckleton, Triggs & Walsh (2005): under the product rule the
    profile random-match probability is the product over loci of the
    single-locus genotype probabilities.  With population
    subdivision, the subpopulation-corrected (NRC II) probabilities
    are used (his reproduction of NRC II Recommendation 4.10):

      homozygote A_i A_i:
        [2t + (1-t) p][3t + (1-t) p] / [(1+t)(1+2t)],
      heterozygote A_i A_j:
        2 [t + (1-t) p_i][t + (1-t) p_j] / [(1+t)(1+2t)],

    with t the coancestry coefficient theta.  At theta = 0 these
    reduce exactly to the Hardy-Weinberg product rule (p^2 and
    2 p_i p_j).  The likelihood ratio for the prosecution hypothesis
    (the suspect is the source) versus the defence hypothesis (an
    unrelated random person) is LR = 1 / RMP.

    Sources
    -------
    Buckleton, J., Triggs, C. M. & Walsh, S. J. (2005). *Forensic
    DNA Evidence Interpretation*. CRC Press, Ch. 3 (product rule,
    NRC II 4.10 subpopulation correction) (local copy fetched-wave3/
    Forensic_DNA_Evidence_Interpretation..pdf).

    Parameters
    ----------
    genotype : sequence of (allele1, allele2) per locus
        The matching profile.
    freqs : sequence of dict
        Per-locus allele frequency dict.
    theta : float
        Coancestry coefficient (0 = product rule; 0.01-0.03 typical).

    Returns
    -------
    RichResult
        Keys: rmp (random match probability), lr, locus_rmp,
        n_loci, theta.
    """
    if len(genotype) != len(freqs) or not genotype:
        raise ValueError("genotype and freqs must be paired, non-empty")
    theta = float(theta)
    if not (0.0 <= theta < 1.0):
        raise ValueError("theta must be in [0, 1)")
    locus = []
    rmp = 1.0
    for (a1, a2), fr in zip(genotype, freqs):
        if a1 not in fr or a2 not in fr:
            raise ValueError("allele frequency missing")
        p = _locus_prob(a1, a2, fr, theta)
        locus.append(p)
        rmp *= p
    if rmp <= 0:
        raise ValueError("zero match probability")
    return RichResult(payload={
        "rmp": rmp,
        "lr": 1.0 / rmp,
        "locus_rmp": locus,
        "n_loci": len(genotype),
        "theta": theta,
        "method": "forensic RMP/LR, NRC II 4.10 (Buckleton 2005)",
    })


# long descriptive alias (stub-era name)
forensic_lr = forsnp


def cheatsheet():
    return "forsnp: product-rule RMP with NRC II theta correction; LR = 1/RMP"

# public names resolved by fn/_lazy_map.json
forensiclr = forsnp
