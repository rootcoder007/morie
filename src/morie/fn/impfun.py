# morie.fn -- function file (rootcoder007/morie)
r"""IMPUTE2: genotype imputation across multiple reference panels.

Imputation fills in untyped genotypes from reference haplotypes, and
the panel bounds what is achievable: with HapMap as the sole
reference the improvement possible is constrained by the panel itself,
while expanding to thousands of chromosomes greatly increases accuracy
at both rare and common SNPs.

**The hard part is that panels disagree about which SNPs they carry.**
Controls genotyped on several chip designs and densely typed
sequencing samples cover different, overlapping marker sets. Merging
them by *intersection* discards exactly the extra coverage that
motivated merging.

**The framework separates two roles.** SNPs typed in the study align
the study haplotypes to the reference; SNPs missing from the study are
the targets. A reference haplotype can be a template at one SNP and
uninformative at another, so panels combine by role rather than by
intersection -- the flexibility that lets panels typed on different
SNP sets be used together.

**Underneath is the Li-Stephens copying model.** A study haplotype is
a mosaic of reference haplotypes; a hidden Markov chain switches
template at a rate set by recombination and absorbs mismatch through a
mutation parameter. The imputed dosage is a posterior mean, so it
carries uncertainty -- conflicting templates give a middling dosage
rather than a confident wrong call.

**Accuracy is measured on masked truth**, never on the model's own
posterior: a confident model can be confidently wrong.

References
----------
Howie, B. N., Donnelly, P. & Marchini, J. (2009) "A Flexible and
Accurate Genotype Imputation Method for the Next Generation of
Genome-Wide Association Studies", *PLoS Genetics* 5(6), e1000529,
doi:10.1371/journal.pgen.1000529. The main innovation as a flexible
modelling framework that increases accuracy and combines information
across MULTIPLE REFERENCE PANELS while remaining computationally
feasible; higher accuracy than other methods when HapMap provides the
sole reference panel, with the panel size constraining the
improvements possible; greatly enhanced accuracy from expanding the
panel to thousands of chromosomes, outperforming other methods at both
rare and common SNPs with error rates 15-20% lower than the closest
competitor; and the practical advantages of this approach to
integrating information across panels genotyped on different sets of
SNPs.

Li, N. & Stephens, M. (2003) "Modeling Linkage Disequilibrium and
Identifying Recombination Hotspots Using Single-Nucleotide
Polymorphism Data", *Genetics* 165(4), 2213-2233,
doi:10.1093/genetics/165.4.2213. The copying model.

Browning, S. R. & Browning, B. L. (2007) "Rapid and Accurate Haplotype
Phasing and Missing-Data Inference for Whole-Genome Association
Studies by Use of Localized Haplotype Clustering", *American Journal
of Human Genetics* 81(5), 1084-1097, doi:10.1086/521987.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["merge_panels", "copying_model", "impute_dosage",
           "info_score", "concordance"]

_EPS = 1e-12


def merge_panels(panels, study_snps):
    r"""Combine panels by ROLE, not by intersection."""
    if not panels:
        raise ValueError("impfun: no reference panels given")
    study = set(study_snps)
    all_snps, per = set(), {}
    for name, snps in panels.items():
        per[name] = set(snps)
        all_snps |= per[name]
    inter = set.intersection(*per.values()) if per else set()
    scaffold = sorted(all_snps & study)
    if not scaffold:
        raise ValueError("impfun: no SNP is typed in both the study "
                         "and a panel; there is nothing to align "
                         "against")
    return {"scaffold": scaffold,
            "targets": sorted(all_snps - study),
            "union": sorted(all_snps), "intersection": sorted(inter),
            "kept_by_union": len(all_snps),
            "kept_by_intersection": len(inter),
            "gain": len(all_snps) - len(inter),
            "note": "intersection would discard the coverage that "
                    "motivated adding the panel"}


def copying_model(study_hap, reference_haps, rho=0.001, theta=0.01):
    r"""Li-Stephens: the study haplotype as a mosaic of references."""
    h = [int(v) for v in k.vec(study_hap)]
    R = [[int(v) for v in r] for r in k.mat(reference_haps)]
    K, L = len(R), len(h)
    if K < 1 or L < 1:
        raise ValueError("impfun: need at least one reference "
                         "haplotype and one site")
    if any(len(r) != L for r in R):
        raise ValueError("impfun: a reference haplotype has the wrong "
                         "length")
    r_, t_ = float(rho), float(theta)
    if not 0.0 < r_ < 1.0 or not 0.0 < t_ < 0.5:
        raise ValueError("impfun: rho must lie in (0,1) and theta in "
                         "(0,0.5)")

    def emit(kk, l):
        return 1.0 - t_ if R[kk][l] == h[l] else t_

    F = [[0.0] * K for _ in range(L)]
    scale = []
    for kk in range(K):
        F[0][kk] = emit(kk, 0) / K
    s = sum(F[0]) or 1.0
    F[0] = [v / s for v in F[0]]
    scale.append(s)
    for l in range(1, L):
        tot = sum(F[l - 1])
        for kk in range(K):
            F[l][kk] = ((1.0 - r_) * F[l - 1][kk]
                        + r_ * tot / K) * emit(kk, l)
        s = sum(F[l]) or 1.0
        F[l] = [v / s for v in F[l]]
        scale.append(s)
    return {"posterior": F, "n_templates": K, "n_sites": L,
            "log_likelihood": sum(math.log(max(v, _EPS))
                                  for v in scale)}


def impute_dosage(posterior, reference_haps, site):
    r"""Posterior mean over templates -- uncertainty included."""
    P = [[float(v) for v in r] for r in k.mat(posterior)]
    R = [[int(v) for v in r] for r in k.mat(reference_haps)]
    l = int(site)
    if l < 0 or l >= len(P):
        raise ValueError("impfun: site %d is outside the region" % l)
    w = P[l]
    tot = sum(w) or 1.0
    p1 = sum(w[kk] * R[kk][l] for kk in range(len(R))) / tot
    return {"dosage": 2.0 * p1, "allele_freq": p1,
            "certainty": max(p1, 1.0 - p1),
            "note": "conflicting templates give a middling dosage, "
                    "which is the honest answer"}


def info_score(dosages):
    r"""The IMPUTE info measure -- how much information was
    recovered."""
    d = [float(v) for v in k.vec(dosages)]
    n = len(d)
    if n < 2:
        raise ValueError("impfun: at least 2 individuals are needed")
    theta = sum(d) / (2.0 * n)
    if theta <= _EPS or theta >= 1.0 - _EPS:
        return {"info": 1.0, "theta": theta,
                "note": "monomorphic: no information to lose"}
    m = sum(d) / n
    var_d = sum((v - m) ** 2 for v in d) / n
    return {"info": min(max(var_d / (2.0 * theta * (1.0 - theta)),
                            0.0), 1.0),
            "theta": theta,
            "note": "filtering on info is how badly-imputed SNPs are "
                    "excluded before testing"}


def concordance(imputed, truth):
    r"""Accuracy against MASKED truth, not the model's own
    posterior."""
    a = [float(v) for v in k.vec(imputed)]
    b = [float(v) for v in k.vec(truth)]
    if len(a) != len(b):
        raise ValueError("impfun: %d imputed but %d true genotypes"
                         % (len(a), len(b)))
    ok = sum(1 for i in range(len(a)) if round(a[i]) == round(b[i]))
    return RichResult(payload={
        "estimate": ok / float(len(a)),
        "concordance": ok / float(len(a)),
        "mean_absolute_error": sum(abs(a[i] - b[i])
                                   for i in range(len(a))) / len(a),
        "n": len(a),
        "method": "IMPUTE2 evaluation on masked genotypes; Howie, "
                  "Donnelly & Marchini (2009)",
    })


def cheatsheet():
    return ("impfun: imputation is bounded by the REFERENCE PANEL, and "
            "panels disagree about which SNPs they carry -- merging by "
            "INTERSECTION discards the coverage that motivated "
            "merging. IMPUTE2 merges by ROLE: SNPs typed in the study "
            "align the haplotypes, the rest are targets. Underneath is "
            "Li-Stephens copying, the study haplotype as a MOSAIC of "
            "references switching at the recombination rate. Dosages "
            "carry uncertainty, and accuracy is measured on MASKED "
            "truth, because a confident model can be confidently "
            "wrong.")


# compact alias per ledger/NAMING.md
impute2 = copying_model

# public names resolved by fn/_lazy_map.json
genotype_imputation = copying_model
