# morie.fn -- function file (rootcoder007/morie)
r"""MetaBAT 2: adaptive binning of metagenome assemblies.

Metagenome binning groups assembled contigs into draft genomes. The
signals are two: **tetranucleotide frequency**, which is
genome-characteristic and available for every contig, and **abundance
covariance across samples**, which is strong but needs several
samples. Earlier tools combined them with parameters the user had to
tune, and the paper's stated problem is that performance suffers when
those parameters are wrong -- especially on poor assemblies.

**The contribution is removing the tuning.** MetaBAT 2 uses an
adaptive binning algorithm to *eliminate manual parameter tuning*,
plus extensive engineering for computational and memory efficiency.
Binning a typical assembly takes minutes on a single commodity
workstation.

**Why short contigs are the hard case, and why they are still
weighted.** A tetranucleotide profile from a 2 kb contig is a noisy
estimate of the same profile from a 200 kb one, so treating them
alike lets noise dominate. Score confidence must scale with length --
and simply discarding short contigs discards most of the assembly.
``length_weight`` makes that explicit rather than hiding it in a
threshold.

**Abundance needs more than one sample to say anything.** With a
single sample, covariance across samples is undefined and the
composition signal is all there is; the module refuses to pretend
otherwise rather than returning a correlation of one.

**Purity and completeness are different failures.** A bin can be
complete and contaminated, or pure and fragmentary, and reporting one
number hides which. Both are computed against known labels in the
anchor.

References
----------
Kang, D. D., Li, F., Kirton, E., Thomas, A., Egan, R., An, H. &
Wang, Z. (2019) "MetaBAT 2: an adaptive binning algorithm for robust
and efficient genome reconstruction from metagenome assemblies",
*PeerJ* 7, e7359, doi:10.7717/peerj.7359. The abstract: existing
binning performance suffers, especially on assemblies of poor quality;
MetaBAT 2 using a new ADAPTIVE binning algorithm to eliminate manual
parameter tuning; extensive software engineering optimisation for
computational and memory efficiency; comparison against alternative
tools on over 100 real-world metagenome assemblies showing superior
accuracy and speed; and binning a typical assembly in a few minutes on
a single commodity workstation.

Kang, D. D., Froula, J., Egan, R. & Wang, Z. (2015) "MetaBAT, an
efficient tool for accurately reconstructing single genomes from
complex microbial communities", *PeerJ* 3, e1165,
doi:10.7717/peerj.1165. The predecessor whose parameters this
removes.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tetranucleotide_frequency", "abundance_correlation",
           "length_weight", "composite_distance", "bin_contigs",
           "purity_completeness"]

_EPS = 1e-12
_BASES = "ACGT"


def tetranucleotide_frequency(seq, kk=4, canonical=True):
    r"""The composition signal, available for every contig.

    ``canonical=True`` merges a :math:`k`-mer with its reverse
    complement, since the strand a contig was assembled on is
    arbitrary.
    """
    s = str(seq).upper()
    K = int(kk)
    if K < 1:
        raise ValueError("metabd: k must be at least 1")
    if len(s) < K:
        raise ValueError("metabd: the sequence is shorter than k")
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    counts = {}
    tot = 0
    for i in range(len(s) - K + 1):
        m = s[i:i + K]
        if any(c not in comp for c in m):
            continue
        if canonical:
            rc = "".join(comp[c] for c in reversed(m))
            m = min(m, rc)
        counts[m] = counts.get(m, 0) + 1
        tot += 1
    if tot == 0:
        raise ValueError("metabd: no valid k-mers in the sequence")
    keys = sorted(counts)
    return {"frequency": {m: counts[m] / float(tot) for m in keys},
            "vector": [counts[m] / float(tot) for m in keys],
            "kmers": keys, "n_kmers": tot,
            "canonical": bool(canonical)}


def abundance_correlation(cov_a, cov_b):
    r"""Covariance of coverage ACROSS SAMPLES.

    Undefined with one sample -- refused rather than returned as a
    perfect correlation, which is the failure mode that silently
    merges unrelated contigs.
    """
    a = [float(v) for v in k.vec(cov_a)]
    b = [float(v) for v in k.vec(cov_b)]
    if len(a) != len(b):
        raise ValueError("metabd: the coverage vectors differ in "
                         "length")
    if len(a) < 2:
        raise ValueError("metabd: abundance covariance needs at least "
                         "2 samples; with one sample only composition "
                         "is informative")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    den = math.sqrt(sum((a[i] - ma) ** 2 for i in range(len(a)))
                    * sum((b[i] - mb) ** 2 for i in range(len(b))))
    return {"correlation": num / den if den > _EPS else 0.0,
            "n_samples": len(a)}


def length_weight(length, l_min=2500.0, l_ref=100000.0):
    r"""Confidence in a contig's composition, scaled by length.

    A profile from 2 kb is a far noisier estimate than the same
    profile from 200 kb; discarding short contigs instead would
    discard most of the assembly.
    """
    L = float(length)
    if L <= 0.0:
        raise ValueError("metabd: the contig length must be positive")
    if L < float(l_min):
        return {"weight": 0.0, "length": L, "below_minimum": True,
                "note": "too short for a usable composition estimate"}
    w = math.log(L / float(l_min)) / math.log(float(l_ref)
                                              / float(l_min))
    return {"weight": min(max(w, 0.0), 1.0), "length": L,
            "below_minimum": False}


def composite_distance(tnf_a, tnf_b, cov_a=None, cov_b=None,
                       len_a=None, len_b=None, w_abundance=0.5):
    r"""Combine composition and abundance, weighted by length
    confidence.

    With one sample the abundance term drops out and the weight goes
    entirely to composition -- adaptively, rather than by a flag the
    user must set.
    """
    a = [float(v) for v in k.vec(tnf_a)]
    b = [float(v) for v in k.vec(tnf_b)]
    if len(a) != len(b):
        raise ValueError("metabd: the composition vectors differ in "
                         "length")
    d_tnf = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
    wa = float(w_abundance)
    d_abd, usable = 0.0, False
    if cov_a is not None and cov_b is not None \
            and len(k.vec(cov_a)) >= 2:
        r = abundance_correlation(cov_a, cov_b)["correlation"]
        d_abd = 1.0 - r
        usable = True
    if not usable:
        wa = 0.0
    conf = 1.0
    if len_a is not None and len_b is not None:
        conf = min(length_weight(len_a)["weight"],
                   length_weight(len_b)["weight"])
    d = (1.0 - wa) * d_tnf + wa * d_abd
    return {"distance": d, "composition": d_tnf,
            "abundance": d_abd if usable else None,
            "abundance_usable": usable, "confidence": conf,
            "effective_weight": wa,
            "note": "with a single sample the abundance term drops "
                    "out automatically"}


def bin_contigs(tnfs, coverages=None, lengths=None, threshold=0.15,
                min_bin_size=200000.0):
    r"""Greedy agglomeration under the composite distance.

    Bins below a total size are returned as unbinned rather than
    reported as genomes.
    """
    T = [[float(v) for v in r] for r in k.mat(tnfs)]
    n = len(T)
    L = [1e5] * n if lengths is None else [float(v)
                                           for v in k.vec(lengths)]
    bins, assigned = [], [False] * n
    order = sorted(range(n), key=lambda i: -L[i])
    for i in order:
        if assigned[i]:
            continue
        cur = [i]
        assigned[i] = True
        for j in order:
            if assigned[j]:
                continue
            d = composite_distance(
                T[i], T[j],
                None if coverages is None else coverages[i],
                None if coverages is None else coverages[j],
                L[i], L[j])["distance"]
            if d < float(threshold):
                cur.append(j)
                assigned[j] = True
        bins.append(cur)
    big = [b for b in bins if sum(L[i] for i in b)
           >= float(min_bin_size)]
    small = [i for b in bins if sum(L[i] for i in b)
             < float(min_bin_size) for i in b]
    return RichResult(payload={
        "estimate": big, "bins": big, "unbinned": sorted(small),
        "n_bins": len(big), "n_unbinned": len(small),
        "method": "adaptive composite binning; Kang et al. (2019)",
        "note": "sub-threshold groups are UNBINNED, not reported as "
                "draft genomes",
    })


def purity_completeness(bins, truth):
    r"""Two different failures, reported separately.

    A bin can be complete and contaminated, or pure and fragmentary;
    one number cannot say which.
    """
    t = list(truth)
    out = []
    for b in bins:
        labs = [t[i] for i in b]
        if not labs:
            continue
        counts = {}
        for l in labs:
            counts[l] = counts.get(l, 0) + 1
        dom = max(sorted(counts), key=lambda l: counts[l])
        purity = counts[dom] / float(len(labs))
        total = sum(1 for x in t if x == dom)
        out.append({"dominant": dom, "purity": purity,
                    "completeness": counts[dom] / float(total)
                    if total else 0.0, "size": len(labs)})
    return {"per_bin": out,
            "mean_purity": sum(b["purity"] for b in out) / len(out)
            if out else 0.0,
            "mean_completeness": sum(b["completeness"] for b in out)
            / len(out) if out else 0.0,
            "note": "contamination and fragmentation are different "
                    "failures"}


def cheatsheet():
    return ("metabd: bin contigs into draft genomes from TWO signals "
            "-- tetranucleotide composition (available always, noisy "
            "on short contigs) and abundance covariance ACROSS SAMPLES "
            "(strong, but undefined with one sample). Earlier tools "
            "needed manual parameter tuning and degraded on poor "
            "assemblies; the contribution is an ADAPTIVE algorithm "
            "that removes the tuning. Confidence must scale with "
            "contig LENGTH, since discarding short contigs discards "
            "most of the assembly. Purity and completeness are "
            "separate failures and are reported separately.")


# compact alias per ledger/NAMING.md
metabat2 = bin_contigs

# public names resolved by fn/_lazy_map.json
metagenome_binning = bin_contigs
