r"""GWAS quality control: the seven standard steps.

Marees, A. T., de Kluiver, H., Stringer, S., Vorspan, F., Curis, E., Marie-
Claire, C., & Derks, E. M. (2018) "A tutorial on conducting genome-wide
association studies: Quality control and statistical analysis",
*International Journal of Methods in Psychiatric Research* 27(2), e1608.

The tutorial's Table 1 lists seven QC steps with the thresholds it
recommends, and those thresholds are implemented here as the defaults, each
one traceable to the sentence that gives it:

1. **Missingness**, SNPs then individuals. "We recommend to first filter
   SNPs and individuals based on a relaxed threshold (0.2; >20%) ... Then a
   filter with a more stringent threshold can be applied (0.02). Note, SNP
   filtering should be performed before individual filtering." Both passes
   are run in that order, and the order is not cosmetic -- see the anchor.
2. **Sex discrepancy** from X-chromosome homozygosity: "Males should have an
   X chromosome homozygosity estimate >0.8 and females should have a value
   <0.2."
3. **Minor allele frequency**: "for large (N = 100.000) vs. moderate samples
   (N = 10000), 0.01 and 0.05 are commonly used as MAF threshold."
4. **Hardy-Weinberg equilibrium**: "For binary traits we suggest to exclude:
   HWE p value <1e-10 in cases and <1e-6 in controls. Less strict case
   threshold avoids discarding disease-associated SNPs under selection. For
   quantitative traits, we recommend HWE p value <1e-6."
5. **Heterozygosity**: "removing individuals who deviate ±3 SD from the
   samples' heterozygosity rate mean."
6. **Relatedness**: "we suggest to use a pi-hat threshold of 0.2", using
   "independent SNPs (pruning) ... and limit it to autosomal chromosomes
   only".
7. **Population stratification** by MDS on the IBS matrix, "typically 10"
   dimensions.

Two arithmetic notes, because the tutorial specifies tools rather than
formulas for them. The HWE p-value is computed here either by the standard
chi-square goodness-of-fit test against Hardy-Weinberg expectations, or by
the exact conditional test -- the probability of the observed heterozygote
count given the allele counts, summed over tables no more probable than the
observed one -- which is what genotype QC normally uses because the
chi-square approximation fails on rare variants; both are offered
(``hwe_test``) and neither is attributed to the tutorial.

**Relatedness has two routes, and the tutorial's own statistic is the
default.** ``relatedness="pihat"`` is PLINK's method-of-moments IBD
estimator, from the tool the tutorial actually invokes:

    Purcell, S., Neale, B., Todd-Brown, K., Thomas, L., Ferreira, M. A. R.,
    Bender, D., Maller, J., Sklar, P., de Bakker, P. I. W., Daly, M. J., &
    Sham, P. C. (2007) "PLINK: A Tool Set for Whole-Genome Association and
    Population-Based Linkage Analyses", *American Journal of Human
    Genetics* 81(3), 559-575.

Its equations are printed there: conditional on an IBD state :math:`Z`, the
expected count of SNPs at IBS state :math:`I` is
:math:`N(I{=}i \mid Z{=}z) = \sum_m P(I{=}i \mid Z{=}z)`, and the moments
are inverted in order,

.. math::

   P(Z{=}0) &= rac{N(I{=}0)}{N(I{=}0 \mid Z{=}0)} \
   P(Z{=}1) &= rac{N(I{=}1) - P(Z{=}0)N(I{=}1 \mid Z{=}0)}
                    {N(I{=}1 \mid Z{=}1)} \
   P(Z{=}2) &= rac{N(I{=}2) - P(Z{=}0)N(I{=}2 \mid Z{=}0)
                     - P(Z{=}1)N(I{=}2 \mid Z{=}1)}{N(I{=}2 \mid Z{=}2)},

with :math:`\hat\pi = P(Z{=}2) + 	frac12 P(Z{=}1)`. The paper is explicit
that "these estimates of P(Z) are not bounded :math:`0 \le x \le 1` and are
also not constrained to biologically plausible values", and gives the
bounding rules, which are implemented as printed: if :math:`P(Z{=}0) > 1` it
is set to 1 and the others to 0; if :math:`P(Z{=}0) < 0` it is set to 0 and
:math:`P(Z{=}1), P(Z{=}2)` are rescaled by their sum.

The :math:`P(I \mid Z)` table carries PLINK's ascertainment correction, "where
:math:`T_A` is the total number of nonmissing alleles and :math:`X` and
:math:`Y` are the number of :math:`A` and :math:`a` alleles, respectively, so
that :math:`p = X/T_A` and :math:`q = Y/T_A`" -- factors of
:math:`(X-1)/X` and :math:`T_A/(T_A-1)` and so on, which correct for
estimating the frequencies from the same sample. ``correction=False`` drops
them for the textbook large-sample forms, and the anchor shows the two agree
to three decimals at 4000 alleles and visibly disagree at 40.

``relatedness="kinship"`` is the other route: the genomic kinship from
centred, scaled genotypes, which lands on the same scale (about 1 for a
duplicate pair, 0.5 for full sibs, 0.25 at second degree) but is a different
estimator, not an IBD decomposition. Both are reported; the 0.2 cutoff is
the tutorial's either way.

Genotypes are counts of the minor allele, 0/1/2, with ``None`` for a missing
call.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["snpqc1", "snp_quality_control", "snp_qc", "call_rates", "maf",
           "hwe_pvalue", "heterozygosity", "kinship_matrix", "ld_prune",
           "sex_check", "ibd_moments", "pihat_matrix", "ibs_given_ibd"]


def _check(genotypes):
    G = [list(row) for row in genotypes]
    if not G or not G[0]:
        raise ValueError("snpqc1: genotypes must be a non-empty "
                         "individual x SNP matrix")
    m = len(G[0])
    for row in G:
        if len(row) != m:
            raise ValueError("snpqc1: ragged genotype matrix")
        for g in row:
            if g is not None and g not in (0, 1, 2, 0.0, 1.0, 2.0):
                raise ValueError("snpqc1: genotypes must be 0, 1, 2 or None")
    return G, len(G), m


def call_rates(genotypes):
    """Per-SNP and per-individual call rates."""
    G, n, m = _check(genotypes)
    per_snp = [sum(1 for i in range(n) if G[i][j] is not None) / float(n)
               for j in range(m)]
    per_ind = [sum(1 for j in range(m) if G[i][j] is not None) / float(m)
               for i in range(n)]
    return per_snp, per_ind


def maf(genotypes):
    """Minor allele frequency per SNP, over non-missing calls."""
    G, n, m = _check(genotypes)
    out = []
    for j in range(m):
        called = [G[i][j] for i in range(n) if G[i][j] is not None]
        if not called:
            out.append(0.0)
            continue
        p = sum(called) / (2.0 * len(called))
        out.append(min(p, 1.0 - p))
    return out


def _log_fact(n, cache={0: 0.0}):
    if n in cache:
        return cache[n]
    v = math.lgamma(n + 1.0)
    cache[n] = v
    return v


def hwe_pvalue(n_hom_minor, n_het, n_hom_major, test="exact"):
    r"""Hardy-Weinberg p-value for one SNP.

    ``"exact"`` is the conditional test: with the minor allele count
    :math:`n_A` fixed, the probability of a table with :math:`n_{AB}`
    heterozygotes is

    .. math:: \Pr(n_{AB} \mid n_A) \propto
              \frac{n!}{n_{AA}!\,n_{AB}!\,n_{BB}!}\;2^{n_{AB}},

    and the p-value sums the probabilities of every table no more probable
    than the observed one. ``"chisq"`` is the goodness-of-fit test against
    :math:`p^2, 2pq, q^2` with one degree of freedom.
    """
    a, h, b = int(n_hom_minor), int(n_het), int(n_hom_major)
    if min(a, h, b) < 0:
        raise ValueError("snpqc1: genotype counts must be non-negative")
    n = a + h + b
    if n == 0:
        raise ValueError("snpqc1: no genotypes")
    if test not in ("exact", "chisq"):
        raise ValueError("snpqc1: test must be 'exact' or 'chisq'")
    n_minor = 2 * a + h
    if test == "chisq":
        p = n_minor / (2.0 * n)
        exp = [n * p * p, 2.0 * n * p * (1 - p), n * (1 - p) ** 2]
        obs = [a, h, b]
        if min(exp) <= 0:
            return 1.0
        chi = sum((obs[k] - exp[k]) ** 2 / exp[k] for k in range(3))
        return math.erfc(math.sqrt(chi / 2.0))
    # exact conditional test
    n_major = 2 * n - n_minor
    probs = {}
    lognorm = None
    for het in range(n_minor % 2, min(n_minor, n_major) + 1, 2):
        hom_a = (n_minor - het) // 2
        hom_b = (n_major - het) // 2
        lp = (_log_fact(n) - _log_fact(hom_a) - _log_fact(het) -
              _log_fact(hom_b) + het * math.log(2.0))
        probs[het] = lp
        lognorm = lp if lognorm is None else max(lognorm, lp)
    tot = sum(math.exp(v - lognorm) for v in probs.values())
    obs_lp = probs.get(h)
    if obs_lp is None:
        raise ValueError("snpqc1: the observed heterozygote count is "
                         "impossible given the allele counts")
    thresh = math.exp(obs_lp - lognorm) * (1.0 + 1e-9)
    p = sum(math.exp(v - lognorm) for v in probs.values()
            if math.exp(v - lognorm) <= thresh) / tot
    return min(max(p, 0.0), 1.0)


def heterozygosity(genotypes):
    """Per-individual heterozygosity rate over non-missing calls."""
    G, n, m = _check(genotypes)
    out = []
    for i in range(n):
        called = [g for g in G[i] if g is not None]
        out.append(sum(1 for g in called if g == 1) / float(len(called))
                   if called else 0.0)
    return out


def sex_check(x_genotypes, reported_sex=None, male_min=0.8, female_max=0.2):
    r"""X-chromosome homozygosity :math:`F` and the tutorial's cutoffs.

    :math:`F = (O - E)/(n - E)` with :math:`O` the observed homozygote
    count and :math:`E` its Hardy-Weinberg expectation. Males should exceed
    ``male_min`` (0.8), females fall below ``female_max`` (0.2).
    ``reported_sex`` (1 male, 2 female, as in a PLINK fam file) turns the
    result into a discrepancy list.
    """
    G, n, m = _check(x_genotypes)
    freqs = maf(G)
    out = []
    for i in range(n):
        obs_hom = exp_hom = 0.0
        for j in range(m):
            g = G[i][j]
            if g is None:
                continue
            p = freqs[j]
            obs_hom += 1.0 if g != 1 else 0.0
            exp_hom += 1.0 - 2.0 * p * (1.0 - p)
        denom = sum(1 for j in range(m) if G[i][j] is not None) - exp_hom
        out.append((obs_hom - exp_hom) / denom if abs(denom) > 1e-12
                   else float("nan"))
    called = [1 if f > male_min else (2 if f < female_max else 0)
              for f in out]
    res = {"F": out, "inferred_sex": called}
    if reported_sex is not None:
        rep = [int(v) for v in reported_sex]
        if len(rep) != n:
            raise ValueError("snpqc1: one reported sex per individual")
        res["discrepant"] = [i for i in range(n)
                             if called[i] != 0 and called[i] != rep[i]]
        res["undetermined"] = [i for i in range(n) if called[i] == 0]
    return res


def ibs_given_ibd(x_count, y_count, correction=True):
    r"""PLINK's Table 1: :math:`P(I \mid Z)` for one SNP.

    ``x_count`` and ``y_count`` are the counts of the two alleles among
    non-missing calls, so :math:`T_A = X + Y`, :math:`p = X/T_A`,
    :math:`q = Y/T_A`. With ``correction`` the ascertainment factors of the
    paper's note are applied; without it the textbook large-sample forms
    are used.

    Returns ``[[P(I=0|Z=0), P(I=1|Z=0), P(I=2|Z=0)],
    [0, P(I=1|Z=1), P(I=2|Z=1)], [0, 0, 1]]``.
    """
    X = float(x_count)
    Y = float(y_count)
    T = X + Y
    if T <= 0:
        raise ValueError("snpqc1: a SNP with no non-missing alleles")
    p = X / T
    q = Y / T
    if not correction or T < 5 or X < 4 or Y < 4:
        # textbook forms; also the fallback when the corrected factors
        # would divide by a count too small to support them
        z0 = [2 * p * p * q * q,
              4 * p ** 3 * q + 4 * p * q ** 3,
              p ** 4 + q ** 4 + 4 * p * p * q * q]
        z1 = [0.0, 2 * p * q, 1.0 - 2 * p * q]
        return [z0, z1, [0.0, 0.0, 1.0]]

    t1, t2, t3 = T / (T - 1.0), T / (T - 2.0), T / (T - 3.0)
    xa, xb, xc = (X - 1.0) / X, (X - 2.0) / X, (X - 3.0) / X
    ya, yb, yc = (Y - 1.0) / Y, (Y - 2.0) / Y, (Y - 3.0) / Y

    i0z0 = 2 * p * p * q * q * xa * ya * t1 * t2 * t3
    i1z0 = (4 * p ** 3 * q * xa * xb * t1 * t2 * t3 +
            4 * p * q ** 3 * ya * yb * t1 * t2 * t3)
    i2z0 = (p ** 4 * xa * xb * xc * t1 * t2 * t3 +
            q ** 4 * ya * yb * yc * t1 * t2 * t3 +
            4 * p * p * q * q * xa * ya * t1 * t2 * t3)
    i1z1 = (2 * p * p * q * xa * t1 * t2 + 2 * p * q * q * ya * t1 * t2)
    i2z1 = (p ** 3 * xa * xb * t1 * t2 + q ** 3 * ya * yb * t1 * t2 +
            p * p * q * xa * t1 * t2 + p * q * q * ya * t1 * t2)
    return [[i0z0, i1z0, i2z0], [0.0, i1z1, i2z1], [0.0, 0.0, 1.0]]


def ibd_moments(genotypes, correction=True):
    r"""PLINK's method-of-moments IBD estimates for every pair.

    Returns ``(Z, pihat)`` where ``Z[i][k]`` is
    ``(P(Z=0), P(Z=1), P(Z=2))`` after the paper's bounding rules and
    ``pihat[i][k] = P(Z=2) + P(Z=1)/2``.
    """
    G, n, m = _check(genotypes)
    tables = []
    for j in range(m):
        called = [G[i][j] for i in range(n) if G[i][j] is not None]
        if not called:
            tables.append(None)
            continue
        X = sum(called)                       # copies of the minor allele
        Y = 2 * len(called) - X
        if X <= 0 or Y <= 0:
            tables.append(None)               # monomorphic: uninformative
            continue
        tables.append(ibs_given_ibd(X, Y, correction))

    Z = [[None] * n for _ in range(n)]
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Z[i][i] = (0.0, 0.0, 1.0)
        P[i][i] = 1.0
        for k in range(i + 1, n):
            obs = [0.0, 0.0, 0.0]
            exp = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            for j in range(m):
                if tables[j] is None:
                    continue
                gi, gk = G[i][j], G[k][j]
                if gi is None or gk is None:
                    continue
                ibs = 2 - int(abs(gi - gk))
                obs[ibs] += 1.0
                for z in range(3):
                    for t in range(3):
                        exp[z][t] += tables[j][z][t]
            if exp[0][0] <= 0:
                Z[i][k] = Z[k][i] = (1.0, 0.0, 0.0)
                continue
            z0 = obs[0] / exp[0][0]
            z1 = ((obs[1] - z0 * exp[0][1]) / exp[1][1]
                  if exp[1][1] > 0 else 0.0)
            z2 = ((obs[2] - z0 * exp[0][2] - z1 * exp[1][2]) / exp[2][2]
                  if exp[2][2] > 0 else 0.0)
            # the paper's bounding rules, as printed
            if z0 > 1.0:
                z0, z1, z2 = 1.0, 0.0, 0.0
            elif z0 < 0.0:
                z0 = 0.0
                s = z1 + z2
                if s > 0:
                    z1, z2 = z1 / s, z2 / s
                else:
                    z1, z2 = 0.0, 1.0
            z1 = max(z1, 0.0)
            z2 = max(z2, 0.0)
            tot = z0 + z1 + z2
            if tot > 0:
                z0, z1, z2 = z0 / tot, z1 / tot, z2 / tot
            Z[i][k] = Z[k][i] = (z0, z1, z2)
            P[i][k] = P[k][i] = z2 + 0.5 * z1
    return Z, P


def pihat_matrix(genotypes, correction=True):
    r"""Just the :math:`\hat\pi = P(Z{=}2) + \tfrac12 P(Z{=}1)` matrix."""
    return ibd_moments(genotypes, correction)[1]


def kinship_matrix(genotypes):
    r"""Genomic kinship from centred, scaled genotypes.

    :math:`K_{ik} = \frac{1}{M}\sum_j \frac{(g_{ij} - 2p_j)
    (g_{kj} - 2p_j)}{2 p_j (1 - p_j)}`, which estimates twice the kinship
    coefficient and therefore sits on the same scale as the tutorial's
    pi-hat: about 1 for a duplicate or monozygotic pair, 0.5 for full sibs
    or parent-offspring, 0.25 at second degree -- which is where the
    tutorial's 0.2 cutoff bites. It is NOT PLINK's pi-hat, which is a
    method-of-moments IBD estimate from a different set of statistics; the
    difference is stated rather than hidden.
    """
    G, n, m = _check(genotypes)
    freqs = []
    for j in range(m):
        called = [G[i][j] for i in range(n) if G[i][j] is not None]
        freqs.append(sum(called) / (2.0 * len(called)) if called else 0.0)
    use = [j for j in range(m) if 1e-6 < freqs[j] < 1 - 1e-6]
    if not use:
        raise ValueError("snpqc1: no polymorphic SNP for kinship")
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            tot = 0.0
            cnt = 0
            for j in use:
                gi, gk = G[i][j], G[k][j]
                if gi is None or gk is None:
                    continue
                p = freqs[j]
                tot += ((gi - 2 * p) * (gk - 2 * p) /
                        (2.0 * p * (1.0 - p)))
                cnt += 1
            v = tot / cnt if cnt else 0.0
            K[i][k] = K[k][i] = v
    return K


def ld_prune(genotypes, window=50, step=5, r2=0.2):
    r"""Window-based pruning: drop one of any pair with :math:`r^2` above
    the threshold, "based on a user-specified threshold of LD ... pruning
    does not take the p value of a SNP into account"."""
    G, n, m = _check(genotypes)
    keep = list(range(m))

    def corr2(j, k):
        pairs = [(G[i][j], G[i][k]) for i in range(n)
                 if G[i][j] is not None and G[i][k] is not None]
        if len(pairs) < 3:
            return 0.0
        mj = sum(a for a, _ in pairs) / len(pairs)
        mk = sum(b for _, b in pairs) / len(pairs)
        sj = sum((a - mj) ** 2 for a, _ in pairs)
        sk = sum((b - mk) ** 2 for _, b in pairs)
        if sj <= 0 or sk <= 0:
            return 0.0
        c = sum((a - mj) * (b - mk) for a, b in pairs)
        return c * c / (sj * sk)

    start = 0
    while start < len(keep):
        block = keep[start:start + window]
        drop = set()
        for a in range(len(block)):
            if block[a] in drop:
                continue
            for b in range(a + 1, len(block)):
                if block[b] in drop:
                    continue
                if corr2(block[a], block[b]) > r2:
                    drop.add(block[b])
        keep = [j for j in keep if j not in drop]
        start += step
    return keep


def snpqc1(genotypes, phenotype=None, trait="binary", geno_relaxed=0.2,
           mind_relaxed=0.2, geno=0.02, mind=0.02, maf_threshold=0.01,
           hwe_case=1e-10, hwe_control=1e-6, hwe_quantitative=1e-6,
           het_sd=3.0, pihat=0.2, hwe_test="exact", x_genotypes=None,
           reported_sex=None, relatedness="pihat", ibd_correction=True):
    r"""Run the tutorial's QC steps and report what each one removes.

    Defaults are the tutorial's own thresholds. ``trait`` selects the HWE
    rule: ``"binary"`` applies :math:`10^{-10}` in cases and
    :math:`10^{-6}` in controls (the looser case threshold "avoids
    discarding disease-associated SNPs under selection"), ``"quantitative"``
    applies :math:`10^{-6}` throughout.

    Returns
    -------
    RichResult
        ``estimate`` / ``keep_snps`` and ``keep_individuals`` are the
        surviving indices; ``removed`` breaks the exclusions down by step;
        ``call_rate_snp``, ``call_rate_ind``, ``maf``, ``hwe_p``,
        ``heterozygosity``, ``kinship``, ``thresholds`` report the
        quantities each decision used.

    Examples
    --------
    ::

        r = snpqc1(genotypes, phenotype)
        len(r["keep_snps"]), r["removed"]["maf"]

    References
    ----------
    Marees et al. (2018) *Int J Methods Psychiatr Res* 27(2), e1608,
    Table 1 and the QC section.
    """
    G, n, m = _check(genotypes)
    if trait not in ("binary", "quantitative"):
        raise ValueError("snpqc1: trait must be 'binary' or 'quantitative'")
    for name, v in (("geno", geno), ("mind", mind),
                    ("geno_relaxed", geno_relaxed),
                    ("mind_relaxed", mind_relaxed)):
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("snpqc1: %s must lie in [0, 1]" % name)
    if not 0.0 <= float(maf_threshold) < 0.5:
        raise ValueError("snpqc1: maf_threshold must lie in [0, 0.5)")

    snps = list(range(m))
    inds = list(range(n))
    removed = {"geno_relaxed": [], "mind_relaxed": [], "geno": [],
               "mind": [], "maf": [], "hwe": [], "heterozygosity": [],
               "relatedness": [], "sex": []}

    def sub():
        return [[G[i][j] for j in snps] for i in inds]

    # step 1, relaxed then stringent, SNPs before individuals each time
    for gthr, mthr, gkey, mkey in ((geno_relaxed, mind_relaxed,
                                    "geno_relaxed", "mind_relaxed"),
                                   (geno, mind, "geno", "mind")):
        cr_snp, _ = call_rates(sub())
        drop = [snps[t] for t in range(len(snps))
                if 1.0 - cr_snp[t] > gthr]
        removed[gkey] = drop
        snps = [j for j in snps if j not in set(drop)]
        if not snps:
            break
        _, cr_ind = call_rates(sub())
        dropi = [inds[t] for t in range(len(inds))
                 if 1.0 - cr_ind[t] > mthr]
        removed[mkey] = dropi
        inds = [i for i in inds if i not in set(dropi)]
        if not inds:
            break
    if not snps or not inds:
        raise ValueError("snpqc1: missingness filtering removed everything")

    # step 2, sex discrepancy
    if x_genotypes is not None:
        sx = sex_check([[x_genotypes[i][j]
                         for j in range(len(x_genotypes[0]))]
                        for i in inds], None if reported_sex is None
                       else [reported_sex[i] for i in inds])
        if reported_sex is not None:
            bad = [inds[t] for t in sx["discrepant"]]
            removed["sex"] = bad
            inds = [i for i in inds if i not in set(bad)]

    # step 3, MAF
    freqs = maf(sub())
    drop = [snps[t] for t in range(len(snps)) if freqs[t] < maf_threshold]
    removed["maf"] = drop
    snps = [j for j in snps if j not in set(drop)]
    if not snps:
        raise ValueError("snpqc1: the MAF filter removed every SNP")

    # step 4, HWE
    pheno = None if phenotype is None else [phenotype[i] for i in inds]
    hwe_p = []
    drop = []
    for t, j in enumerate(snps):
        def counts(rows):
            a = h = b = 0
            for i in rows:
                g = G[i][j]
                if g is None:
                    continue
                if g == 1:
                    h += 1
                elif g == 2:
                    a += 1
                else:
                    b += 1
            return a, h, b
        if trait == "quantitative" or pheno is None:
            a, h, b = counts(inds)
            p = hwe_pvalue(a, h, b, hwe_test)
            hwe_p.append(p)
            if p < hwe_quantitative:
                drop.append(j)
        else:
            cases = [i for k, i in enumerate(inds) if pheno[k] == 1]
            ctrls = [i for k, i in enumerate(inds) if pheno[k] == 0]
            pc = hwe_pvalue(*counts(cases), test=hwe_test) if cases else 1.0
            pk = hwe_pvalue(*counts(ctrls), test=hwe_test) if ctrls else 1.0
            hwe_p.append(min(pc, pk))
            if pc < hwe_case or pk < hwe_control:
                drop.append(j)
    removed["hwe"] = drop
    snps = [j for j in snps if j not in set(drop)]
    if not snps:
        raise ValueError("snpqc1: the HWE filter removed every SNP")

    # step 5, heterozygosity, +- het_sd SD from the mean
    het = heterozygosity(sub())
    mean = sum(het) / len(het)
    var = (sum((v - mean) ** 2 for v in het) / max(1, len(het) - 1))
    sd = math.sqrt(var)
    drop = [inds[t] for t in range(len(inds))
            if sd > 0 and abs(het[t] - mean) > het_sd * sd]
    removed["heterozygosity"] = drop
    inds = [i for i in inds if i not in set(drop)]

    # step 6, relatedness on pruned SNPs
    if relatedness not in ("pihat", "kinship"):
        raise ValueError("snpqc1: relatedness must be 'pihat' (PLINK's "
                         "method-of-moments IBD) or 'kinship'")
    pruned = ld_prune(sub())
    pruned_geno = [[G[i][snps[t]] for t in pruned] for i in inds]
    if relatedness == "pihat":
        Zstates, K = ibd_moments(pruned_geno, ibd_correction)
    else:
        Zstates, K = None, kinship_matrix(pruned_geno)
    drop = []
    for a in range(len(inds)):
        for b in range(a + 1, len(inds)):
            if K[a][b] > pihat and inds[b] not in drop:
                drop.append(inds[b])
    removed["relatedness"] = drop
    inds = [i for i in inds if i not in set(drop)]

    return RichResult(payload={
        "estimate": snps,
        "keep_snps": snps,
        "keep_individuals": inds,
        "removed": removed,
        "n_snps_kept": len(snps),
        "n_individuals_kept": len(inds),
        "call_rate_snp": call_rates(genotypes)[0],
        "call_rate_ind": call_rates(genotypes)[1],
        "maf": freqs,
        "hwe_p": hwe_p,
        "heterozygosity": het,
        "relatedness_matrix": K,
        "kinship": K,
        "ibd_states": Zstates,
        "relatedness": relatedness,
        "pruned_snps": [snps[t] for t in pruned],
        "thresholds": {"geno_relaxed": geno_relaxed,
                       "mind_relaxed": mind_relaxed, "geno": geno,
                       "mind": mind, "maf": maf_threshold,
                       "hwe_case": hwe_case, "hwe_control": hwe_control,
                       "hwe_quantitative": hwe_quantitative,
                       "het_sd": het_sd, "pihat": pihat},
        "trait": trait,
        "hwe_test": hwe_test,
        "note": ("relatedness by PLINK's method-of-moments IBD (Purcell "
                 "et al. 2007), pi-hat = P(Z=2) + P(Z=1)/2"
                 if relatedness == "pihat" else
                 "relatedness by genomic kinship, NOT PLINK's pi-hat; "
                 "pass relatedness='pihat' for the IBD estimator") +
                "; the 0.2 cutoff is the tutorial's",
        "method": "GWAS quality control (Marees et al. 2018, Table 1)",
    })


def cheatsheet():
    return ("snpqc1: GWAS QC (Marees et al. 2018, Table 1). Seven steps "
            "with the tutorial's own thresholds: missingness in TWO passes "
            "(0.2 relaxed, then 0.02) and SNPs BEFORE individuals each "
            "time; sex check on X homozygosity (>0.8 male, <0.2 female); "
            "MAF 0.01 large samples / 0.05 moderate; HWE 1e-10 in cases "
            "and 1e-6 in controls for binary traits, 1e-6 for "
            "quantitative; heterozygosity +-3 SD from the mean; "
            "relatedness above 0.2 after LD pruning. HWE by exact "
            "conditional test or chi-square. Relatedness has TWO routes: "
            "PLINK's method-of-moments IBD (Purcell 2007) giving "
            "pi-hat = P(Z=2) + P(Z=1)/2 with the paper's bounding rules "
            "and ascertainment correction, which is the default and the "
            "statistic the 0.2 cutoff was written for, or a genomic "
            "kinship on the same scale.")


# compact aliases
snp_quality_control = snpqc1
snp_qc = snpqc1
