# morie.fn -- function file (rootcoder007/morie)
r"""edgeR: negative binomial counts, moderated dispersions, TMM.

Microarray abundance is a fluorescence intensity -- effectively
continuous. Digital gene expression is a **count**, so procedures that
work for microarrays do not transfer. edgeR models

.. math:: Y_{gi} \sim \mathrm{NB}(M_i p_{gj}, \phi_g),

with :math:`M_i` the library size, :math:`p_{gj}` the relative
abundance of gene :math:`g` in group :math:`j`, mean
:math:`\mu_{gi} = M_i p_{gj}` and variance
:math:`\mu_{gi}(1 + \mu_{gi}\phi_g)`. At :math:`\phi_g = 0` this is
Poisson, which is technical variation alone; :math:`\phi_g` is the
squared coefficient of biological variation, so the model **separates
biological from technical variation** rather than assuming the latter
is all there is.

**Dispersions are moderated, not estimated gene by gene.** With a
handful of libraries there are almost no degrees of freedom per gene.
Following limma's logic for probe-wise variances, edgeR shrinks each
:math:`\phi_g` toward a common value by an empirical Bayes procedure --
mathematically more complex, same idea.

**Normalisation is a separate problem from library size, and this is
where analyses go wrong.** Total RNA production cannot be estimated,
but the *ratio* between two samples can. TMM assumes most genes are
not differentially expressed and takes a **doubly trimmed, weighted
mean of the log ratios**: trim :math:`M_g` by 30% and :math:`A_g` by
5%, then average what remains with weights equal to the inverse
delta-method variances. The liver-versus-kidney example is the point:
without it 77% of DE genes come out higher in kidney; with a factor of
0.68 the split is 47/53.

Crucially the **counts are not modified** -- the factor enters the
model as an offset, so the sampling properties of the data survive.

**Two tests, both here.** The classic NB exact/likelihood-ratio test,
and the **quasi-likelihood F-test**, which compares
:math:`\mathrm{LRT}_k/(q\hat\Phi_k)` to an :math:`F` distribution.
The QL route is more conservative and, because the denominator
degrees of freedom are finite, it **propagates the uncertainty in the
estimated dispersion** into the test -- which the LRT does not.

References
----------
Robinson, M. D., McCarthy, D. J. & Smyth, G. K. (2010) "edgeR: a
Bioconductor package for differential expression analysis of digital
gene expression data", *Bioinformatics* 26(1), 139-140,
doi:10.1093/bioinformatics/btp616. [PDF supplied by Vee.] Sec. 1-2:
that microarray abundance is a continuous fluorescence intensity while
DGE abundance is a count, so microarray procedures are not directly
applicable; the model Y_gi ~ NB(M_i p_gj, phi_g) with mean mu_gi =
M_i p_gj and variance mu_gi(1 + mu_gi phi_g); that the NB reduces to
Poisson at phi_g = 0 and that phi_g represents the coefficient of
biological variation, so the model separates biological from technical
variation; and the empirical Bayes moderation of the overdispersion
across genes, analogous to limma's moderated variances.

Robinson, M. D. & Oshlack, A. (2010) "A scaling normalization method
for differential expression analysis of RNA-seq data", *Genome
Biology* 11(3), R25, doi:10.1186/gb-2010-11-3-r25. [PDF supplied by
Vee.] The TMM method: that total RNA production S_k cannot be
estimated directly but the ratio f_k = S_k/S_k' can, under the
assumption that the majority of genes are not DE; the doubly trimmed
weighted mean of the log expression ratios M_g, trimmed by 30% on M
and 5% on A by default, with weights the inverse approximate
asymptotic variances from the delta method; robustness up to about 30%
DE in one direction; the liver-versus-kidney factor of 0.68 (-0.56 on
the log2 scale) changing the DE split from 77% higher in kidney to
47%/53%; and that the DATA are not modified -- the factors enter the
statistical model as an offset, preserving the sampling properties.

Lund, S. P., Nettleton, D., McCarthy, D. J. & Smyth, G. K. (2012)
"Detecting differential expression in RNA-sequence data using
quasi-likelihood with shrunken dispersion estimates", *Statistical
Applications in Genetics and Molecular Biology* 11(5), Article 8,
doi:10.1515/1544-6115.1826. [PDF supplied by Vee.] The
quasi-likelihood approach with shrunken dispersion estimates adapting
Smyth's (2004) treatment of gene-specific error variances; the QL test
comparing LRT_k/(q Phi_hat_k) to an F distribution, with q the
difference in dimension between the full and null parameter spaces;
and the statement that the QL methods incorporate the uncertainty in
the estimated variances when testing, unlike approaches that treat the
dispersion as known.

Smyth, G. K. (2004) "Linear models and empirical Bayes methods for
assessing differential expression in microarray experiments",
*Statistical Applications in Genetics and Molecular Biology* 3(1),
Article 3, doi:10.2202/1544-6115.1027. The moderation being adapted.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmm_factor", "nb_variance", "moderate_dispersion",
           "exact_test", "ql_f_test", "effective_library_size"]

_EPS = 1e-12


def nb_variance(mu, dispersion):
    r""":math:`\mu(1+\mu\phi)`. Poisson exactly at :math:`\phi=0`."""
    m, p = float(mu), float(dispersion)
    if m < 0.0 or p < 0.0:
        raise ValueError("edgrn: the mean and dispersion must be "
                         "non-negative")
    return {"variance": m * (1.0 + m * p), "poisson": m,
            "biological": m * m * p, "bcv": math.sqrt(p),
            "note": "phi = 0 leaves the Poisson variance exactly; "
                    "sqrt(phi) is the biological coefficient of "
                    "variation"}


def tmm_factor(counts_sample, counts_reference, trim_m=0.3,
               trim_a=0.05, lib_sample=None, lib_reference=None):
    r"""The trimmed mean of M-values.

    Doubly trimmed -- by log-fold-change :math:`M_g` and by absolute
    intensity :math:`A_g` -- then averaged with inverse delta-method
    variances as weights.
    """
    y = [float(v) for v in k.vec(counts_sample)]
    r = [float(v) for v in k.vec(counts_reference)]
    if len(y) != len(r):
        raise ValueError("edgrn: %d genes in the sample but %d in "
                         "the reference" % (len(y), len(r)))
    Nk = float(lib_sample) if lib_sample is not None else sum(y)
    Nr = float(lib_reference) if lib_reference is not None else sum(r)
    if Nk <= 0.0 or Nr <= 0.0:
        raise ValueError("edgrn: a library size is zero")
    rows = []
    for g in range(len(y)):
        if y[g] <= 0.0 or r[g] <= 0.0:
            continue
        M = math.log(y[g] / Nk, 2) - math.log(r[g] / Nr, 2)
        A = 0.5 * (math.log(y[g] / Nk, 2) + math.log(r[g] / Nr, 2))
        w = ((Nk - y[g]) / (Nk * y[g])) + ((Nr - r[g]) / (Nr * r[g]))
        rows.append((M, A, 1.0 / w if w > _EPS else 0.0))
    if not rows:
        raise ValueError("edgrn: no gene is positive in both "
                         "libraries, so no ratio can be formed")
    n = len(rows)
    tm, ta = float(trim_m), float(trim_a)
    if not 0.0 <= tm < 0.5 or not 0.0 <= ta < 0.5:
        raise ValueError("edgrn: the trim fractions must lie in "
                         "[0, 0.5)")
    by_m = sorted(range(n), key=lambda i: rows[i][0])
    by_a = sorted(range(n), key=lambda i: rows[i][1])
    cut_m = int(math.floor(n * tm))
    cut_a = int(math.floor(n * ta))
    keep = set(by_m[cut_m:n - cut_m]) & set(by_a[cut_a:n - cut_a])
    if not keep:
        raise ValueError("edgrn: the trimming removed every gene")
    num = sum(rows[i][2] * rows[i][0] for i in keep)
    den = sum(rows[i][2] for i in keep)
    log2f = num / den if den > _EPS else 0.0
    return {"factor": 2.0 ** log2f, "log2_factor": log2f,
            "n_used": len(keep), "n_genes": n,
            "trimmed_m": 2 * cut_m, "trimmed_a": 2 * cut_a,
            "note": "the counts are NOT modified; this factor enters "
                    "the model as an offset"}


def effective_library_size(library_size, factor):
    r"""Where the factor actually goes: into the offset."""
    N, f = float(library_size), float(factor)
    if N <= 0.0 or f <= 0.0:
        raise ValueError("edgrn: the library size and factor must be "
                         "positive")
    return {"effective": N * f, "offset": math.log(N * f),
            "note": "an offset in the GLM, so the sampling "
                    "properties of the counts survive"}


def moderate_dispersion(gene_dispersions, common=None, prior_df=10.0,
                        df_residual=1.0):
    r"""Empirical Bayes shrinkage of :math:`\phi_g` toward a common
    value.

    Squeezing on the log scale with weights :math:`d_0` and
    :math:`d_g`: with few libraries per gene, :math:`d_0` dominates
    and the estimates are pulled to the trend.
    """
    p = [float(v) for v in k.vec(gene_dispersions)]
    if not p:
        raise ValueError("edgrn: no dispersions given")
    if any(v < 0.0 for v in p):
        raise ValueError("edgrn: a dispersion cannot be negative")
    d0, dg = float(prior_df), float(df_residual)
    if d0 < 0.0 or dg <= 0.0:
        raise ValueError("edgrn: the degrees of freedom must be "
                         "positive")
    pos = [v for v in p if v > _EPS]
    if common is None:
        c = (math.exp(sum(math.log(v) for v in pos) / len(pos))
             if pos else 0.0)
    else:
        c = float(common)
    w = d0 / (d0 + dg)
    out = []
    for v in p:
        if v <= _EPS or c <= _EPS:
            out.append((1.0 - w) * v + w * c)
        else:
            out.append(math.exp((1.0 - w) * math.log(v)
                                + w * math.log(c)))
    return {"dispersion": out, "common": c, "shrinkage": w,
            "prior_df": d0, "df_residual": dg,
            "note": "with few libraries per gene there are almost no "
                    "degrees of freedom, so d0 dominates"}


def _betainc(a, b, x, iters=300):
    """Regularised incomplete beta I_x(a,b) by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = k.lgamma(a) + k.lgamma(b) - k.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x, iters)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(iters):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0)
                                       * (a + 2.0 * m))
        else:
            num = (-((a + m) * (a + b + m) * x)
                   / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    return front * (f - 1.0) / a


def _chi2_sf(x, df, iters=400):
    """Upper tail of the chi-squared distribution."""
    if x <= 0.0:
        return 1.0
    a, xx = 0.5 * float(df), 0.5 * float(x)
    if xx < a + 1.0:
        term, tot, n = 1.0 / a, 1.0 / a, 0
        while n < iters:
            n += 1
            term *= xx / (a + n)
            tot += term
            if abs(term) < abs(tot) * 1e-15:
                break
        lower = tot * math.exp(-xx + a * math.log(xx) - k.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    f, c, d = 1.0, 1e30, 1.0 / (xx + 1.0 - a)
    f = d
    for i in range(1, iters):
        an = -i * (i - a)
        b = xx + 2.0 * i + 1.0 - a
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = f * math.exp(-xx + a * math.log(xx) - k.lgamma(a))
    return max(0.0, min(1.0, upper))


def _nb_logpmf(y, mu, phi):
    if phi <= _EPS:
        return (-mu + y * math.log(max(mu, _EPS))
                - k.lgamma(y + 1.0))
    r = 1.0 / phi
    return (k.lgamma(y + r) - k.lgamma(r) - k.lgamma(y + 1.0)
            + r * math.log(r / (r + mu))
            + y * math.log(mu / (r + mu)))


def exact_test(count_a, count_b, lib_a, lib_b, dispersion):
    r"""The NB analogue of Fisher's exact test on the two totals.

    Conditional on the total, sum the probabilities of every split at
    most as likely as the observed one.
    """
    ya, yb = float(count_a), float(count_b)
    Na, Nb = float(lib_a), float(lib_b)
    phi = float(dispersion)
    if min(ya, yb) < 0.0 or Na <= 0.0 or Nb <= 0.0:
        raise ValueError("edgrn: counts must be non-negative and "
                         "library sizes positive")
    total = ya + yb
    p_common = total / (Na + Nb)
    obs = _nb_logpmf(ya, Na * p_common, phi) \
        + _nb_logpmf(yb, Nb * p_common, phi)
    num, den = 0.0, 0.0
    for s in range(int(total) + 1):
        lp = (_nb_logpmf(float(s), Na * p_common, phi)
              + _nb_logpmf(total - s, Nb * p_common, phi))
        den += math.exp(lp)
        if lp <= obs + 1e-12:
            num += math.exp(lp)
    return {"p_value": min(1.0, num / den) if den > 0.0 else 1.0,
            "logFC": math.log((ya / Na + _EPS) / (yb / Nb + _EPS), 2),
            "dispersion": phi,
            "note": "conditional on the total count, as in Fisher's "
                    "exact test"}


def ql_f_test(lrt, q, quasi_dispersion, df_residual, df_prior=None):
    r""":math:`\mathrm{LRT}/(q\hat\Phi)` against an :math:`F`.

    The denominator degrees of freedom are FINITE, which is how the
    uncertainty in the estimated dispersion reaches the test; a
    chi-squared LRT treats it as known.
    """
    L, Q = float(lrt), int(q)
    P = float(quasi_dispersion)
    d2 = float(df_residual) + (0.0 if df_prior is None
                               else float(df_prior))
    if Q < 1 or P <= 0.0 or d2 <= 0.0:
        raise ValueError("edgrn: need q >= 1 and positive dispersion "
                         "and degrees of freedom")
    F = L / (Q * P)
    x = float(Q) * F / (float(Q) * F + d2)
    p = 1.0 - _betainc(0.5 * Q, 0.5 * d2, x) if x > 0.0 else 1.0
    chisq = _chi2_sf(L, Q)
    return RichResult(payload={
        "estimate": F, "F": F, "df1": Q, "df2": d2, "p_value": p,
        "lrt_p_value": chisq,
        "method": "quasi-likelihood F-test with shrunken dispersion; "
                  "Lund, Nettleton, McCarthy & Smyth (2012)",
        "note": "finite df2 carries the uncertainty in the estimated "
                "dispersion into the test; the LRT treats it as known "
                "and is therefore more liberal",
    })


def cheatsheet():
    return ("edgrn: counts, not intensities -- Y_gi ~ NB(M_i p_gj, "
            "phi_g) with variance mu(1 + mu phi), Poisson exactly at "
            "phi = 0, so BIOLOGICAL and TECHNICAL variation separate. "
            "Few libraries means almost no df per gene, so MODERATE "
            "the dispersions toward a common trend (limma's logic). "
            "Normalisation is a SEPARATE problem: TMM takes a doubly "
            "trimmed (30% on M, 5% on A) weighted mean of log ratios "
            "under the assumption most genes are not DE, and the "
            "counts are NOT modified -- the factor is an OFFSET. Liver "
            "vs kidney: 0.68, turning a 77%-toward-kidney result into "
            "47/53. Test by NB exact/LRT or by the QL F-test, whose "
            "finite df2 carries the dispersion's own uncertainty.")


# compact alias per ledger/NAMING.md
edger = ql_f_test

# public names resolved by fn/_lazy_map.json
edger_diff = ql_f_test
edgerdiff = ql_f_test
