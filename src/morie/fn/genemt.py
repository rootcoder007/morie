# morie.fn -- function file (rootcoder007/morie)
r"""MAGMA: gene and gene-set analysis by regression.

Single-marker association is underpowered when individual effects are
weak. Aggregating markers into genes, and genes into sets, is the
standard response, and the existing tools shared three problems:
statistical power strongly affected by **linkage disequilibrium**
between markers, multi-marker associations hard to detect, and
reliance on **permutation** for p-values, which is what made the
analysis expensive.

**Gene analysis as multiple regression.** The markers in a gene are
projected onto principal components of their LD structure, and the
gene statistic is an F-test of the joint fit. Because the components
are orthogonal, LD is accounted for by construction rather than by
resampling -- and the p-value is analytic, which is where the speed
comes from.

**Gene-set analysis as a separate layer.** It is built *around* the
gene analysis rather than fused into it, and that separation is the
design decision: the gene-level results are computed once and reused,
so a new gene set costs almost nothing. The set test is itself a
regression of gene :math:`Z`-scores on set membership,

.. math:: Z_g = \beta_0 + \beta_s S_g + \text{covariates} + \epsilon,

which generalises immediately to **continuous** gene properties, to
multiple sets at once, and to conditioning one set on another. A
mean-difference test cannot do any of those.

**Gene size and density must be covariates.** Longer genes carry more
markers and larger statistics for reasons that have nothing to do with
the trait; leaving them out yields sets enriched for nothing but
length. ``gene_covariates`` builds them, and the anchor shows the
spurious enrichment appearing when they are omitted.

References
----------
de Leeuw, C. A., Mooij, J. M., Heskes, T. & Posthuma, D. (2015)
"MAGMA: Generalized Gene-Set Analysis of GWAS Data", *PLoS
Computational Biology* 11(4), e1004219,
doi:10.1371/journal.pcbi.1004219. The stated problems with existing
gene and gene-set tools -- power strongly affected by linkage
disequilibrium between markers, multi-marker associations hard to
detect, and reliance on permutation making analysis computationally
expensive; the gene analysis based on a MULTIPLE REGRESSION model for
better statistical performance; the gene-set analysis built as a
SEPARATE LAYER around the gene analysis for flexibility; the
regression structure allowing generalisation to continuous properties
of genes and simultaneous analysis of multiple gene sets and other
gene properties; and the demonstration of more power at correct type-1
error and considerably faster analysis.

Purcell, S. et al. (2007) "PLINK: A Tool Set for Whole-Genome
Association and Population-Based Linkage Analyses", *American Journal
of Human Genetics* 81(3), 559-575, doi:10.1086/519795.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["ld_principal_components", "gene_statistic",
           "gene_covariates", "gene_set_regression",
           "conditional_set_test"]

_EPS = 1e-12


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p):
    q = min(max(float(p), 1e-12), 1.0 - 1e-12)
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ld_principal_components(G, keep=0.999):
    r"""Project the gene's markers onto the PCs of their LD.

    Orthogonal components mean LD is handled by construction rather
    than by permutation, which is both the power and the speed.
    """
    M = [[float(v) for v in r] for r in k.mat(G)]
    n, p = len(M), len(M[0])
    cols = []
    for j in range(p):
        c = [M[i][j] for i in range(n)]
        m = sum(c) / n
        s = math.sqrt(sum((v - m) ** 2 for v in c) / max(n - 1, 1))
        cols.append([(v - m) / s if s > _EPS else 0.0 for v in c])
    C = [[sum(cols[a][i] * cols[b][i] for i in range(n))
          / max(n - 1, 1) for b in range(p)] for a in range(p)]
    vals, vecs = np.linalg.eigh(C)
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    tot = sum(max(v, 0.0) for v in vals) or 1.0
    acc, take = 0.0, []
    for i in order:
        if vals[i] <= 1e-10:
            continue
        take.append(i)
        acc += vals[i] / tot
        if acc >= float(keep):
            break
    PC = [[sum(cols[a][i] * vecs[a][j] for a in range(p))
           for j in take] for i in range(n)]
    return {"components": PC, "n_components": len(take),
            "n_markers": p,
            "variance_explained": acc,
            "note": "orthogonal by construction, so LD needs no "
                    "permutation"}


def gene_statistic(y, G, keep=0.999):
    r"""F-test of the joint fit on the retained components."""
    yv = [float(v) for v in k.vec(y)]
    pc = ld_principal_components(G, keep)
    X = pc["components"]
    n, m = len(X), pc["n_components"]
    if len(yv) != n:
        raise ValueError("genemt: %d phenotypes but %d individuals"
                         % (len(yv), n))
    if m < 1:
        raise ValueError("genemt: the gene has no non-degenerate "
                         "components")
    co = k.wls(X, yv, [1.0] * n, 1e-8)["coef"]
    fit = [co[0] + sum(X[i][a] * co[1 + a] for a in range(m))
           for i in range(n)]
    ybar = sum(yv) / n
    ssr = sum((fit[i] - ybar) ** 2 for i in range(n))
    sse = sum((yv[i] - fit[i]) ** 2 for i in range(n))
    d2 = max(n - m - 1, 1)
    F = (ssr / m) / (sse / d2) if sse > _EPS else float("inf")
    z = math.sqrt(2.0 * F) - math.sqrt(2.0 * m - 1.0)
    p = 1.0 - _norm_cdf(z)
    return {"F": F, "df1": m, "df2": d2, "p": p,
            "z": _norm_ppf(1.0 - p), "n_markers": pc["n_markers"],
            "note": "an ANALYTIC p-value; no permutation"}


def gene_covariates(n_markers, gene_length, ld_scores=None):
    r"""Gene size and marker density as covariates.

    Longer genes give larger statistics for reasons unrelated to the
    trait; omitting these yields sets enriched for length alone.
    """
    nm = [float(v) for v in k.vec(n_markers)]
    gl = [float(v) for v in k.vec(gene_length)]
    if len(nm) != len(gl):
        raise ValueError("genemt: %d marker counts but %d lengths"
                         % (len(nm), len(gl)))
    if any(v <= 0.0 for v in nm + gl):
        raise ValueError("genemt: marker counts and lengths must be "
                         "positive")
    dens = [nm[i] / gl[i] for i in range(len(nm))]
    cov = [[math.log(nm[i]), math.log(gl[i]), math.log(dens[i])]
           for i in range(len(nm))]
    if ld_scores is not None:
        ls = [float(v) for v in k.vec(ld_scores)]
        cov = [cov[i] + [ls[i]] for i in range(len(cov))]
    return {"covariates": cov,
            "names": ["log_n_markers", "log_length", "log_density"]
            + (["ld_score"] if ld_scores is not None else []),
            "note": "not optional: without them, long genes look "
                    "enriched for everything"}


def gene_set_regression(z_scores, membership, covariates=None):
    r"""Regress gene :math:`Z` on set membership, with covariates.

    A regression rather than a mean-difference test, which is what
    permits continuous gene properties and conditioning.
    """
    z = [float(v) for v in k.vec(z_scores)]
    s = [float(v) for v in k.vec(membership)]
    n = len(z)
    if len(s) != n:
        raise ValueError("genemt: %d z-scores but %d membership "
                         "values" % (n, len(s)))
    X = [[s[i]] for i in range(n)]
    if covariates is not None:
        C = [[float(v) for v in r] for r in k.mat(covariates)]
        if len(C) != n:
            raise ValueError("genemt: %d covariate rows for %d genes"
                             % (len(C), n))
        X = [X[i] + list(C[i]) for i in range(n)]
    co = k.wls(X, z, [1.0] * n, 1e-8)["coef"]
    fit = [co[0] + sum(X[i][a] * co[1 + a] for a in range(len(X[0])))
           for i in range(n)]
    res = [z[i] - fit[i] for i in range(n)]
    dof = max(n - len(X[0]) - 1, 1)
    s2 = sum(v * v for v in res) / dof
    sm = sum(s) / n
    sxx = sum((s[i] - sm) ** 2 for i in range(n))
    se = math.sqrt(s2 / sxx) if sxx > _EPS else float("inf")
    t = co[1] / se if se > 0 else 0.0
    return RichResult(payload={
        "estimate": co[1], "beta": co[1], "se": se, "t": t,
        "p": 1.0 - _norm_cdf(t),
        "n_genes": n, "covariates_used": covariates is not None,
        "method": "MAGMA gene-set regression; de Leeuw et al. (2015)",
        "note": "one-sided: enrichment means a POSITIVE coefficient",
    })


def conditional_set_test(z_scores, set_a, set_b, covariates=None):
    r"""Test one set conditional on another.

    Two overlapping sets can each look enriched on their own while
    only one carries the signal -- a mean-difference test cannot ask
    this question at all.
    """
    z = [float(v) for v in k.vec(z_scores)]
    a = [float(v) for v in k.vec(set_a)]
    b = [float(v) for v in k.vec(set_b)]
    n = len(z)
    if not (len(a) == len(b) == n):
        raise ValueError("genemt: the sets and z-scores differ in "
                         "length")
    base = [[b[i]] for i in range(n)]
    if covariates is not None:
        C = [[float(v) for v in r] for r in k.mat(covariates)]
        base = [base[i] + list(C[i]) for i in range(n)]
    marg = gene_set_regression(z, a, covariates)
    cond = gene_set_regression(z, a, base)
    return {"marginal_beta": marg["beta"], "marginal_p": marg["p"],
            "conditional_beta": cond["beta"],
            "conditional_p": cond["p"],
            "attenuation": (marg["beta"] - cond["beta"])
            / marg["beta"] if abs(marg["beta"]) > _EPS else 0.0,
            "note": "if the signal was really the other set, the "
                    "conditional coefficient collapses"}


def cheatsheet():
    return ("genemt: single markers are underpowered, so aggregate -- "
            "but existing tools lost power to LINKAGE DISEQUILIBRIUM "
            "and needed PERMUTATION for p-values. MAGMA's gene test is "
            "a MULTIPLE REGRESSION on principal components of the LD "
            "structure: orthogonal by construction, analytic p-value, "
            "hence fast. The gene-set test is a SEPARATE LAYER around "
            "it -- a regression of gene Z-scores on membership, which "
            "generalises to CONTINUOUS gene properties, multiple sets "
            "at once, and conditioning one set on another. Gene size "
            "and density are covariates, or long genes look enriched "
            "for everything.")


# compact alias per ledger/NAMING.md
magma = gene_set_regression
