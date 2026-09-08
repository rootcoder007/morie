# morie.fn -- function file (rootcoder007/morie)
r"""Whole-genome regression in two ridge levels, REGENIE style.

**The problem.** A genome-wide association scan tests each variant
against a phenotype, but relatedness and polygenic background inflate
the statistics. Mixed models fix that and are slow. REGENIE gets the
same correction from stacked ridge regressions, decoupled into two
steps so Step 1 is fitted once and reused.

**Step 1, level 0.** Partition the array markers into consecutive
blocks of :math:`B` (the paper uses 1000). Inside a block, fit
:math:`J` ridge regressions (the paper uses 5) at *different* shrinkage
parameters,

.. math:: \hat\beta_\lambda = (X_b'X_b + \lambda I)^{-1} X_b' y ,

which is the maximum a posteriori estimate under a Gaussian prior on
the block's effect sizes. The spread of :math:`\lambda` is the point:
nobody knows how many markers in a window are truly associated or how
big their effects are, so the block emits a small range of local
polygenic scores rather than one. :math:`M = 500{,}000` markers at
:math:`B = 1000, J = 5` become :math:`2{,}500` predictors -- a 200-fold
reduction, and the reason the method is fast.

**Step 1, level 1.** Stack those predictors and fit a second ridge
*within cross-validation* to combine them into one predictor. The CV
matters: the level-0 predictions were fitted on the same phenotype, so
combining them in-sample would be circular. ``fit`` uses K-fold by
default and leave-one-out on request, both of which the paper offers.

**LOCO.** The combined predictor is then decomposed by chromosome:
the prediction for chromosome :math:`c` is the sum of the level-0
contributions from every *other* chromosome. Testing a variant against
a background that includes its own chromosome is proximal
contamination -- the background absorbs the very signal being tested,
and power drops. ``loco_predictions`` builds each one by exclusion, and
the anchor checks the exclusion actually happened: no block on
chromosome :math:`c` may contribute to the chromosome-:math:`c`
background.

Because the LOCO scheme here works on *block* effects rather than
individual variant effects, a pair of markers in long-distance LD is
combined within its block first, which the paper notes makes it less
prone to the interchromosomal-LD problem than variant-level schemes.

**Step 2.** Each variant is tested with the LOCO prediction as a
covariate. The score statistic for variant :math:`g` given offset
:math:`\hat y_{\rm LOCO}` is the usual residual correlation, and the
module returns the effect, its standard error, the chi-square and the
p-value.

References
----------
Mbatchou, J., Barnard, L., Backman, J., Marcketta, A., Kosmicki, J. A.,
Ziyatdinov, A., Benner, C., O'Dushlaine, C., Barber, M., Boutkov, B.,
Habegger, L., Ferreira, M., Baras, A., Reid, J., Abecasis, G., Maxwell,
E. & Marchini, J. (2021) "Computationally efficient whole-genome
regression for quantitative and binary traits", *Nature Genetics*
53(7), 1097-1103, doi:10.1038/s41588-021-00870-7 (preprint:
bioRxiv 2020.06.19.162354). The Step 1 / Step 2 split, the partition
into blocks of :math:`B` markers with :math:`J` ridge predictors each
at different shrinkage parameters, the Gaussian-prior / MAP reading of
those predictors, the reduction from 500,000 markers to 2,500
predictors at :math:`B = 1000, J = 5`, the second ridge under K-fold or
leave-one-out cross-validation, the decomposition into per-chromosome
LOCO predictions, and the argument that block-level LOCO is less
exposed to interchromosomal LD than variant-level LOCO.

Hoerl, A. E. & Kennard, R. W. (1970) "Ridge Regression: Biased
Estimation for Nonorthogonal Problems", *Technometrics* 12(1), 55-67,
doi:10.1080/00401706.1970.10488634, for the estimator itself.

Yang, J., Zaitlen, N. A., Goddard, M. E., Visscher, P. M. & Price,
A. L. (2014) "Advantages and pitfalls in the application of
mixed-model association methods", *Nature Genetics* 46(2), 100-106,
doi:10.1038/ng.2876, for proximal contamination and why LOCO is used.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["make_blocks", "ridge_fit", "level0_predictors",
           "level1_stack", "loco_predictions", "test_variant",
           "fit", "CV_SCHEMES"]

CV_SCHEMES = ("kfold", "loo")


def make_blocks(n_markers, chromosomes=None, block_size=1000):
    r"""Consecutive blocks of ``block_size`` markers.

    Blocks never straddle a chromosome boundary: a block is meant to
    capture local linkage disequilibrium, and two markers on different
    chromosomes are not in LD.
    """
    n = int(n_markers)
    b = int(block_size)
    if n < 1:
        raise ValueError("regmlm: no markers")
    if b < 1:
        raise ValueError("regmlm: block_size must be positive")
    chrom = ([0] * n if chromosomes is None
             else [int(c) for c in chromosomes])
    if len(chrom) != n:
        raise ValueError("regmlm: one chromosome label per marker")
    blocks = []
    start = 0
    for i in range(1, n + 1):
        boundary = (i == n or chrom[i] != chrom[i - 1]
                    or i - start >= b)
        if boundary:
            blocks.append({"start": start, "stop": i,
                           "chromosome": chrom[start],
                           "size": i - start})
            start = i
    return blocks


def ridge_fit(X, y, lam):
    r"""MAP estimate under a Gaussian prior:
    :math:`(X'X + \lambda I)^{-1}X'y`."""
    n = len(y)
    if n != len(X):
        raise ValueError("regmlm: X and y must have the same length")
    p = len(X[0])
    if lam < 0.0:
        raise ValueError("regmlm: the shrinkage parameter cannot be "
                         "negative")
    A = [[sum(X[i][r] * X[i][c] for i in range(n))
          + (float(lam) if r == c else 0.0) for c in range(p)]
         for r in range(p)]
    b = [sum(X[i][r] * y[i] for i in range(n)) for r in range(p)]
    beta = [float(v) for v in np.linalg.solve(np.array(A), np.array(b))]
    fitted = [sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    return {"beta": beta, "fitted": fitted, "lam": float(lam)}


def _lambda_grid(p, n, n_ridge=5):
    r"""The paper's spread of shrinkage values.

    Parameterised through a grid of prior heritabilities
    :math:`h^2 \in (0,1)`, so :math:`\lambda = p(1-h^2)/h^2` runs from
    almost no shrinkage to almost total shrinkage regardless of block
    size.
    """
    j = int(n_ridge)
    if j < 1:
        raise ValueError("regmlm: need at least one ridge predictor")
    hs = [(k + 1.0) / (j + 1.0) for k in range(j)]
    return [p * (1.0 - h) / h for h in hs]


def level0_predictors(G, y, blocks, n_ridge=5):
    r"""Local polygenic scores: :math:`J` per block."""
    n = len(y)
    preds = []
    meta = []
    for b in blocks:
        Xb = [[float(G[i][j]) for j in range(b["start"], b["stop"])]
              for i in range(n)]
        for lam in _lambda_grid(b["size"], n, n_ridge):
            f = ridge_fit(Xb, y, lam)
            preds.append(f["fitted"])
            meta.append({"chromosome": b["chromosome"],
                         "start": b["start"], "stop": b["stop"],
                         "lam": lam})
    return {"predictors": preds, "meta": meta,
            "n_predictors": len(preds),
            "reduction": (blocks[-1]["stop"] / float(len(preds))
                          if preds else 0.0)}


def _folds(n, k, scheme):
    if scheme == "loo":
        return [[i] for i in range(n)]
    k = max(2, min(int(k), n))
    return [[i for i in range(n) if i % k == f] for f in range(k)]


def level1_stack(preds, y, cv="kfold", k=5, lam=None):
    r"""Second ridge over the level-0 predictors, inside CV.

    The level-0 predictions were fitted against this same phenotype, so
    combining them in-sample would be circular; the weights are learnt
    on held-out folds.
    """
    if cv not in CV_SCHEMES:
        raise ValueError("regmlm: cv must be one of %s, got %r"
                         % (", ".join(CV_SCHEMES), cv))
    n = len(y)
    m = len(preds)
    if m == 0:
        raise ValueError("regmlm: no level-0 predictors")
    X = [[preds[j][i] for j in range(m)] for i in range(n)]
    lam = float(m) if lam is None else float(lam)
    oof = [0.0] * n
    for fold in _folds(n, k, cv):
        keep = [i for i in range(n) if i not in set(fold)]
        if not keep:
            continue
        f = ridge_fit([X[i] for i in keep], [y[i] for i in keep], lam)
        for i in fold:
            oof[i] = sum(X[i][j] * f["beta"][j] for j in range(m))
    full = ridge_fit(X, y, lam)
    return {"weights": full["beta"], "prediction": full["fitted"],
            "out_of_fold": oof, "cv": cv, "lam": lam,
            "n_predictors": m}


def loco_predictions(preds, meta, weights, chromosomes=None):
    r"""One background per chromosome, built by leaving it out."""
    n = len(preds[0])
    chroms = sorted({m["chromosome"] for m in meta}) \
        if chromosomes is None else sorted(set(chromosomes))
    out = {}
    for c in chroms:
        keep = [j for j in range(len(preds))
                if meta[j]["chromosome"] != c]
        out[c] = [sum(preds[j][i] * weights[j] for j in keep)
                  for i in range(n)]
    return {"loco": out, "chromosomes": chroms,
            "note": "the chromosome-c background excludes every block "
                    "on chromosome c, which is what avoids proximal "
                    "contamination"}


def test_variant(g, y, offset=None, covariates=()):
    r"""Step 2: test one variant with the LOCO prediction as offset."""
    n = len(y)
    if n != len(g):
        raise ValueError("regmlm: genotype and phenotype lengths "
                         "differ")
    off = [0.0] * n if offset is None else [float(v) for v in offset]
    if len(off) != n:
        raise ValueError("regmlm: the offset must cover every sample")
    cols = [[1.0] * n] + [list(c) for c in covariates]
    resid_y = _residualise([y[i] - off[i] for i in range(n)], cols)
    resid_g = _residualise([float(v) for v in g], cols)
    sgg = sum(v * v for v in resid_g)
    if sgg <= 0.0:
        raise ValueError("regmlm: the variant is monomorphic after "
                         "adjustment")
    beta = sum(resid_g[i] * resid_y[i] for i in range(n)) / sgg
    dof = n - len(cols) - 1
    rss = sum((resid_y[i] - beta * resid_g[i]) ** 2 for i in range(n))
    s2 = rss / max(dof, 1)
    se = math.sqrt(s2 / sgg)
    chisq = (beta / se) ** 2 if se > 0.0 else float("inf")
    return {"beta": beta, "se": se, "chisq": chisq,
            "p_value": math.erfc(math.sqrt(chisq / 2.0)),
            "n": n}


def _residualise(v, cols):
    n = len(v)
    p = len(cols)
    A = [[sum(cols[r][i] * cols[c][i] for i in range(n))
          for c in range(p)] for r in range(p)]
    b = [sum(cols[r][i] * v[i] for i in range(n)) for r in range(p)]
    coef = [float(x) for x in np.linalg.solve(np.array(A), np.array(b))]
    return [v[i] - sum(cols[j][i] * coef[j] for j in range(p))
            for i in range(n)]


def fit(G, y, chromosomes=None, block_size=1000, n_ridge=5,
        cv="kfold", k=5):
    r"""Step 1 end to end: blocks, level 0, level 1, LOCO."""
    n = len(y)
    if n != len(G):
        raise ValueError("regmlm: G and y must have the same number "
                         "of samples")
    blocks = make_blocks(len(G[0]), chromosomes, block_size)
    lvl0 = level0_predictors(G, y, blocks, n_ridge)
    lvl1 = level1_stack(lvl0["predictors"], y, cv, k)
    loco = loco_predictions(lvl0["predictors"], lvl0["meta"],
                            lvl1["weights"])
    return RichResult(payload={
        "estimate": lvl1["prediction"], "blocks": blocks,
        "n_blocks": len(blocks), "level0": lvl0, "level1": lvl1,
        "loco": loco["loco"], "chromosomes": loco["chromosomes"],
        "n_predictors": lvl0["n_predictors"],
        "reduction": lvl0["reduction"],
        "method": "two-level ridge whole-genome regression with LOCO; "
                  "Mbatchou et al. (2021) Step 1",
    })


def cheatsheet():
    return ("regmlm: Step 1 is two stacked ridges. Level 0 fits J "
            "ridges per block of B markers at DIFFERENT shrinkages "
            "(MAP under a Gaussian prior), turning 500k markers into "
            "2.5k local polygenic scores at B=1000, J=5. Level 1 "
            "combines them under cross-validation -- in-sample would "
            "be circular. The combined predictor is split by "
            "chromosome, each background EXCLUDING its own "
            "chromosome, because testing a variant against a "
            "background containing it is proximal contamination. Step "
            "2 tests each variant with that background as an offset.")


# compact alias per ledger/NAMING.md
whole_genome_regression = fit
