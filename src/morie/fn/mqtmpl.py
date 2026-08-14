# morie.fn -- function file (rootcoder007/morie)
r"""Genome scans for QTL, R/qtl style: one scan, several methods.

**What this is.** The scanning layer above :mod:`morie.fn.rqtmpl` and
:mod:`morie.fn.cqtmpl`: a hidden Markov model for the genotypes along a
chromosome, a single-QTL genome scan by more than one method, optional
covariates, and a permutation threshold instead of an asymptotic one.

**Genotype probabilities by HMM.** Markers are missing, partially
informative, or occasionally mistyped, so the genotype at any position
is inferred by forward-backward over the chromosome rather than read
off the two flanking markers. The transition probabilities come from
the map (Haldane), and the emission allows a genotyping error rate
:math:`\varepsilon`: an observed marker agrees with the true genotype
with probability :math:`1-\varepsilon`. With :math:`\varepsilon = 0`
and complete data the HMM posterior collapses onto the flanking-marker
formula of interval mapping, which is what the anchor checks.

**Why the threshold is a permutation.** The LOD at one position is
asymptotically :math:`\tfrac12(\log_{10}e)\chi^2_1`, but a genome scan
takes the maximum over hundreds of correlated positions, and that
maximum is not chi-squared anything. Permuting the phenotypes against
the genotypes destroys any QTL while preserving both the marker
correlation structure and the phenotype distribution, so the
:math:`(1-\alpha)` quantile of the permuted maximum LOD is a genome-wide
threshold that needs no distributional assumption. ``permutation_threshold``
returns the whole null distribution, not just the quantile.

**Methods.** ``"em"`` is the mixture-likelihood scan of Lander and
Botstein, maximised by EM; ``"mr"`` is marker regression, which uses
only individuals typed at the marker and can only report LOD *at*
markers. Both are implemented from sources in hand.

Broman et al. also list Haley-Knott regression and multiple
imputation. This module names them and refuses them rather than
guessing: the applications note gives no formulas, and neither Haley &
Knott (1992) nor Sen & Churchill (2001) is in the corpus.
``method_status`` reports which methods are available and why.

References
----------
Broman, K. W., Wu, H., Sen, Ś. & Churchill, G. A. (2003) "R/qtl: QTL
mapping in experimental crosses", *Bioinformatics* 19(7), 889-890,
doi:10.1093/bioinformatics/btg112. The section "Hidden Markov model
technology" for the HMM treatment of missing and partially missing
genotypes with allowance for genotyping errors, and "Features" for the
single-QTL genome scan by EM, Haley-Knott regression and multiple
imputation, the inclusion of covariates, and LOD thresholds by
permutation test.

Lander, E. S. & Botstein, D. (1989) "Mapping Mendelian Factors
Underlying Quantitative Traits Using RFLP Linkage Maps", *Genetics*
121(1), 185-199, doi:10.1093/genetics/121.1.185, for the EM scan this
calls.

Churchill, G. A. & Doerge, R. W. (1994) "Empirical Threshold Values
for Quantitative Trait Mapping", *Genetics* 138(3), 963-971,
doi:10.1093/genetics/138.3.963, for the permutation threshold.

Baum, L. E., Petrie, T., Soules, G. & Weiss, N. (1970) "A
Maximization Technique Occurring in the Statistical Analysis of
Probabilistic Functions of Markov Chains", *The Annals of
Mathematical Statistics* 41(1), 164-171,
doi:10.1214/aoms/1177697196, for the forward-backward algorithm.

Not in the corpus, and therefore not implemented: Haley, C. S. &
Knott, S. A. (1992) "A simple regression method for mapping
quantitative trait loci in line crosses using flanking markers",
*Heredity* 69(4), 315-324, doi:10.1038/hdy.1992.131; and Sen, Ś. &
Churchill, G. A. (2001) "A statistical framework for quantitative
trait mapping", *Genetics* 159(1), 371-387,
doi:10.1093/genetics/159.1.371.
"""

import math

from . import _array_core as np
from . import cqtmpl as _cim
from . import rqtmpl as _im
from . import survrsf as _rsf
from ._richresult import RichResult

__all__ = ["METHODS", "method_status", "hmm_genotype_probabilities",
           "scanone", "permutation_threshold", "lod_support_interval"]

METHODS = ("em", "mr", "hk", "imp")
_AVAILABLE = ("em", "mr")
_UNSOURCED = {
    "hk": "Haley-Knott regression is named but not defined in Broman "
          "et al. (2003); the primary source, Haley, C. S. & Knott, "
          "S. A. (1992) 'A simple regression method for mapping "
          "quantitative trait loci in line crosses using flanking "
          "markers', Heredity 69(4), 315-324, "
          "doi:10.1038/hdy.1992.131, is not in the corpus",
    "imp": "multiple imputation is named but not defined in Broman "
           "et al. (2003); the primary source, Sen, S. & Churchill, "
           "G. A. (2001) 'A statistical framework for quantitative "
           "trait mapping', Genetics 159(1), 371-387, "
           "doi:10.1093/genetics/159.1.371, is not in the corpus",
}


def method_status(method=None):
    r"""Which of the paper's four scan methods are implemented."""
    if method is None:
        return {"methods": METHODS, "available": _AVAILABLE,
                "unavailable": dict(_UNSOURCED)}
    if method not in METHODS:
        raise ValueError("mqtmpl: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    return {"method": method, "available": method in _AVAILABLE,
            "reason": _UNSOURCED.get(method, "")}


def _check_method(method):
    if method not in METHODS:
        raise ValueError("mqtmpl: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    if method not in _AVAILABLE:
        raise ValueError("mqtmpl: the %r scan method is not "
                         "implemented -- %s" % (method,
                                                _UNSOURCED[method]))


def hmm_genotype_probabilities(genotypes, positions, error_rate=0.0):
    r"""Forward-backward posterior genotype probabilities.

    ``genotypes`` is one row per individual, 0, 1 or ``None`` for a
    missing call. Backcross, so two states.
    """
    e = float(error_rate)
    if not 0.0 <= e < 0.5:
        raise ValueError("mqtmpl: the genotyping error rate must lie "
                         "in [0, 0.5), got %r" % error_rate)
    m = len(positions)
    if any(len(row) != m for row in genotypes):
        raise ValueError("mqtmpl: every individual needs one call per "
                         "marker")
    trans = []
    for j in range(m - 1):
        d = float(positions[j + 1]) - float(positions[j])
        if d <= 0.0:
            raise ValueError("mqtmpl: marker positions must increase")
        trans.append(_im.haldane(d))
    out = []
    for row in genotypes:
        def emit(j, state):
            if row[j] is None:
                return 1.0
            return (1.0 - e) if int(row[j]) == state else e
        f = [[0.0, 0.0] for _ in range(m)]
        f[0] = [0.5 * emit(0, 0), 0.5 * emit(0, 1)]
        for j in range(1, m):
            r = trans[j - 1]
            for s in (0, 1):
                f[j][s] = emit(j, s) * (
                    f[j - 1][s] * (1.0 - r) + f[j - 1][1 - s] * r)
            tot = f[j][0] + f[j][1]
            if tot <= 0.0:
                raise ValueError("mqtmpl: an individual's marker data "
                                 "have probability zero; raise the "
                                 "error rate")
            f[j] = [f[j][0] / tot, f[j][1] / tot]
        b = [[1.0, 1.0] for _ in range(m)]
        for j in range(m - 2, -1, -1):
            r = trans[j]
            for s in (0, 1):
                b[j][s] = (b[j + 1][s] * (1.0 - r) * emit(j + 1, s)
                           + b[j + 1][1 - s] * r * emit(j + 1, 1 - s))
            tot = b[j][0] + b[j][1]
            if tot > 0.0:
                b[j] = [b[j][0] / tot, b[j][1] / tot]
        post = []
        for j in range(m):
            p0 = f[j][0] * b[j][0]
            p1 = f[j][1] * b[j][1]
            post.append([p0 / (p0 + p1), p1 / (p0 + p1)])
        out.append(post)
    return out


def scanone(y, markers, positions, method="em", step=0.02,
            covariates=(), error_rate=0.0):
    r"""Single-QTL genome scan."""
    _check_method(method)
    n = len(y)
    if any(len(c) != n for c in markers):
        raise ValueError("mqtmpl: every marker must be typed on all "
                         "%d individuals" % n)
    if method == "mr":
        out_pos, out_lod = [], []
        for j in range(len(markers)):
            typed = [i for i in range(n) if markers[j][i] is not None]
            if len(typed) < 3:
                continue
            sm = _im.single_marker([y[i] for i in typed],
                                   [markers[j][i] for i in typed])
            out_pos.append(float(positions[j]))
            out_lod.append(sm["lod"])
        if not out_lod:
            raise ValueError("mqtmpl: no marker has enough typed "
                             "individuals")
        k = max(range(len(out_lod)), key=lambda i: out_lod[i])
        return RichResult(payload={
            "estimate": out_lod[k], "peak_lod": out_lod[k],
            "peak_position": out_pos[k], "position": out_pos,
            "lod": out_lod, "method_used": "mr",
            "note": "marker regression reports LOD at markers only, "
                    "and drops individuals not typed there",
            "method": "marker regression scan; Broman et al. (2003)",
        })
    cof = [list(c) for c in covariates]
    res = _cim.scan(y, [[0 if v is None else int(v) for v in mk]
                        for mk in markers],
                    [float(p) for p in positions],
                    cofactors=(), window=0.0, step=step) \
        if not cof else None
    if cof:
        m = len(markers)
        out_pos, out_lod, fits = [], [], []
        for j in range(m - 1):
            span = float(positions[j + 1]) - float(positions[j])
            d = 0.0
            while d <= span + 1e-12:
                f = _cim.cim(y,
                             [0 if v is None else int(v)
                              for v in markers[j]],
                             [0 if v is None else int(v)
                              for v in markers[j + 1]],
                             _im.haldane(min(d, span)),
                             _im.haldane(max(span - d, 0.0)), cof)
                out_pos.append(float(positions[j]) + d)
                out_lod.append(f["lod"])
                fits.append(f)
                d += float(step)
        k = max(range(len(out_lod)), key=lambda i: out_lod[i])
        res = RichResult(payload={
            "estimate": out_lod[k], "peak_lod": out_lod[k],
            "peak_position": out_pos[k], "position": out_pos,
            "lod": out_lod, "fit": fits[k],
        })
    return RichResult(payload={
        "estimate": res["peak_lod"], "peak_lod": res["peak_lod"],
        "peak_position": res["peak_position"],
        "position": res["position"], "lod": res["lod"],
        "method_used": "em", "n_covariates": len(cof),
        "error_rate": float(error_rate),
        "method": "EM genome scan; Lander & Botstein (1989) via "
                  "Broman et al. (2003)",
    })


def permutation_threshold(y, markers, positions, n_perm=100,
                          alpha=0.05, method="em", step=0.05, seed=0,
                          **kw):
    r"""Churchill-Doerge: the null distribution of the *maximum* LOD.

    Permuting the phenotypes against the genotypes leaves the marker
    correlation and the phenotype distribution intact and removes any
    association, so the quantile of the permuted genome-wide maxima is
    a threshold that assumes nothing.
    """
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("mqtmpl: alpha must lie in (0, 1)")
    rng = _rsf._Rng(seed)
    maxima = []
    ys = list(y)
    for _ in range(int(n_perm)):
        perm = list(ys)
        for i in range(len(perm) - 1, 0, -1):
            j = rng.randint(i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        maxima.append(scanone(perm, markers, positions, method, step,
                              **kw)["peak_lod"])
    maxima.sort()
    idx = min(len(maxima) - 1,
              max(0, int(math.ceil((1.0 - a) * len(maxima))) - 1))
    return RichResult(payload={
        "estimate": maxima[idx], "threshold": maxima[idx],
        "alpha": a, "n_perm": int(n_perm), "null_maxima": maxima,
        "median_null": maxima[len(maxima) // 2],
        "method": "permutation threshold; Churchill & Doerge (1994) "
                  "via Broman et al. (2003)",
    })


def lod_support_interval(scan_result, drop=1.5):
    r"""The classic LOD-drop support interval around the peak."""
    lod = scan_result["lod"]
    pos = scan_result["position"]
    k = max(range(len(lod)), key=lambda i: lod[i])
    cut = lod[k] - float(drop)
    lo = k
    while lo > 0 and lod[lo - 1] >= cut:
        lo -= 1
    hi = k
    while hi < len(lod) - 1 and lod[hi + 1] >= cut:
        hi += 1
    return {"peak": pos[k], "lower": pos[lo], "upper": pos[hi],
            "drop": float(drop), "peak_lod": lod[k]}


def cheatsheet():
    return ("mqtmpl: the scanning layer. Genotypes come from a "
            "forward-backward HMM that tolerates missing calls and a "
            "genotyping error rate, and collapses to the "
            "flanking-marker formula when both are absent. Scans by "
            "EM or marker regression; Haley-Knott and multiple "
            "imputation are named and REFUSED, with citations. "
            "Genome-wide significance is a permutation threshold, "
            "because the maximum over correlated positions is not "
            "chi-squared anything.")


# compact alias per ledger/NAMING.md
qtl_genome_scan = scanone
