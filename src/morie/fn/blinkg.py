"""BLINK: iterative fixed-effect GWAS with LD-filtered pseudo-QTNs.

A genome scan that tests one marker at a time against a null of no
association has two well-known problems. Population structure and
relatedness inflate the test, and a real signal leaks into every marker
in linkage disequilibrium with it, so one causal variant produces a
plateau of significance rather than a peak. The mixed-model answer puts
a kinship random effect in the model and pays for it with a variance
component estimated by restricted maximum likelihood at every step.

BLINK is the fixed-effect answer. The relatedness that the random effect
was carrying is instead absorbed by a handful of markers -- pseudo-QTNs
-- carried as covariates, and the variance component is replaced by a
model-selection criterion. Two models alternate:

  the SCAN         test every marker in turn with the current pseudo-QTNs
                   as covariates. A marker that is itself a pseudo-QTN is
                   dropped from the covariate set while it is being
                   tested, because a variable cannot be its own control.
  the SELECTION    re-choose the pseudo-QTNs from that scan.

The selection is what gives BLINK its name. Markers are sorted by p
value and the ones above a Bonferroni threshold are discarded. The most
significant survivor is taken; every marker whose correlation with it
exceeds a threshold -- 0.7 in the paper -- is dropped; the most
significant of what remains is taken next; and so on. This replaces
FarmCPU's fixed genomic bins, which is the older route and is kept here
because a bin is the right tool when marker positions are known and
correlation is not: it never drops a distant marker that happens to
correlate by chance. Both are selectable and the choice travels in the
result.

How MANY of those markers to keep is then a model-selection question, and
BLINK answers it with a criterion rather than a variance component: fit
the first k of them, for k running from one to all of them, and take the
k that minimises

    BIC = 2 (-log likelihood) + k log n

The whole thing iterates until the pseudo-QTN set stops changing. That
it stops is not guaranteed by anything -- it is a fixed point of a
discrete map -- so the number of iterations and whether it actually
settled are both reported rather than assumed.

References
  Huang, M., Liu, X., Zhou, Y., Summers, R.M. and Zhang, Z. (2019)
    "BLINK: a package for the next level of genome-wide association
    studies with both individuals and markers in the millions."
    GigaScience 8(2), giy154. doi:10.1093/gigascience/giy154. The LD
    filter with its 0.7 threshold and the Bonferroni pre-filter at
    alpha = 0.01, the BIC = 2(-LL) + k log n selection, and the
    iteration to a stable pseudo-QTN set.
  Liu, X., Huang, M., Fan, B., Buckler, E.S. and Zhang, Z. (2016)
    "Iterative usage of fixed and random effect models for powerful and
    efficient genome-wide association studies." PLoS Genetics 12(2),
    e1005767. FarmCPU, the bin-based predecessor.
  Devlin, B. and Roeder, K. (1999) "Genomic control for association
    studies." Biometrics 55(4), 997-1004. The inflation factor reported
    alongside the scan.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["blinkg", "blink_gwas", "marker_scan", "ld_filter", "bin_filter",
           "select_by_criterion", "SELECTIONS", "CRITERIA", "LD_THRESHOLD",
           "ALPHA", "cheatsheet"]

SELECTIONS = ("ld", "bin")
CRITERIA = ("bic", "aic", "none")

# The paper's defaults: markers correlated above this with an already
# chosen pseudo-QTN are dropped, and the Bonferroni pre-filter runs at
# this alpha.
LD_THRESHOLD = 0.7
ALPHA = 0.01

# The median of a chi-square on one degree of freedom. Genomic control
# divides the observed median by this, so it is a constant of the method
# and not a fitted quantity.
CHISQ1_MEDIAN = 0.45493642311957283


def _corr(a, b):
    """Pearson correlation, compensated, zero when either is constant."""
    n = len(a)
    ma = _w.csum(a) / n
    mb = _w.csum(b) / n
    saa = _w.csum((v - ma) * (v - ma) for v in a)
    sbb = _w.csum((v - mb) * (v - mb) for v in b)
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    sab = _w.csum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return sab / math.sqrt(saa * sbb)


def _design(n, covars, cols):
    """Intercept, fixed covariates, then the given genotype columns."""
    d = []
    for i in range(n):
        row = [1.0]
        for c in covars:
            row.append(c[i])
        for c in cols:
            row.append(c[i])
        d.append(row)
    return d


def marker_scan(y, geno, covars=None, qtn=()):
    """Test every marker with the current pseudo-QTNs as covariates.

    Returns the effect, its standard error, the t statistic and the two
    sided p value for each marker. A marker with no variation, or one
    whose design is rank deficient once the covariates are in, gets a
    p value of nan rather than a fabricated number.
    """
    n = len(y)
    m = len(geno)
    covars = [] if covars is None else [[float(v) for v in c]
                                        for c in covars]
    qtn = list(qtn)
    beta = []
    se = []
    tstat = []
    pval = []
    for j in range(m):
        xj = geno[j]
        # A pseudo-QTN cannot be its own control, so it comes out of the
        # covariate set exactly while it is the marker under test.
        cols = [geno[q] for q in qtn if q != j]
        d = _design(n, covars, cols + [xj])
        p = len(d[0])
        if n <= p:
            beta.append(float("nan"))
            se.append(float("nan"))
            tstat.append(float("nan"))
            pval.append(float("nan"))
            continue
        try:
            fit = _w.ols(y, d)
        except Exception:
            beta.append(float("nan"))
            se.append(float("nan"))
            tstat.append(float("nan"))
            pval.append(float("nan"))
            continue
        b = fit["beta"][p - 1]
        v = fit["sigma2"] * fit["xtx_inv"][p - 1][p - 1]
        if not v > 0.0 or v != v:
            beta.append(b)
            se.append(float("nan"))
            tstat.append(float("nan"))
            pval.append(float("nan"))
            continue
        s = math.sqrt(v)
        t = b / s
        beta.append(b)
        se.append(s)
        tstat.append(t)
        pval.append(2.0 * _w.t_sf(abs(t), fit["df"]))
    return {"beta": beta, "se": se, "t": tstat, "p": pval}


def _order_by_p(pval):
    """Markers sorted by p value, ties broken by index.

    The tie break matters: on a small panel several markers can share a
    p value to the last bit, and a selection that depended on which one
    the sort happened to put first would not be reproducible.
    """
    live = [(pval[j], j) for j in range(len(pval)) if pval[j] == pval[j]]
    live.sort()
    return [j for _, j in live]


def ld_filter(geno, order, threshold=LD_THRESHOLD):
    """Keep the most significant marker, drop what correlates with it.

    Walks the p-value ordering once. Each surviving marker is compared
    with every marker already kept, and dropped if the absolute
    correlation exceeds the threshold. This is BLINK's replacement for
    FarmCPU's bins.
    """
    kept = []
    for j in order:
        ok = True
        for q in kept:
            if abs(_corr(geno[j], geno[q])) > threshold:
                ok = False
                break
        if ok:
            kept.append(j)
    return kept


def bin_filter(order, positions, bin_size):
    """One marker per genomic bin, the most significant in the bin.

    FarmCPU's rule. It cannot drop a distant marker that correlates by
    chance, and it cannot keep two real signals that fall in one bin --
    which is the trade the LD filter is making.
    """
    if bin_size <= 0.0:
        raise ValueError("the bin size must be positive")
    seen = {}
    kept = []
    for j in order:
        b = int(math.floor(positions[j] / bin_size))
        if b in seen:
            continue
        seen[b] = j
        kept.append(j)
    return kept


def _loglik(rss, n):
    """Gaussian log likelihood at the least-squares fit."""
    if rss <= 0.0:
        return float("inf")
    return -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)


def select_by_criterion(y, geno, candidates, covars=None, criterion="bic"):
    """Choose how many of the ranked candidates to keep.

    Fits the first k of them for k from one to all, scores each fit and
    takes the best. The criterion counts the pseudo-QTNs, as the paper
    writes it, and not the intercept or the fixed covariates -- those
    are in every model being compared, so they cannot separate them.
    """
    if criterion not in CRITERIA:
        raise ValueError("criterion must be one of %r" % (CRITERIA,))
    n = len(y)
    covars = [] if covars is None else covars
    if criterion == "none" or not candidates:
        return list(candidates), [], len(candidates)
    scores = []
    best = None
    best_k = 0
    for k in range(1, len(candidates) + 1):
        cols = [geno[q] for q in candidates[:k]]
        d = _design(n, covars, cols)
        if n <= len(d[0]):
            scores.append(float("inf"))
            continue
        try:
            fit = _w.ols(y, d)
        except Exception:
            scores.append(float("inf"))
            continue
        ll = _loglik(fit["rss"], n)
        pen = math.log(float(n)) if criterion == "bic" else 2.0
        s = 2.0 * (-ll) + k * pen
        scores.append(s)
        if best is None or s < best:
            best = s
            best_k = k
    return list(candidates[:best_k]), scores, best_k


def blink_gwas(y, geno, positions=None, covars=None, selection="ld",
               criterion="bic", ld_threshold=LD_THRESHOLD, alpha=ALPHA,
               bin_size=None, max_iter=10):
    """Iterate the scan and the pseudo-QTN selection to a fixed point.

    Parameters
    ----------
    y : sequence
        The phenotype.
    geno : sequence of sequences
        One row per marker, one column per individual.
    positions : sequence or None
        Genomic positions, needed only by the bin selection.
    covars : sequence of sequences or None
        Fixed covariates -- principal components, say -- carried in
        every model.
    selection : str
        A member of SELECTIONS.
    criterion : str
        A member of CRITERIA.
    ld_threshold : float
        Correlation above which a candidate is dropped.
    alpha : float
        The Bonferroni level for the pre-filter.
    bin_size : float or None
        Bin width for the bin selection.
    max_iter : int
        Iteration cap. Reaching it is reported, not hidden.

    Returns
    -------
    RichResult
        The final scan, the pseudo-QTNs and how they were chosen, the
        criterion path, the iteration count and whether the set settled,
        and the genomic inflation factor.

    References
    ----------
    Huang et al. (2019) GigaScience 8(2), giy154.
    """
    if selection not in SELECTIONS:
        raise ValueError("selection must be one of %r" % (SELECTIONS,))
    if criterion not in CRITERIA:
        raise ValueError("criterion must be one of %r" % (CRITERIA,))
    ys = [float(v) for v in y]
    n = len(ys)
    if n < 3:
        raise ValueError("need at least three individuals")
    g = [[float(v) for v in row] for row in geno]
    m = len(g)
    if m < 1:
        raise ValueError("need at least one marker")
    if any(len(row) != n for row in g):
        raise ValueError("every marker must have one value per individual")
    if selection == "bin":
        if positions is None:
            raise ValueError("the bin selection needs marker positions")
        positions = [float(v) for v in positions]
        if len(positions) != m:
            raise ValueError("positions must have one entry per marker")
        if bin_size is None:
            raise ValueError("the bin selection needs a bin size")
    cv = None if covars is None else [[float(v) for v in c] for c in covars]
    thr = float(alpha) / m

    qtn = []
    scan = None
    scores = []
    cand = []
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        scan = marker_scan(ys, g, cv, qtn)
        order = [j for j in _order_by_p(scan["p"]) if scan["p"][j] < thr]
        if selection == "ld":
            cand = ld_filter(g, order, float(ld_threshold))
        else:
            cand = bin_filter(order, positions, float(bin_size))
        new, scores, _ = select_by_criterion(ys, g, cand, cv, criterion)
        if new == qtn:
            converged = True
            qtn = new
            break
        qtn = new
    if not converged:
        # One last scan so the reported p values belong to the reported
        # pseudo-QTN set rather than to the previous one.
        scan = marker_scan(ys, g, cv, qtn)

    chi = sorted(t * t for t in scan["t"] if t == t)
    if chi:
        h = len(chi) // 2
        med = chi[h] if len(chi) % 2 else 0.5 * (chi[h - 1] + chi[h])
        lam = med / CHISQ1_MEDIAN
    else:
        lam = float("nan")

    sig = [j for j in range(m) if scan["p"][j] == scan["p"][j]
           and scan["p"][j] < thr]
    live_p = [v for v in scan["p"] if v == v]
    live_se = [v for v in scan["se"] if v == v]
    best_p = min(live_p) if live_p else float("nan")
    best_se = min(live_se) if live_se else float("nan")
    return RichResult(payload={
        "p": scan["p"],
        "beta": scan["beta"],
        "se": scan["se"],
        "t": scan["t"],
        "qtn": qtn,
        "candidates": cand,
        "criterion_path": scores,
        "n_qtn": len(qtn),
        "significant": sig,
        "n_significant": len(sig),
        "threshold": thr,
        "lambda_gc": lam,
        "iterations": it,
        "converged": converged,
        "estimate": best_p,
        "se_min": best_se,
        "n": n,
        "m": m,
        "selection": selection,
        "criterion": criterion,
        "ld_threshold": float(ld_threshold),
        "alpha": float(alpha),
        "method": "BLINK iterative fixed-effect GWAS",
    })


blinkg = blink_gwas


def cheatsheet():
    return ("blinkg: BLINK iterative fixed-effect GWAS. selections "
            + ", ".join(SELECTIONS) + "; criteria " + ", ".join(CRITERIA))
