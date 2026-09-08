"""Variant quality filtering: GATK hard filters and a VQSR-style recalibration.

A variant caller emits everything it can defend and leaves the deciding
to you. Two ways of deciding are implemented here, and they fail in
opposite directions, which is why both are present.

HARD FILTERING applies a fixed threshold to each annotation
independently. It is transparent, needs no training data, and is what
you use when there are too few variants to fit anything. Its weakness is
that it treats the annotations as independent: a variant slightly over
the line on two annotations at once is no worse off than one slightly
over on a single annotation, even though the joint evidence against it
is far stronger.

The GATK recommendations are the defaults, and they differ between SNPs
and indels because the annotations do:

  SNP    QD < 2.0, QUAL < 30.0, SOR > 3.0, FS > 60.0, MQ < 40.0,
         MQRankSum < -12.5, ReadPosRankSum < -8.0
  INDEL  QD < 2.0, QUAL < 30.0, FS > 200.0, SOR > 10.0,
         ReadPosRankSum < -20.0

The indel thresholds on FS and SOR are far looser because indel calls
carry more strand and position skew for reasons that are not artefacts
-- applying the SNP numbers to indels throws away real variants, which
is the single most common way this gets done wrong.

VQSR fits the joint distribution instead. A Gaussian mixture is trained
on variants believed good, a second on variants believed bad, and each
variant is scored by the log ratio of the two densities -- VQSLOD.
Because it is a joint model it can accept a variant that is marginal on
one annotation and excellent on the others, which is exactly the case
hard filtering handles worst. Its weakness is the mirror image: it needs
a training set, and a bad one produces a confident, wrong ranking.

Tranches turn the score into a decision. Sort the TRAINING-POSITIVE
variants by VQSLOD; the threshold that retains 99% of them defines the
99.0 tranche. A tranche is therefore a statement about sensitivity to
known variants, not about the number of calls kept, and reading it as
the latter is a mistake worth naming.

The mixture is fitted by EM with either full or diagonal covariances.
Diagonal is not merely cheaper: with a handful of training variants a
full covariance is singular, and the module says so rather than
inverting something it should not.

References
  Broad Institute, GATK Best Practices: "Hard-filtering germline short
    variants," which is the source of the threshold values above.
  DePristo, M.A., Banks, E., Poplin, R., Garimella, K.V., Maguire, J.R.,
    Hartl, C., Philippakis, A.A. et al. (2011) "A framework for
    variation discovery and genotyping using next-generation DNA
    sequencing data." Nature Genetics 43(5), 491-498. The recalibration
    framework.
  Van der Auwera, G.A., Carneiro, M.O., Hartl, C., Poplin, R., del
    Angel, G., Levy-Moonshine, A. et al. (2013) "From FastQ data to
    high-confidence variant calls: the Genome Analysis Toolkit best
    practices pipeline." Current Protocols in Bioinformatics 43,
    11.10.1-11.10.33.
  Dempster, A.P., Laird, N.M. and Rubin, D.B. (1977) "Maximum likelihood
    from incomplete data via the EM algorithm." JRSS B 39(1), 1-38.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["vcf_filter", "varqc1", "hard_filter", "fit_mixture",
           "mixture_logpdf", "DEFAULT_THRESHOLDS", "METHODS",
           "COVARIANCES", "cheatsheet"]

METHODS = ("hard", "vqsr", "both")
COVARIANCES = ("full", "diagonal")

# Each entry is (annotation, direction, cutoff). Direction "lt" means a
# variant FAILS when the value is below the cutoff.
DEFAULT_THRESHOLDS = {
    "snp": (("QD", "lt", 2.0), ("QUAL", "lt", 30.0), ("SOR", "gt", 3.0),
            ("FS", "gt", 60.0), ("MQ", "lt", 40.0),
            ("MQRankSum", "lt", -12.5), ("ReadPosRankSum", "lt", -8.0)),
    "indel": (("QD", "lt", 2.0), ("QUAL", "lt", 30.0),
              ("FS", "gt", 200.0), ("SOR", "gt", 10.0),
              ("ReadPosRankSum", "lt", -20.0)),
}


def hard_filter(records, fields, thresholds):
    """Apply each threshold independently; return one FILTER per record.

    A missing annotation does NOT fail its filter. A caller omits an
    annotation when it could not be computed, and treating "not
    measured" as "measured and bad" would discard exactly the variants
    with unusual read support -- which is where real novel variation
    lives.
    """
    pos = {}
    for j, f in enumerate(fields):
        pos[f] = j
    out = []
    counts = {}
    for rec in records:
        failed = []
        for (ann, direction, cut) in thresholds:
            if ann not in pos:
                continue
            v = rec[pos[ann]]
            if v is None or v != v:
                continue
            bad = v < cut if direction == "lt" else v > cut
            if bad:
                failed.append("%s%s%g" % (ann, "<" if direction == "lt"
                                          else ">", cut))
        if failed:
            out.append(";".join(failed))
        else:
            out.append("PASS")
        counts[out[-1]] = counts.get(out[-1], 0) + 1
    return out, counts


def mixture_logpdf(x, weights, means, chols):
    """log sum_k w_k N(x; mu_k, L_k L_k'), from the Cholesky factors."""
    d = len(x)
    terms = []
    for k in range(len(weights)):
        L = chols[k]
        z = [0.0] * d
        for i in range(d):
            z[i] = (x[i] - means[k][i] - _w.dot(L[i][:i], z[:i])) / L[i][i]
        logdet = 2.0 * _w.csum(math.log(L[i][i]) for i in range(d))
        q = _w.csum(v * v for v in z)
        terms.append(math.log(weights[k]) - 0.5 * q - 0.5 * logdet
                     - 0.5 * d * math.log(2.0 * math.pi))
    return _w.logsumexp(terms)


def fit_mixture(X, n_components=2, n_iter=50, seed=1, covariance="full",
                min_variance=1e-6, jitter=1e-8, shrinkage=0.05):
    """EM for a Gaussian mixture, returning Cholesky factors.

    Initialised by assigning point i to component i mod K -- a
    deterministic split, not a random one, so the fit does not move with
    the seed unless the seed is actually used. The seed is retained
    because a jittered restart is the usual escape from a degenerate
    component and it must be reproducible when it happens.

    Each component covariance is SHRUNK towards the diagonal of the
    whole training set's covariance:

        S_k <- (1 - lambda) S_k + lambda diag(S_all)

    That is not cosmetic. With a handful of training variants in seven
    dimensions the per-component covariance is very nearly singular, its
    Cholesky has a diagonal entry near zero, and the quadratic form in
    the density blows up -- which turns a VQSLOD, a log RATIO that
    should sit in single digits, into something of order 1e9 and makes
    the last bits of the fit decide the answer. An absolute variance
    floor does not help, because it is a floor in the annotation's own
    units and every annotation here has a different scale. Shrinking
    towards the data's own diagonal is scale-free and is the standard
    remedy.
    """
    if covariance not in COVARIANCES:
        raise ValueError("covariance must be one of %r" % (COVARIANCES,))
    n = len(X)
    d = len(X[0])
    K = int(n_components)
    if K < 1:
        raise ValueError("need at least one component")
    if n < K * (d + 1):
        raise ValueError("too few training variants for %d components in "
                         "%d dimensions; use a diagonal covariance or "
                         "fewer components" % (K, d))
    rng = _core._SplitMix64(seed)
    gmean = [_w.csum(X[i][j] for i in range(n)) / n for j in range(d)]
    gvar = [_w.csum((X[i][j] - gmean[j]) * (X[i][j] - gmean[j])
                    for i in range(n)) / n for j in range(d)]
    for j in range(d):
        if gvar[j] < min_variance:
            gvar[j] = min_variance
    lam = float(shrinkage)
    if not (0.0 <= lam < 1.0):
        raise ValueError("shrinkage must lie in [0, 1)")
    resp = [[1.0 if (i % K) == k else 0.0 for k in range(K)]
            for i in range(n)]
    weights = [1.0 / K] * K
    means = [[0.0] * d for _ in range(K)]
    chols = [[[0.0] * d for _ in range(d)] for _ in range(K)]
    ll_trace = []
    for _ in range(int(n_iter)):
        # M step
        for k in range(K):
            nk = _w.csum(resp[i][k] for i in range(n))
            if nk <= 1e-12:
                # A component that lost every point is restarted at a
                # jittered overall mean rather than left singular.
                nk = 1e-12
                means[k] = [_w.csum(X[i][j] for i in range(n)) / n
                            + 1e-3 * float(rng.normal()) for j in range(d)]
                cov = [[(gvar[a] if a == b else 0.0) for b in range(d)]
                       for a in range(d)]
            else:
                means[k] = [_w.csum(resp[i][k] * X[i][j]
                                    for i in range(n)) / nk
                            for j in range(d)]
                cov = [[0.0] * d for _ in range(d)]
                for a in range(d):
                    for b in range(d):
                        if covariance == "diagonal" and a != b:
                            continue
                        cov[a][b] = _w.csum(
                            resp[i][k] * (X[i][a] - means[k][a])
                            * (X[i][b] - means[k][b])
                            for i in range(n)) / nk
                for a in range(d):
                    for b in range(d):
                        cov[a][b] *= (1.0 - lam)
                    cov[a][a] += lam * gvar[a]
                    if cov[a][a] < min_variance:
                        cov[a][a] = min_variance
                    cov[a][a] += jitter
            weights[k] = nk / n
            chols[k] = _w.chol(cov)
        s = _w.csum(weights)
        weights = [v / s for v in weights]
        # E step
        ll = 0.0
        for i in range(n):
            lp = []
            for k in range(K):
                lp.append(mixture_logpdf(X[i], [weights[k]], [means[k]],
                                         [chols[k]]))
            tot = _w.logsumexp(lp)
            ll += tot
            resp[i] = [math.exp(v - tot) for v in lp]
        ll_trace.append(ll)
    return {"weights": weights, "means": means, "chols": chols,
            "loglik": ll_trace[-1], "loglik_trace": ll_trace,
            "covariance": covariance, "n_components": K}


def vcf_filter(vcf, thresholds=None, fields=None, mode="snp",
               method="hard", positive=None, negative=None,
               n_components=2, n_iter=50, seed=1, covariance="full",
               tranches=(90.0, 99.0, 99.9, 100.0), vqsr_fields=None,
               min_variance=1e-6, shrinkage=0.05):
    """Filter variant records by hard thresholds, by VQSR, or by both.

    Parameters
    ----------
    vcf : sequence of sequences
        One row per variant, one column per annotation.
    thresholds : sequence or None
        Triples (annotation, "lt" or "gt", cutoff). The GATK
        recommendations for `mode` when omitted.
    fields : sequence
        Column names. Required.
    mode : str
        "snp" or "indel", which selects the default thresholds.
    method : str
        "hard", "vqsr" or "both".
    positive, negative : sequences of int or None
        Row indices of the training-positive and training-negative sets.
        Required by the VQSR routes.
    n_components, n_iter, seed, covariance, min_variance, shrinkage :
        Mixture settings. `shrinkage` pulls each component covariance
        towards the diagonal of the training set's own covariance, which
        is what keeps a seven-dimensional fit on twenty variants from
        being singular.
    tranches : sequence
        Target sensitivities to the training-positive set, in percent.
    vqsr_fields : sequence or None
        Which annotations the mixture uses. All of them by default.

    Returns
    -------
    RichResult
        Per-record FILTER strings and counts, and for the VQSR routes
        the VQSLOD score, the tranche each record falls in, and the
        fitted mixtures.

    References
    ----------
    GATK Best Practices hard-filtering recommendations; DePristo et al.
    (2011) Nature Genetics 43(5), 491-498; Van der Auwera et al. (2013)
    Curr. Protoc. Bioinformatics 43, 11.10.1-11.10.33.
    """
    if method not in METHODS:
        raise ValueError("method must be one of %r" % (METHODS,))
    if mode not in DEFAULT_THRESHOLDS:
        raise ValueError("mode must be snp or indel")
    if fields is None:
        raise ValueError("fields (the column names) is required")
    fields = [str(f) for f in fields]
    recs = [[None if v is None else float(v) for v in row] for row in vcf]
    n = len(recs)
    if n < 1:
        raise ValueError("no records")
    thr = DEFAULT_THRESHOLDS[mode] if thresholds is None else \
        tuple((str(a), str(b), float(c)) for (a, b, c) in thresholds)

    hard, counts = hard_filter(recs, fields, thr)
    res = {"filter": list(hard), "counts": counts,
           "n": n, "mode": mode, "method": method,
           "thresholds": [[a, b, c] for (a, b, c) in thr],
           "n_pass_hard": sum(1 for v in hard if v == "PASS"),
           "fields": fields,
           "method_name": "variant quality filtering"}

    if method == "hard":
        res["estimate"] = res["n_pass_hard"] / float(n)
        res["se"] = math.sqrt(res["estimate"] * (1.0 - res["estimate"])
                              / n)
        return RichResult(payload=res)

    if positive is None or negative is None:
        raise ValueError("the VQSR routes need positive and negative "
                         "training indices")
    use = fields if vqsr_fields is None else [str(f) for f in vqsr_fields]
    cols = [fields.index(f) for f in use]
    for i in range(n):
        for c in cols:
            if recs[i][c] is None or recs[i][c] != recs[i][c]:
                raise ValueError("record %d is missing annotation %s, "
                                 "which the mixture cannot use; drop the "
                                 "record or the annotation"
                                 % (i, fields[c]))
    X = [[recs[i][c] for c in cols] for i in range(n)]
    good = fit_mixture([X[i] for i in positive], n_components, n_iter,
                       seed, covariance, min_variance, shrinkage=shrinkage)
    bad = fit_mixture([X[i] for i in negative], n_components, n_iter,
                      seed, covariance, min_variance, shrinkage=shrinkage)

    ln10 = math.log(10.0)
    lod = [(mixture_logpdf(X[i], good["weights"], good["means"],
                           good["chols"])
            - mixture_logpdf(X[i], bad["weights"], bad["means"],
                             bad["chols"])) / ln10 for i in range(n)]

    # Tranches: the VQSLOD cut that retains the stated percentage of the
    # TRAINING-POSITIVE variants. A tranche is a sensitivity to known
    # variants, not a count of calls kept.
    ps = sorted((lod[i] for i in positive), reverse=True)
    m = len(ps)
    cuts = []
    for t in tranches:
        keep = int(math.ceil(float(t) / 100.0 * m))
        if keep < 1:
            keep = 1
        if keep > m:
            keep = m
        cuts.append((float(t), ps[keep - 1]))
    cuts.sort(key=lambda p: (-p[1], p[0]))

    tranche = []
    for i in range(n):
        lab = "FAIL"
        for (t, c) in cuts:
            if lod[i] >= c:
                lab = "%.1f" % t
                break
        tranche.append(lab)

    res["vqslod"] = lod
    res["tranche"] = tranche
    res["tranche_cuts"] = [[t, c] for (t, c) in cuts]
    res["good_model"] = {"weights": good["weights"],
                         "means": good["means"],
                         "loglik": good["loglik"]}
    res["bad_model"] = {"weights": bad["weights"], "means": bad["means"],
                        "loglik": bad["loglik"]}
    res["good_loglik_trace"] = good["loglik_trace"]
    res["bad_loglik_trace"] = bad["loglik_trace"]
    res["covariance"] = covariance
    res["vqsr_fields"] = use

    if method == "both":
        combined = []
        for i in range(n):
            if hard[i] != "PASS":
                combined.append(hard[i])
            elif tranche[i] == "FAIL":
                combined.append("VQSRFail")
            else:
                combined.append("PASS")
        res["filter"] = combined
        cc = {}
        for v in combined:
            cc[v] = cc.get(v, 0) + 1
        res["counts"] = cc
    else:
        res["filter"] = ["PASS" if tranche[i] != "FAIL" else "VQSRFail"
                         for i in range(n)]
        cc = {}
        for v in res["filter"]:
            cc[v] = cc.get(v, 0) + 1
        res["counts"] = cc

    res["n_pass"] = sum(1 for v in res["filter"] if v == "PASS")
    res["estimate"] = res["n_pass"] / float(n)
    res["se"] = math.sqrt(res["estimate"] * (1.0 - res["estimate"]) / n)
    return RichResult(payload=res)


varqc1 = vcf_filter


def cheatsheet():
    return ("varqc1: variant quality filtering. methods "
            + ", ".join(METHODS) + "; covariances "
            + ", ".join(COVARIANCES))


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
vcffilter = vcf_filter
