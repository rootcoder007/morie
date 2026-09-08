# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gene-set enrichment analysis (GSEA) running-sum statistic."""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._rrng_core import RRandom

__all__ = ["gnsetenr", "geneset_enrichment"]


def _es(r_sorted, member_sorted, p):
    # Appendix, "Enrichment Score ES(S)": walking down the ranked list,
    #   Phit(S, i) = sum_{g_j in S, j <= i} |r_j|^p / N_R,
    #       N_R = sum_{g_j in S} |r_j|^p,
    #   Pmiss(S, i) = sum_{g_j not in S, j <= i} 1 / (N - N_H),
    # ES(S) is the maximum deviation from zero of Phit - Pmiss.
    n = len(r_sorted)
    nh = sum(1 for m in member_sorted if m)
    if nh == 0 or nh == n:
        raise ValueError("gene set must be a proper nonempty subset")
    nr = sum(abs(r_sorted[i]) ** p for i in range(n) if member_sorted[i])
    miss_w = 1.0 / float(n - nh)
    run = 0.0
    best = 0.0
    best_i = 0
    running = []
    for i in range(n):
        if member_sorted[i]:
            if nr > 0.0:
                run += abs(r_sorted[i]) ** p / nr
            else:
                run += 1.0 / float(nh)
        else:
            run -= miss_w
        running.append(run)
        if abs(run) > abs(best):
            best = run
            best_i = i
    return best, best_i, running


def gnsetenr(correlations, in_set, p=1.0, nperm=0, seed=None):
    """
    Gene-set enrichment analysis enrichment score (GSEA).

    Genes are ranked by decreasing correlation r_j with the phenotype
    to form the list L. Walking down L, a running sum increases by
    the absolute value of r_j raised to the power p, normalized by
    N_R = sum over the set of the same weights, when gene j belongs to
    the set S, and decreases by 1/(N - N_H) otherwise (N_H genes in
    S). The enrichment score ES(S) is the maximum deviation from zero
    of the running sum, a weighted Kolmogorov-Smirnov-like statistic;
    p = 0 reduces it to the standard Kolmogorov-Smirnov statistic and
    p = 1 (the paper's default) weights hits by their correlation.

    If nperm > 0, a nominal p-value is estimated by permuting the
    gene labels of the set (the "gene set" permutation used for
    pre-ranked lists; the paper's primary recommendation, phenotype
    permutation, requires the full expression matrix which this
    ranked-list interface does not carry). Positive and negative
    scores are assessed separately against the same-sign side of the
    permutation null, as prescribed in the paper.

    Parameters
    ----------
    correlations : array-like
        r_j for every gene (any order; sorted descending internally).
    in_set : array-like of bool/0-1
        Membership of each gene in the set S, aligned to correlations.
    p : float
        Weighting exponent (paper default 1).
    nperm : int
        Number of gene-label permutations (0 = no p-value).
    seed : int or None
        Seed for the R-compatible Mersenne-Twister stream.

    Returns
    -------
    result : RichResult
        Keys: es, arg_es (0-based rank at which the extreme deviation
        occurs), running (running-sum vector along the ranked list),
        n, n_hits, pvalue (if nperm > 0), nperm, method.

    References
    ----------
    Subramanian, A., Tamayo, P., Mootha, V. K., Mukherjee, S., Ebert,
    B. L., Gillette, M. A., Paulovich, A., Pomeroy, S. L., Golub,
    T. R., Lander, E. S. and Mesirov, J. P. (2005), "Gene set
    enrichment analysis: A knowledge-based approach for interpreting
    genome-wide expression profiles", PNAS 102(43), 15545-15550.
    ES from the Appendix, "Enrichment Score ES(S)" (p. 15550);
    sign-separated significance from p. 15546. Local source:
    library/pdf/fetched-wave3/Subramanian-2005-GSEA-PNAS.pdf.
    """
    r = np.atleast_1d(np.asarray(correlations, dtype=float))
    mem = np.atleast_1d(np.asarray(in_set, dtype=float))
    n = len(r)
    if len(mem) != n:
        raise ValueError("correlations and in_set must have equal length")
    order = sorted(range(n), key=lambda i: (-r[i], i))
    r_s = [float(r[i]) for i in order]
    m_s = [mem[i] != 0.0 for i in order]
    nh = sum(1 for m in m_s if m)
    es, arg_es, running = _es(r_s, m_s, float(p))
    payload = {
        "es": es,
        "arg_es": arg_es,
        "running": np.asarray(running),
        "n": n,
        "n_hits": nh,
        "nperm": int(nperm),
        "method": "GSEA enrichment score (Subramanian et al. 2005)",
    }
    if nperm > 0:
        rng = RRandom(seed)
        same_sign = 0
        as_extreme = 0
        for _ in range(int(nperm)):
            idx = rng.sample_int(n, size=nh)
            pm = [False] * n
            for j in idx:
                pm[j - 1] = True
            es_p, _, _ = _es(r_s, pm, float(p))
            if (es >= 0.0 and es_p >= 0.0) or (es < 0.0 and es_p < 0.0):
                same_sign += 1
                if abs(es_p) >= abs(es):
                    as_extreme += 1
        payload["pvalue"] = (float(as_extreme) / float(same_sign)
                             if same_sign > 0 else float("nan"))
    return RichResult(payload=payload)


geneset_enrichment = gnsetenr


def cheatsheet():
    return ("gnsetenr(correlations, in_set, p=1, nperm=0, seed=None) -> "
            "GSEA weighted KS enrichment score with sign-separated "
            "permutation p-value.")
