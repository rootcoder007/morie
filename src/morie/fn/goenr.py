# SPDX-License-Identifier: AGPL-3.0-or-later
"""GO term enrichment by the hypergeometric (Fisher exact) upper tail."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["goenr", "go_enrichment"]


def _lchoose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0)
            - math.lgamma(n - k + 1.0))


def _hyper_upper(k, n, M, N):
    # P(X >= k) for X ~ Hypergeometric(N, M, n), computed as the
    # direct upper-tail sum of Boyle et al. (2004), p. 3711:
    #   P = 1 - sum_{i=0}^{k-1} C(M,i) C(N-M,n-i) / C(N,n).
    # Summed on the shorter tail in log space for stability.
    lo = max(0, n - (N - M))
    hi = min(n, M)
    if k <= lo:
        return 1.0
    if k > hi:
        return 0.0
    denom = _lchoose(N, n)
    upper = sum(math.exp(_lchoose(M, i) + _lchoose(N - M, n - i) - denom)
                for i in range(k, hi + 1))
    lower = sum(math.exp(_lchoose(M, i) + _lchoose(N - M, n - i) - denom)
                for i in range(lo, k))
    if upper <= lower:
        return min(1.0, upper)
    return min(1.0, max(0.0, 1.0 - lower))


def goenr(hits, list_size, term_size, background_size, correction="none"):
    """
    GO term enrichment p-values by the hypergeometric upper tail.

    For each GO node, with N genes in the background, M of them
    annotated (directly or indirectly) to the node, a study list of n
    genes of which k are annotated to the node, the p-value is

        P = 1 - sum_{i=0}^{k-1} C(M, i) C(N - M, n - i) / C(N, n),

    the probability of seeing k or more annotated genes by chance
    (one-tailed Fisher exact test). Optional Bonferroni correction
    multiplies each p-value by the number of terms tested and caps at
    1, as described in the same paper.

    Parameters
    ----------
    hits : int or array-like of int
        k, annotated genes in the study list, one entry per GO term.
    list_size : int
        n, size of the study gene list.
    term_size : int or array-like of int
        M, genes annotated to each term in the background.
    background_size : int
        N, total genes in the background distribution.
    correction : str
        "none" (default) or "bonferroni".

    Returns
    -------
    result : RichResult
        Keys: pvalue, padj, expected (n M / N), fold_enrichment
        ((k/n)/(M/N)), hits, term_size, n, N, method.

    References
    ----------
    Boyle, E. I., Weng, S., Gollub, J., Jin, H., Botstein, D.,
    Cherry, J. M. and Sherlock, G. (2004), "GO::TermFinder--open
    source software for accessing Gene Ontology information and
    finding significantly enriched Gene Ontology terms associated
    with a list of genes", Bioinformatics 20(18), 3710-3715.
    Hypergeometric upper-tail formula and Bonferroni correction from
    the Algorithm section, p. 3711. Source consulted:
    pmc.ncbi.nlm.nih.gov/articles/PMC3037731 (saved as
    library/pdf/fetched-wave3/Boyle-2004-GO-TermFinder-Bioinformatics.html).
    """
    k = np.atleast_1d(np.asarray(hits, dtype=float))
    M = np.atleast_1d(np.asarray(term_size, dtype=float))
    if len(M) == 1 and len(k) > 1:
        M = np.asarray([float(M[0])] * len(k))
    n = int(list_size)
    N = int(background_size)
    if len(M) != len(k):
        raise ValueError("hits and term_size must have equal length")
    if n > N:
        raise ValueError("list_size cannot exceed background_size")
    nt = len(k)
    pv = []
    for j in range(nt):
        kj, Mj = int(k[j]), int(M[j])
        if kj > min(n, Mj):
            raise ValueError("hits cannot exceed min(list_size, term_size)")
        pv.append(_hyper_upper(kj, n, Mj, N))
    pvalue = np.asarray(pv)
    if correction == "bonferroni":
        padj = np.clip(pvalue * float(nt), 0.0, 1.0)
    elif correction == "none":
        padj = pvalue
    else:
        raise ValueError("correction must be 'none' or 'bonferroni'")
    expected = np.asarray([float(n) * float(M[j]) / float(N)
                           for j in range(nt)])
    fold = np.asarray([
        (float(k[j]) / float(n)) / (float(M[j]) / float(N))
        if M[j] > 0 else float("nan") for j in range(nt)])
    return RichResult(payload={
        "pvalue": pvalue,
        "padj": padj,
        "expected": expected,
        "fold_enrichment": fold,
        "hits": k,
        "term_size": M,
        "n": n,
        "N": N,
        "method": "GO enrichment, hypergeometric upper tail (Boyle et al. 2004)",
    })


go_enrichment = goenr
goenrichment = goenr


def cheatsheet():
    return ("goenr(hits, list_size, term_size, background_size) -> "
            "one-tailed Fisher exact GO enrichment p-values.")
