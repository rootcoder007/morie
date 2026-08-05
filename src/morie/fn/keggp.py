# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""KEGG pathway over-representation analysis.

Kanehisa, Furumichi, Tanabe, Sato and Morishima (2017), "KEGG: new
perspectives on genomes, pathways, diseases and drugs", Nucleic Acids
Research 45(D1):D353-D361, doi:10.1093/nar/gkw1092, describes the
pathway maps; the enrichment test itself is the standard
over-representation (Fisher / hypergeometric) test.  With a universe of
N genes, a pathway containing m of them, a selected list of n genes and
q of those in the pathway,

    P(X >= q) = sum_{j=q}^{min(m,n)} C(m,j) C(N-m, n-j) / C(N, n),

which is the one-sided Fisher exact p-value for the 2x2 table.  The
tail sum is formed in log space through the log-gamma function, and the
p-values across pathways are adjusted by the Benjamini-Hochberg step-up
procedure (Benjamini and Hochberg 1995, JRSS-B 57(1):289-300,
doi:10.1111/j.2517-6161.1995.tb02031.x).

Membership is passed as indicator vectors over a fixed gene universe so
that the two language arms index genes identically.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kegg_pathway"]


def _lchoose(a, b):
    if b < 0 or b > a:
        return float("-inf")
    return core.lgamma(a + 1.0) - core.lgamma(b + 1.0) - core.lgamma(a - b + 1.0)


def _phyper_upper(q, m, N, n):
    """P(X >= q) for X hypergeometric with m successes in N, n drawn."""
    hi = min(m, n)
    if q <= max(0, n - (N - m)):
        return 1.0
    if q > hi:
        return 0.0
    den = _lchoose(N, n)
    tot = 0.0
    for j in range(int(q), int(hi) + 1):
        tot += math.exp(_lchoose(m, j) + _lchoose(N - m, n - j) - den)
    if tot > 1.0:
        tot = 1.0
    return tot


def kegg_pathway(genes, kegg_pathways, alpha=0.05):
    """Hypergeometric over-representation of a gene list in each pathway.

    Parameters
    ----------
    genes : array-like
        Indicator over the gene universe: 1 if the gene is in the
        selected list, 0 otherwise.
    kegg_pathways : array-like
        Membership matrix, one row per gene of the universe and one
        column per pathway, entries 0 or 1.
    alpha : float
        FDR level used for the reported decisions.
    """
    g = core.vec(genes)
    N = len(g)
    if N == 0:
        raise ValueError("kegg_pathway: genes is empty")
    for v in g:
        if v not in (0.0, 1.0):
            raise ValueError("kegg_pathway: genes must be a 0/1 indicator over the universe")
    P = core.mat(kegg_pathways)
    if len(P) != N:
        raise ValueError("kegg_pathway: kegg_pathways must have one row per gene of the universe")
    K = len(P[0])
    for row in P:
        for v in row:
            if v not in (0.0, 1.0):
                raise ValueError("kegg_pathway: kegg_pathways must be a 0/1 membership matrix")
    if not (0.0 < alpha < 1.0):
        raise ValueError("kegg_pathway: alpha must lie in (0, 1)")
    nsel = int(sum(g))
    if nsel == 0:
        raise ValueError("kegg_pathway: no genes selected")
    sizes = []
    ov = []
    pv = []
    for c in range(K):
        m = int(sum(P[i][c] for i in range(N)))
        q = int(sum(P[i][c] for i in range(N) if g[i] == 1.0))
        sizes.append(float(m))
        ov.append(float(q))
        pv.append(_phyper_upper(q, m, N, nsel))
    # Benjamini-Hochberg step-up
    order = sorted(range(K), key=lambda i: (pv[i], i))
    qv = [0.0] * K
    prev = 1.0
    for rank in range(K - 1, -1, -1):
        i = order[rank]
        val = pv[i] * K / (rank + 1.0)
        if val > prev:
            val = prev
        if val > 1.0:
            val = 1.0
        qv[i] = val
        prev = val
    best = min(range(K), key=lambda i: (pv[i], i))
    return RichResult(
        title="KEGG pathway over-representation",
        summary_lines=[("pathways", K), ("selected genes", nsel), ("min p", pv[best])],
        payload={
            "estimate": pv[best],
            "pvalue": pv,
            "qvalue": qv,
            "overlap": ov,
            "pathway_size": sizes,
            "top_pathway": float(best),
            "n_significant": float(sum(1 for v in qv if v <= alpha)),
            "significant": [1.0 if v <= alpha else 0.0 for v in qv],
            "n_selected": float(nsel),
            "n_pathways": float(K),
            "alpha": alpha,
            "n": N,
            "method": "hypergeometric P(X >= q) per pathway with Benjamini-Hochberg FDR, Kanehisa et al (2017)",
        },
    )


def cheatsheet():
    return "keggp: KEGG pathway enrichment"


# compact alias per ledger/NAMING.md
keggpathway = kegg_pathway
