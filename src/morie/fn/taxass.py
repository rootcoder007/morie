# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kraken-style taxonomic classification by RTL path scoring."""

from ._richresult import RichResult

__all__ = ["taxass", "taxonomic_assignment"]


def _path_to_root(node, parent):
    path = [node]
    while parent.get(node, node) != node:
        node = parent[node]
        path.append(node)
    return path  # node ... root


def _lca(a, b, parent):
    pa = set(_path_to_root(a, parent))
    x = b
    while x not in pa:
        x = parent[x]
    return x


def taxass(kmer_taxa, parent):
    """
    Classify a sequence from its k-mer taxon hits, Kraken style.

    Each k-mer of the query maps (via the database) to the lowest
    common ancestor taxon of the genomes containing it; unmapped
    k-mers are ignored. The hit taxa and their ancestors form a
    pruned classification tree in which each node is weighted by its
    number of k-mer hits. Every root-to-leaf (RTL) path is scored by
    the sum of node weights along the path, and the sequence is
    assigned the leaf of the maximum-scoring RTL path; if several
    paths tie, the LCA of their leaves is used. A query with no hit
    k-mers is left unclassified (returned taxon 0). Kraken 2 applies
    exactly this classification step to minimizer-based LCA hits.

    Parameters
    ----------
    kmer_taxa : sequence of int
        Taxon id hit by each k-mer (0 = not found in the database).
    parent : dict
        Maps taxon id to parent id; the root maps to itself.

    Returns
    -------
    result : RichResult
        Keys: taxon (0 if unclassified), leaf_scores (leaf -> RTL
        score), weights (taxon -> hit count), n_kmers, n_hit, method.

    References
    ----------
    Wood, D. E. and Salzberg, S. L. (2014), "Kraken: ultrafast
    metagenomic sequence classification using exact alignments",
    Genome Biology 15(3), R46. Classification algorithm (RTL path
    scoring, tie -> LCA of maximal leaves), section "Sequence
    classification algorithm", p. 8 and Figure 1. Wood, D. E., Lu,
    J. and Langmead, B. (2019), "Improved metagenomic analysis with
    Kraken 2", Genome Biology 20, 257, Methods ("the leaf of the
    maximally scoring root-to-leaf path"). Local sources:
    library/pdf/fetched-wave3/Wood-Salzberg-2014-Kraken1-GenomeBiology.pdf,
    library/pdf/fetched-wave3/Wood-Lu-Langmead-2019-Kraken2-GenomeBiology.pdf.
    """
    hits = [int(t) for t in kmer_taxa]
    par = {int(k): int(v) for k, v in parent.items()}
    for k, v in par.items():
        if v != k and v not in par:
            raise ValueError("parent map is missing taxon %d" % v)
    weights = {}
    for t in hits:
        if t == 0:
            continue
        if t not in par:
            raise ValueError("hit taxon %d not in parent map" % t)
        weights[t] = weights.get(t, 0) + 1
    n_hit = sum(weights.values())
    if n_hit == 0:
        return RichResult(payload={
            "taxon": 0, "leaf_scores": {}, "weights": {},
            "n_kmers": len(hits), "n_hit": 0,
            "method": "Kraken RTL-path classification (Wood-Salzberg 2014)",
        })
    # classification tree: hit taxa plus their ancestors
    tree = set()
    for t in weights:
        tree.update(_path_to_root(t, par))
    children = {t: 0 for t in tree}
    for t in tree:
        p = par.get(t, t)
        if p != t and p in children:
            children[p] += 1
    leaves = sorted(t for t in tree if children[t] == 0)
    scores = {}
    for leaf in leaves:
        scores[leaf] = sum(weights.get(t, 0)
                           for t in _path_to_root(leaf, par))
    best = max(scores.values())
    top = [leaf for leaf in leaves if scores[leaf] == best]
    label = top[0]
    for other in top[1:]:
        label = _lca(label, other, par)
    return RichResult(payload={
        "taxon": label,
        "leaf_scores": scores,
        "weights": weights,
        "n_kmers": len(hits),
        "n_hit": n_hit,
        "method": "Kraken RTL-path classification (Wood-Salzberg 2014)",
    })


taxonomic_assignment = taxass


def cheatsheet():
    return ("taxass(kmer_taxa, parent) -> Kraken root-to-leaf path "
            "scoring; ties resolve to the LCA of maximal leaves.")
