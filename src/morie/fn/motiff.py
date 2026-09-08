"""Network motif significance (Milo et al. 2002)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["motiff", "network_motifs"]


def _triad_counts(adj, n):
    # counts of the 13 connected directed 3-node subgraph classes;
    # here we count the two most-used motifs explicitly (feed-forward
    # loop and 3-cycle) plus all connected triads, which is enough to
    # score any single motif the caller asks about.
    ff = 0        # feed-forward loop: i->j, i->k, j->k
    cyc = 0       # 3-cycle: i->j->k->i
    for i in range(n):
        for j in range(n):
            if j == i or not adj[i][j]:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if adj[i][k] and adj[j][k]:
                    ff += 1
                if adj[j][k] and adj[k][i]:
                    cyc += 1
    return {"ffl": ff, "cycle3": cyc // 3}


def _classify_edges(adj, n):
    # row-major: single = directed edges with no reciprocal; mutual =
    # bidirectional pairs stored once as [min, max].
    single = []
    mutual = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if i == j or not adj[i][j]:
                continue
            if adj[j][i]:
                key = (i, j) if i < j else (j, i)
                if key not in seen:
                    seen.add(key)
                    mutual.append([key[0], key[1]])
            else:
                single.append([i, j])
    return single, mutual


def _degree_preserving_shuffle(adj, n, rng, swaps, preserve_mutual):
    # Milo et al. (2002) / mfinder switching (their refs. 17, 18).
    # The default ensemble preserves each node's in-degree, out-degree
    # AND the network's mutual (bidirectional) edge count: single
    # edges only switch with single edges (and a switch that would
    # create a reciprocal is rejected), mutual pairs only switch with
    # mutual pairs.  So the mutual-edge count is invariant.  With
    # preserve_mutual=False only in/out degree is fixed (the weaker
    # Fig. 2 caption ensemble), which may create/destroy reciprocals.
    single, mutual = _classify_edges(adj, n)
    present = set()
    for i, j in single:
        present.add((i, j))
    for i, j in mutual:
        present.add((i, j))
        present.add((j, i))
    ns = len(single)
    nm = len(mutual)
    tot = ns + nm
    frac_m = nm / tot if tot else 0.0
    for _ in range(swaps):
        pool_mut = False
        if preserve_mutual:
            pool_mut = float(rng.uniform()) < frac_m
        ua = float(rng.uniform())
        ub = float(rng.uniform())
        if pool_mut:
            if nm < 2:
                continue
            a = int(ua * nm)
            b = int(ub * nm)
            if a == b:
                continue
            i1, j1 = mutual[a]
            i2, j2 = mutual[b]
            if len({i1, j1, i2, j2}) < 4:
                continue
            # new mutual pairs {i1,j2}, {i2,j1}; both directions free
            if ((i1, j2) in present or (j2, i1) in present or
                    (i2, j1) in present or (j1, i2) in present):
                continue
            present.discard((i1, j1)); present.discard((j1, i1))
            present.discard((i2, j2)); present.discard((j2, i2))
            present.add((i1, j2)); present.add((j2, i1))
            present.add((i2, j1)); present.add((j1, i2))
            mutual[a] = [min(i1, j2), max(i1, j2)]
            mutual[b] = [min(i2, j1), max(i2, j1)]
        else:
            if ns < 2:
                continue
            a = int(ua * ns)
            b = int(ub * ns)
            if a == b:
                continue
            i1, j1 = single[a]
            i2, j2 = single[b]
            if len({i1, j1, i2, j2}) < 4:
                continue
            # new directed edges i1->j2, i2->j1; reject if they exist
            # or if either would become a reciprocal (mutual) edge.
            if (i1, j2) in present or (i2, j1) in present:
                continue
            if preserve_mutual and \
                    ((j2, i1) in present or (j1, i2) in present):
                continue
            present.discard((i1, j1)); present.discard((i2, j2))
            present.add((i1, j2)); present.add((i2, j1))
            single[a] = [i1, j2]; single[b] = [i2, j1]
    new = [[0] * n for _ in range(n)]
    for i, j in present:
        new[i][j] = 1
    return new


def motiff(adjacency, motif="ffl", n_random=100, seed=0, swaps=None,
           preserve_mutual=True):
    """
    Network-motif significance by degree-preserving randomization.

    Milo et al. (2002): a network motif is a subgraph pattern that
    recurs significantly more often in the real network than in
    randomized networks that preserve each node's connectivity
    (their Fig. 2 and Methods).  The rigorous ensemble (their refs.
    17, 18, the mfinder switching algorithm) preserves each node's
    in-degree and out-degree AND the network's number of mutual
    (bidirectional) edges -- so a randomized network cannot invent or
    destroy reciprocal links, which would otherwise bias motifs made
    of mutual edges.  This is the default (preserve_mutual=True); set
    it False for the weaker Fig. 2-caption ensemble that fixes only
    in/out degree.  The significance is the Z score
    Z = (N_real - mean N_rand) / sd N_rand and the
    empirical p-value is the fraction of randomized networks with an
    equal-or-greater motif count (their P < 0.01 cutoff).  The
    feed-forward loop (i->j, i->k, j->k) and the 3-cycle are scored.

    Sources
    -------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N.,
    Chklovskii, D. & Alon, U. (2002). Network motifs: simple
    building blocks of complex networks. *Science*, 298(5594),
    824-827 (local copy fetched-wave3/Network motifs simple
    building blocks of complex networks.pdf).

    Parameters
    ----------
    adjacency : matrix (n x n)
        Directed adjacency (0/1, no self-loops used).
    motif : str
        "ffl" (feed-forward loop) or "cycle3".
    n_random : int
        Size of the randomized ensemble.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    swaps : int, optional
        Edge swaps per randomization (default 10 * #edges).
    preserve_mutual : bool
        True (default) preserves the mutual-edge count (mfinder,
        refs. 17-18); False fixes only in/out degree.

    Returns
    -------
    RichResult
        Keys: count (real), z_score, p_value, rand_mean, rand_sd.
    """
    A = [[1 if v else 0 for v in row] for row in adjacency]
    n = len(A)
    if any(len(r) != n for r in A) or n < 3:
        raise ValueError("adjacency must be square, n >= 3")
    if motif not in ("ffl", "cycle3"):
        raise ValueError("motif must be 'ffl' or 'cycle3'")
    real = _triad_counts(A, n)[motif]
    m_edges = sum(sum(r) for r in A)
    if swaps is None:
        swaps = 10 * max(m_edges, 1)
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(int(n_random)):
        Ar = _degree_preserving_shuffle(A, n, rng, int(swaps),
                                        bool(preserve_mutual))
        rand.append(_triad_counts(Ar, n)[motif])
    mu = sum(rand) / len(rand)
    var = sum((c - mu) ** 2 for c in rand) / (len(rand) - 1) \
        if len(rand) > 1 else 0.0
    sd = math.sqrt(var)
    z = (real - mu) / sd if sd > 0 else (0.0 if real == mu else
                                         math.inf)
    p = (sum(1 for c in rand if c >= real) + 1) / (len(rand) + 1)
    return RichResult(payload={
        "count": real,
        "z_score": z,
        "p_value": p,
        "rand_mean": mu,
        "rand_sd": sd,
        "motif": motif,
        "n_random": int(n_random),
        "seed": int(seed),
        "preserve_mutual": bool(preserve_mutual),
        "method": "Milo et al. (2002) motif Z score / p-value"
                  " (mfinder degree+mutual ensemble)"
                  if preserve_mutual else
                  "Milo et al. (2002) motif Z score / p-value"
                  " (in/out-degree ensemble)",
    })


# long descriptive alias (stub-era name)
network_motifs = motiff


def cheatsheet():
    return "motiff: Z = (N_real - mean N_rand)/sd; p = frac rand >= real"

# public names resolved by fn/_lazy_map.json
motif_count = motiff
motifcount = motiff
