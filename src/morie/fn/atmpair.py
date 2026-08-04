# morie.fn -- function file (rootcoder007/morie)
"""Atom-pair fingerprint."""

from ._richresult import RichResult

__all__ = ["atompairfp", "atom_pair_fp"]


def atompairfp(adjacency, atomtype, nbits=2048, maxdist=30):
    """Carhart atom-pair descriptors folded into a bit vector.

    An atom pair is the triple (type of atom i, type of atom j,
    topological distance between them), where the distance is the number
    of bonds on the shortest path.  A molecule is described by the
    multiset of all such triples over its heavy atoms, which is
    substructure-free -- nothing has to be enumerated in advance -- and
    invariant to how the atoms happen to be numbered, because the two
    types are written in canonical order.

    Descriptors are folded into a fixed-width bit vector by a hash; the
    hash used here is the deterministic

        h = ((ta * 1000003 + tb) * 1000033 + dist) mod nbits

    with (ta, tb) sorted.  Any injective-enough integer hash serves; this
    one is fixed so the two arms agree bit for bit.

    Parameters
    ----------
    adjacency : array-like, shape (a, a)
        Bond adjacency; non-zero means bonded.
    atomtype : array-like
        Integer atom type per atom.
    nbits : int
        Width of the folded fingerprint.
    maxdist : int
        Longest topological distance kept.

    Returns
    -------
    RichResult
        ``bits``, ``count``, ``nset``, ``npairs``, ``distance``,
        ``a``, ``nbits``.

    References
    ----------
    Carhart, R. E., Smith, D. H. and Venkataraghavan, R. (1985), "Atom
    pairs as molecular features in structure-activity studies: definition
    and applications", Journal of Chemical Information and Computer
    Sciences 25(2), 64-73, which defines an atom pair as
    <atom descriptor i>-<distance>-<atom descriptor j> with the distance
    counted in bonds along the shortest path.  Standard published form;
    the JCICS article is paywalled and was not read for this
    implementation, and the folding hash is this implementation's own
    choice, stated above rather than attributed.
    """
    A = [list(r) for r in adjacency]
    a = len(A)
    if any(len(r) != a for r in A):
        raise ValueError("adjacency must be square")
    t = [int(v) for v in atomtype]
    if len(t) != a:
        raise ValueError("atomtype must have one entry per atom")
    nbits = int(nbits)
    if nbits < 1:
        raise ValueError("nbits must be positive")
    INF = -1
    D = []
    for s in range(a):
        dist = [INF] * a
        dist[s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in range(a):
                    if A[u][v] and dist[v] == INF:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        D.append(dist)
    bits = [0] * nbits
    cnt = [0] * nbits
    npairs = 0
    dists = []
    for i in range(a):
        for j in range(i + 1, a):
            dd = D[i][j]
            if dd == INF or dd > int(maxdist):
                continue
            ta, tb = (t[i], t[j]) if t[i] <= t[j] else (t[j], t[i])
            h = ((ta * 1000003 + tb) * 1000033 + dd) % nbits
            bits[h] = 1
            cnt[h] += 1
            npairs += 1
            dists.append(dd)
    return RichResult(payload={
        "bits": bits, "count": cnt, "nset": sum(bits), "npairs": npairs,
        "distance": dists, "a": a, "nbits": nbits,
        "method": "Atom-pair fingerprint (Carhart et al. 1985)"})


atom_pair_fp = atompairfp


def cheatsheet():
    return "atmpair: Atom-pair fingerprint."
