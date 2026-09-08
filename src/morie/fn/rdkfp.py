# SPDX-License-Identifier: AGPL-3.0-or-later
"""RDKit path/subgraph-based topological fingerprint."""

from ._richresult import RichResult

from .ecfp4 import _bonds, _mix

__all__ = ["rdkfp", "rdkit_path_fp", "rdkitpathfp"]


def _subgraphs(a, bonds, minpath, maxpath, branched):
    """Connected bond subsets of size minpath..maxpath.

    ``branched`` true enumerates every connected edge subset (RDKit's
    ``findAllSubgraphsOfLengthsMtoN``); false enumerates only linear bond
    paths (``findAllPathsOfLengthsMtoN``).  Returned in a canonical order
    -- by size, then by the sorted bond-index tuple -- so both arms walk
    them identically.
    """
    nb = len(bonds)
    touch = [[] for _ in range(a)]
    for bi, (i, j, _o) in enumerate(bonds):
        touch[i].append(bi)
        touch[j].append(bi)

    out = []
    if branched:
        cur = {(bi,) for bi in range(nb)}
        for size in range(1, int(maxpath) + 1):
            if size >= int(minpath):
                out.extend(sorted(cur))
            if size == int(maxpath):
                break
            nxt = set()
            for s in cur:
                atoms = set()
                for bi in s:
                    atoms.add(bonds[bi][0])
                    atoms.add(bonds[bi][1])
                for at in atoms:
                    for bi in touch[at]:
                        if bi not in s:
                            nxt.add(tuple(sorted(s + (bi,))))
            cur = nxt
            if not cur:
                break
    else:
        # linear bond paths: grow at either end, no atom revisited
        def walk(path, ends):
            if len(path) >= int(minpath):
                out.append(tuple(sorted(path)))
            if len(path) == int(maxpath):
                return
            for k in (0, 1):
                at = ends[k]
                for bi in touch[at]:
                    if bi in path:
                        continue
                    i, j, _o = bonds[bi]
                    other = j if i == at else i
                    used = set()
                    for b in path:
                        used.add(bonds[b][0])
                        used.add(bonds[b][1])
                    if other in used:
                        continue
                    ne = list(ends)
                    ne[k] = other
                    walk(path + (bi,), tuple(ne))

        seen = set()
        for bi in range(nb):
            walk((bi,), (bonds[bi][0], bonds[bi][1]))
        res = []
        for p in out:
            if p not in seen:
                seen.add(p)
                res.append(p)
        out = sorted(res, key=lambda p: (len(p), p))
    return out


def rdkfp(adjacency, atomnum, aromatic=None, nbits=2048, minpath=1, maxpath=7,
          branched=True, use_bond_order=True):
    """RDKit path-based (subgraph) topological fingerprint.

    Every connected subgraph of between ``minpath`` and ``maxpath`` bonds
    is enumerated and reduced to one integer feature, which is folded into
    the bit vector.  The reduction is the RDKit one, reproduced here step
    for step:

    * The atom invariant is ``(atomic number mod 128) * 2 + aromatic``
      (RDKitFPGenerator.cpp lines 49-51).
    * For each bond in the subgraph, count how many other bonds of the
      same subgraph share an atom with it (``bondNbrs``), take the two
      end-atom invariants and the two in-subgraph atom degrees, and order
      the pair so that the larger invariant comes first, breaking a tie on
      the larger degree.  Hash (bondNbrs, bond order, invariant 1,
      degree 1, invariant 2, degree 2) into one bond hash
      (FingerprintUtil.cpp lines 386-444).
    * Sort the bond hashes of the subgraph, append the number of distinct
      atoms it covers -- this is what distinguishes cyclopropane from
      isobutane -- and hash the resulting sequence into the feature
      (RDKitFPGenerator.cpp lines 233-245).  A one-bond subgraph uses its
      single bond hash directly.

    Two deliberate departures, both stated rather than hidden.  The hash
    is the closed form h <- (h * 1000003 + v) mod (2^31 - 1) rather than
    ``boost::hash_combine``, so that Python and both R arms agree exactly
    without 32-bit unsigned arithmetic; bit *indices* are therefore this
    implementation's own while the feature *partition* is RDKit's.  And
    one bit is set per feature: RDKit's default ``numBitsPerFeature`` of 2
    draws the extra bit from a Boost random generator seeded by the
    feature, which cannot be reproduced outside Boost.

    Molecules arrive as a pre-parsed graph: ``adjacency`` is a symmetric
    bond-order matrix with 0 no bond, 1 single, 2 double, 3 triple and
    4 aromatic (this encoding is this implementation's own).  SMILES
    parsing and aromaticity perception are out of scope.

    Parameters
    ----------
    adjacency : array-like, shape (a, a)
        Symmetric bond-order matrix.
    atomnum : array-like, shape (a,)
        Atomic number per atom.
    aromatic : array-like or None
        Per-atom aromaticity flag; default 0.
    nbits : int
        Width of the folded fingerprint.
    minpath, maxpath : int
        Smallest and largest subgraph size in bonds.  RDKit defaults 1
        and 7.
    branched : bool
        Enumerate all connected subgraphs (default, RDKit's
        ``branchedPaths``) rather than linear paths only.
    use_bond_order : bool
        Include bond order in the bond hash.

    Returns
    -------
    RichResult
        ``bits``, ``count``, ``nset``, ``features``, ``nfeature``,
        ``nsubgraph``, ``a``, ``nbits``, ``minpath``, ``maxpath``,
        ``method``.

    References
    ----------
    RDKit: Open-Source Cheminformatics, https://www.rdkit.org.  The
    fingerprint has no journal paper; the reference implementation is the
    specification.  Files followed:
    Code/GraphMol/Fingerprints/RDKitFPGenerator.cpp (atom invariant lines
    44-54; subgraph enumeration and feature assembly lines 196-249) and
    Code/GraphMol/Fingerprints/FingerprintUtil.cpp
    (``generateBondHashes`` lines 357-444), RDKit master revision fetched
    2026-08-09 and stored at library/pdf/fetched-wave3/
    rdkit-reference-source/rdkit-master-RDKitFPGenerator.cpp and
    rdkit-master-FingerprintUtil.cpp.
    """
    a, bonds = _bonds(adjacency)
    at = [int(z) for z in atomnum]
    if len(at) != a:
        raise ValueError("atomnum must have one entry per atom")
    if aromatic is None:
        ar = [0] * a
    else:
        ar = [1 if int(z) else 0 for z in aromatic]
        if len(ar) != a:
            raise ValueError("aromatic must have one entry per atom")
    ainv = [(at[i] % 128) * 2 + ar[i] for i in range(a)]

    minpath = int(minpath)
    maxpath = int(maxpath)
    if minpath < 1:
        raise ValueError("minpath must be at least 1")
    if maxpath < minpath:
        raise ValueError("maxpath must be at least minpath")

    subs = _subgraphs(a, bonds, minpath, maxpath, bool(branched))
    nbits = int(nbits)
    if nbits < 1:
        raise ValueError("nbits must be positive")
    bits = [0] * nbits
    cnt = [0] * nbits
    feats = []
    for sub in subs:
        deg = {}
        atoms = set()
        for bi in sub:
            i, j, _o = bonds[bi]
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
            atoms.add(i)
            atoms.add(j)
        bh = []
        for k, bi in enumerate(sub):
            i, j, o = bonds[bi]
            nbr = 0
            for m, bj in enumerate(sub):
                if m == k:
                    continue
                p, q, _ = bonds[bj]
                if p == i or p == j or q == i or q == j:
                    nbr += 1
            a1, a2 = ainv[i], ainv[j]
            d1, d2 = deg[i], deg[j]
            if a1 < a2:
                a1, a2 = a2, a1
                d1, d2 = d2, d1
            elif a1 == a2 and d1 < d2:
                d1, d2 = d2, d1
            bo = int(o) if use_bond_order else 1
            h = _mix(0, nbr)
            for v in (bo, a1, d1, a2, d2):
                h = _mix(h, v)
            bh.append(h)
        if len(sub) > 1:
            bh.sort()
            bh.append(len(atoms))
            seed = 0
            for v in bh:
                seed = _mix(seed, v)
        else:
            seed = bh[0]
        feats.append(seed)
        b = seed % nbits
        bits[b] = 1
        cnt[b] += 1

    uniq = sorted(set(feats))
    return RichResult(payload={
        "bits": bits, "count": cnt, "nset": sum(bits),
        "features": uniq, "nfeature": len(uniq), "nsubgraph": len(subs),
        "a": a, "nbits": nbits, "minpath": minpath, "maxpath": maxpath,
        "method": "RDKit path-based topological fingerprint"})


rdkit_path_fp = rdkfp
rdkitpathfp = rdkfp


def cheatsheet():
    return "rdkfp: RDKit path/subgraph-based topological fingerprint."
