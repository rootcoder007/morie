# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extended-connectivity fingerprint, radius 2 (ECFP4)."""

from ._richresult import RichResult

__all__ = ["ecfp4", "ecfp_4_fingerprint"]

# Modulus and multiplier of the identifier mixer.  2147483647 = 2^31 - 1 is
# prime; every intermediate product stays below 2^53 and is therefore exact
# in IEEE doubles, which is what lets the R arm reproduce these integers
# bit for bit without 32-bit unsigned arithmetic.
_MOD = 2147483647
_MUL = 1000003


def _mix(h, v):
    """One step of the identifier mixer: h <- (h * 1000003 + v) mod 2^31-1."""
    return (int(h) * _MUL + (int(v) % _MOD)) % _MOD


def _bonds(adjacency):
    """Bond list (i, j, order) over the upper triangle of a bond-order matrix."""
    A = [[float(v) for v in row] for row in adjacency]
    a = len(A)
    if any(len(r) != a for r in A):
        raise ValueError("adjacency must be square")
    out = []
    for i in range(a):
        for j in range(i + 1, a):
            if A[i][j] != A[j][i]:
                raise ValueError("adjacency must be symmetric")
            if A[i][j] != 0.0:
                out.append((i, j, int(A[i][j])))
    return a, out


def _connectivity_invariants(a, bonds, atomnum, numhs, charge, inring, isotope_delta):
    """Round-0 Daylight/ECFP atom invariants.

    The component vector is the one RDKit builds in
    ``getConnectivityInvariants``: atomic number, total degree (heavy
    neighbours plus attached hydrogens), total hydrogen count, formal
    charge, isotope mass delta, and -- when the atom is in a ring -- a
    trailing 1.  See the source reference in the module docstring below.
    """
    deg = [0] * a
    for i, j, _o in bonds:
        deg[i] += 1
        deg[j] += 1
    inv = []
    for i in range(a):
        comps = [
            int(atomnum[i]),
            deg[i] + int(numhs[i]),
            int(numhs[i]),
            int(charge[i]),
            int(isotope_delta[i]),
        ]
        if int(inring[i]):
            comps.append(1)
        h = 0
        for c in comps:
            h = _mix(h, c)
        inv.append(h)
    return inv


def _envkey(bondset):
    """Order- and representation-stable key for a set of bond indices.

    Zero-padded fixed-width concatenation, so that byte-lexicographic
    comparison of two keys reproduces elementwise comparison of the sorted
    index vectors (shorter prefix first).  Both arms sort with this key, so
    the round in which a duplicate environment is retired is identical.
    """
    return "".join("%04d" % b for b in sorted(bondset))


def _morgan(a, bonds, invariants, radius, nbits, use_bond_order=True):
    """Morgan / extended-connectivity relabelling.

    Round 0 emits the atom invariants.  Each further round replaces every
    atom identifier by a hash of (round number, own identifier, the sorted
    list of (bond order, neighbour identifier) pairs).  An environment --
    the set of bonds it covers -- that has already been emitted is not
    emitted again, and the atom that produced it takes no further part
    (the "dead atom" rule).
    """
    nb = len(bonds)
    inc = [[] for _ in range(a)]
    for bi, (i, j, o) in enumerate(bonds):
        inc[i].append((bi, j, o if use_bond_order else 1))
        inc[j].append((bi, i, o if use_bond_order else 1))

    cur = list(invariants)
    ident = []
    for i in range(a):
        ident.append(cur[i])

    seen = set()
    atom_env = [set() for _ in range(a)]
    dead = [False] * a

    for layer in range(int(radius)):
        nxt = [0] * a
        round_env = [set(atom_env[i]) for i in range(a)]
        acc = []
        for i in range(a):
            if dead[i]:
                continue
            nbrs = []
            for bi, oidx, o in inc[i]:
                round_env[i].add(bi)
                round_env[i] |= atom_env[oidx]
                nbrs.append((int(o), cur[oidx]))
            nbrs.sort()
            invar = _mix(0, layer)
            invar = _mix(invar, cur[i])
            for bt, ninv in nbrs:
                invar = _mix(_mix(invar, bt), ninv)
            nxt[i] = invar
            acc.append((_envkey(round_env[i]), invar, i))
        acc.sort()
        for key, invar, i in acc:
            if key not in seen:
                seen.add(key)
                ident.append(invar)
            else:
                dead[i] = True
        for i in range(a):
            if not dead[i]:
                cur[i] = nxt[i]
        atom_env = round_env

    nbits = int(nbits)
    if nbits < 1:
        raise ValueError("nbits must be positive")
    bits = [0] * nbits
    cnt = [0] * nbits
    for v in ident:
        b = v % nbits
        bits[b] = 1
        cnt[b] += 1
    return bits, cnt, ident


def _defaults(a, numhs, charge, inring, isotope_delta):
    def col(x, default):
        if x is None:
            return [default] * a
        v = [int(z) for z in x]
        if len(v) != a:
            raise ValueError("per-atom vector has the wrong length")
        return v

    return (col(numhs, 0), col(charge, 0), col(inring, 0), col(isotope_delta, 0))


def ecfp4(adjacency, atomnum, numhs=None, charge=None, inring=None,
          isotope_delta=None, nbits=2048, radius=2):
    """Extended-connectivity fingerprint of radius 2 (ECFP4).

    ECFP diameter 4 is Morgan radius 2.  The molecule is supplied as a
    pre-parsed graph, not as SMILES: ``adjacency`` is a square bond-order
    matrix (0 no bond, 1 single, 2 double, 3 triple, 4 aromatic -- this
    encoding is this implementation's own and is stated rather than
    attributed), and the remaining arguments are the per-atom properties
    the invariant needs.  SMILES parsing and aromaticity perception are
    deliberately out of scope; this function starts from the graph a
    parser would hand it.

    Algorithm, following the RDKit reference implementation exactly in
    structure:

    * Round-0 atom invariants are the component vector of
      ``RDKit::getConnectivityInvariants``
      (Code/GraphMol/Fingerprints/FingerprintUtil.cpp, lines 242-265 of
      the master revision fetched 2026-08-09): atomic number, total
      degree, total hydrogen count, formal charge, isotope mass delta,
      and a trailing 1 for ring atoms.
    * Each round hashes (layer, own identifier, sorted (bond order,
      neighbour identifier) pairs) -- MorganGenerator.cpp lines 395-455.
    * An environment already emitted in an earlier round is not emitted
      again and its atom is retired -- MorganGenerator.cpp lines 470-495.

    The one deliberate departure is the hash itself.  RDKit uses
    ``boost::hash_combine``; reproducing RDKit's integers bit for bit
    would additionally require RDKit's aromaticity model and bond-type
    enum, which a graph-level implementation cannot have.  The mixer used
    here is the stated closed form h <- (h * 1000003 + v) mod (2^31 - 1),
    fixed so that the Python and both R arms agree exactly.  Bit indices
    are therefore this implementation's own, while the identifier
    *partition* -- which atoms share an identifier at which radius, and
    how many distinct environments a molecule has -- is the published
    ECFP one and is what the tests anchor on.

    Parameters
    ----------
    adjacency : array-like, shape (a, a)
        Symmetric bond-order matrix; 0 means no bond.
    atomnum : array-like, shape (a,)
        Atomic number per atom.
    numhs, charge, inring, isotope_delta : array-like or None
        Attached hydrogen count, formal charge, ring-membership flag and
        isotope mass delta per atom.  Default 0.
    nbits : int
        Width of the folded fingerprint.
    radius : int
        Morgan radius; 2 for ECFP4.

    Returns
    -------
    RichResult
        ``bits``, ``count``, ``nset``, ``identifiers``, ``nenv``, ``a``,
        ``nbits``, ``radius``, ``method``.

    References
    ----------
    Rogers, D. and Hahn, M. (2010), "Extended-connectivity fingerprints",
    Journal of Chemical Information and Modeling 50(5), 742-754,
    doi:10.1021/ci100050t -- the original description of ECFP/FCFP.  The
    article is paywalled at ACS and was NOT read for this
    implementation; it is recorded in
    ledger/wave3/NEEDED_SOURCES.md.  The specification actually followed
    is the open-source RDKit reference implementation, files
    Code/GraphMol/Fingerprints/MorganGenerator.cpp and
    Code/GraphMol/Fingerprints/FingerprintUtil.cpp of the RDKit master
    revision, fetched 2026-08-09 and stored locally at
    library/pdf/fetched-wave3/rdkit-reference-source/
    (rdkit-master-MorganGenerator.cpp, rdkit-master-FingerprintUtil.cpp);
    line numbers cited above.  RDKit: Open-Source Cheminformatics,
    https://www.rdkit.org.
    """
    a, bonds = _bonds(adjacency)
    at = [int(z) for z in atomnum]
    if len(at) != a:
        raise ValueError("atomnum must have one entry per atom")
    nh, ch, ir, isd = _defaults(a, numhs, charge, inring, isotope_delta)
    inv = _connectivity_invariants(a, bonds, at, nh, ch, ir, isd)
    bits, cnt, ident = _morgan(a, bonds, inv, int(radius), int(nbits))
    uniq = sorted(set(ident))
    return RichResult(payload={
        "bits": bits, "count": cnt, "nset": sum(bits),
        "identifiers": uniq, "nenv": len(ident), "a": a,
        "nbits": int(nbits), "radius": int(radius),
        "method": "ECFP4 (Morgan radius 2), Rogers-Hahn / RDKit"})


ecfp_4_fingerprint = ecfp4


def cheatsheet():
    return "ecfp4: extended-connectivity fingerprint, Morgan radius 2 (ECFP4)."
