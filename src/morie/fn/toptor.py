r"""Topological torsion descriptors.

Nilakantan, R., Bauman, N., Dixon, J. S., & Venkataraghavan, R. (1987)
"Topological Torsion: A New Molecular Descriptor for SAR Applications.
Comparison with Other Descriptors", *Journal of Chemical Information and
Computer Sciences* 27(2), 82-85.

A topological torsion is "a linear sequence of four consecutively bonded
non-hydrogen atoms, each described by its atomic type, the number of
non-hydrogen branches attached to it, and its number of pi electron pairs" --
the topological analogue of the torsion angle, which is "the minimal
structural unit in terms of which the conformation of a molecule can be
completely described". Schematically

    (NPI-TYPE-NBR)-(NPI-TYPE-NBR)-(NPI-TYPE-NBR)-(NPI-TYPE-NBR)

**The branch count excludes the torsion itself**: "for the two end atoms of
the torsion the number of branches is calculated as the total number of
branches minus 1, and for the two central atoms this number is calculated as
the total number of branches minus 2".

**Pi electrons stand in for bond types**, and the paper says why: "In benzene,
for example, directly encoding the bond types in the TT descriptor would
result in two different TT types in the molecule. On the other hand, coding
the pi electrons on each atom of the TT implicitly encodes the bonds, making
all the descriptors in benzene equivalent." That claim is an anchor here --
benzene yields exactly one distinct torsion type.

Enumeration follows the paper: loop over atoms and three successive levels of
branching, with "checks ... to assure that the atoms in the TT quartet are
distinct and that the same TT is not counted twice in opposite directions".
The canonical form is the lexicographically smaller of the sequence and its
reverse, which is this module's stand-in for the paper's "canonical packing
scheme" (the packing itself is a 32-bit layout, not a definition).

Two uses from the paper are implemented on top:

* the **similarity probe**, scored by :math:`S = 2 D_{ij}/(d_i + d_j)` with
  :math:`d_i, d_j` the numbers of distinct descriptors and :math:`D_{ij}` the
  number in common;
* the **trend vector** :math:`T = (1/N)\sum_i (a_i - A) S_i` over the
  descriptor indicator vectors, with the paper's randomisation test --
  reassign the activities to the wrong structures, rebuild the vector "say,
  40 times", and report the real vector's length in standard deviations of
  the spurious ones.

The paper distinguishes "only 13 common atom types ... all others being
lumped together as a fictitious element Y" but does not print the list, so
``common_types`` is exposed with a documented default of thirteen common
organic elements. That default is this module's choice, not a quotation.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["toptor", "topological_torsion", "topological_torsions",
           "torsion_similarity", "trend_vector"]

#: Thirteen common atom types; everything else becomes "Y" (see module docs).
COMMON_TYPES = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B",
                "Se", "As")


def _neighbours(n_atoms, bonds):
    adj = dict((i, []) for i in range(n_atoms))
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        if i == j:
            raise ValueError("toptor: a bond from an atom to itself")
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError("toptor: bond refers to an atom outside the "
                             "molecule")
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _pi_electrons(n_atoms, bonds):
    """NPI from the bond types, one pi electron per bond order above one.

    The paper says only that "the number of pi electrons (NPI) on each atom
    of the quartet is calculated from the bond types"; aromatic bonds are
    taken as order 1.5, so a benzene carbon gets 1.
    """
    npi = [0.0] * n_atoms
    for b in bonds:
        order = float(b[2]) if len(b) > 2 else 1.0
        if order < 1.0:
            raise ValueError("toptor: bond order below 1")
        npi[int(b[0])] += order - 1.0
        npi[int(b[1])] += order - 1.0
    return [int(round(v)) for v in npi]


def topological_torsions(elements, bonds, common_types=None):
    r"""Every topological torsion in a molecule, with multiplicities.

    Parameters
    ----------
    elements : sequence of str
        Element symbol per non-hydrogen atom. Hydrogens are not part of the
        descriptor and should not be listed.
    bonds : sequence of ``(i, j)`` or ``(i, j, order)``
        Bonds between those atoms; ``order`` may be 1, 1.5 (aromatic), 2 or
        3 and defaults to 1.
    common_types : sequence of str, optional
        Atom types kept as themselves; everything else becomes ``"Y"``.

    Returns
    -------
    dict
        Canonical torsion code -> count. Each code is a 4-tuple of
        ``(npi, type, nbr)`` triples, in the canonical (lexicographically
        smaller) direction.
    """
    els = [str(e) for e in elements]
    n = len(els)
    if n == 0:
        raise ValueError("toptor: the molecule has no heavy atoms")
    keep = set(common_types if common_types is not None else COMMON_TYPES)
    types = [e if e in keep else "Y" for e in els]
    adj = _neighbours(n, bonds)
    npi = _pi_electrons(n, bonds)
    degree = [len(adj[i]) for i in range(n)]

    out = {}
    for a in range(n):
        for b in adj[a]:
            for c in adj[b]:
                if c == a:
                    continue
                for d in adj[c]:
                    if d == b or d == a:
                        continue
                    path = (a, b, c, d)
                    code = tuple(
                        (npi[p], types[p], degree[p] - (1 if k in (0, 3)
                                                        else 2))
                        for k, p in enumerate(path))
                    rev = tuple(reversed(code))
                    canon = min(code, rev)
                    # each undirected path is walked twice, once per end
                    if a > d:
                        continue
                    out[canon] = out.get(canon, 0) + 1
    return out


def torsion_similarity(t1, t2):
    r"""The paper's similarity score :math:`S = 2 D_{ij}/(d_i + d_j)`.

    ``t1`` and ``t2`` are torsion dictionaries (or any iterables of codes);
    only the *distinct* descriptors count, as in the printed formula.
    """
    s1 = set(t1)
    s2 = set(t2)
    if not s1 and not s2:
        raise ValueError("toptor: both molecules have no torsions, so the "
                         "similarity is undefined")
    return 2.0 * len(s1 & s2) / float(len(s1) + len(s2))


def trend_vector(torsion_sets, activities, permutations=40, seed=0):
    r"""The trend vector :math:`T = (1/N)\sum_i (a_i - A) S_i` and its
    randomisation test.

    :math:`S_i` is the 0/1 indicator vector of the descriptors present in
    structure :math:`i`. Significance follows the paper: reassign the
    activities to the wrong structures, rebuild the vector, repeat (the
    paper uses 40), and report the real vector's length in standard
    deviations of the spurious lengths.

    Returns ``{"vector", "descriptors", "length", "null_mean", "null_sd",
    "z"}``.
    """
    sets = [set(t) for t in torsion_sets]
    a = [float(v) for v in activities]
    n = len(sets)
    if n != len(a):
        raise ValueError("toptor: one activity per structure is required")
    if n < 2:
        raise ValueError("toptor: the trend vector needs at least two "
                         "structures")
    permutations = int(permutations)
    if permutations < 1:
        raise ValueError("toptor: permutations must be >= 1")
    keys = sorted(set().union(*sets) if sets else set(), key=repr)
    if not keys:
        raise ValueError("toptor: no descriptors in any structure")
    S = [[1.0 if k in s else 0.0 for k in keys] for s in sets]

    def build(order):
        mean = sum(a) / n
        vec = [0.0] * len(keys)
        for i in range(n):
            w = a[order[i]] - mean
            for j in range(len(keys)):
                vec[j] += w * S[i][j]
        return [v / n for v in vec]

    real = build(list(range(n)))
    length = math.sqrt(sum(v * v for v in real))
    rng = np.random.default_rng(seed)
    lens = []
    for _ in range(permutations):
        order = list(range(n))
        for t in range(n - 1, 0, -1):
            u = int(rng.random() * (t + 1))
            order[t], order[u] = order[u], order[t]
        v = build(order)
        lens.append(math.sqrt(sum(x * x for x in v)))
    m = sum(lens) / len(lens)
    var = sum((v - m) ** 2 for v in lens) / max(1, len(lens) - 1)
    sd = math.sqrt(var)
    if sd > 0:
        z = (length - m) / sd
    elif abs(length - m) < 1e-12:
        # every permutation reproduces the real vector, which happens when
        # the structures carry identical descriptor sets: no signal, and no
        # spread to measure it against
        z = 0.0
    else:
        z = float("inf")
    return {"vector": real, "descriptors": keys, "length": length,
            "null_mean": m, "null_sd": sd, "z": z}


def toptor(elements, bonds, reference=None, common_types=None,
           activities=None, permutations=40, seed=0):
    r"""Topological torsions for a molecule, or for a set of molecules.

    Parameters
    ----------
    elements : sequence of str, or sequence of such sequences
        One molecule's heavy-atom elements, or several molecules'.
    bonds : sequence
        The matching bond list(s).
    reference : ``(elements, bonds)``, optional
        A probe molecule. Given it, every molecule is scored against it by
        :math:`S = 2D/(d_i + d_j)` and the result is sorted -- the paper's
        similarity probe.
    common_types : sequence of str, optional
        See :func:`topological_torsions`.
    activities : sequence of float, optional
        Given with several molecules, the trend vector and its
        randomisation test are computed.
    permutations : int
        Randomisations for the trend vector's significance ("say, 40").
    seed : int
        Seed for those randomisations.

    Returns
    -------
    RichResult
        ``estimate`` / ``torsions`` is the torsion dictionary for a single
        molecule, or the list of them; ``n_distinct`` and ``n_total``
        count them; ``similarity`` and ``ranking`` appear with a
        ``reference``; ``trend`` with ``activities``.

    Examples
    --------
    Benzene has one distinct torsion type, which is the paper's own
    argument for coding pi electrons instead of bond types::

        els = ["C"] * 6
        ring = [(i, (i + 1) % 6, 1.5) for i in range(6)]
        toptor(els, ring)["n_distinct"]        # 1

    References
    ----------
    Nilakantan, Bauman, Dixon & Venkataraghavan (1987) *J. Chem. Inf.
    Comput. Sci.* 27(2), 82-85.
    """
    many = bool(elements) and isinstance(elements[0], (list, tuple))
    if many:
        mols = [(list(e), list(b)) for e, b in zip(elements, bonds)]
    else:
        mols = [(list(elements), list(bonds))]
    tors = [topological_torsions(e, b, common_types) for e, b in mols]

    payload = {
        "estimate": tors if many else tors[0],
        "torsions": tors if many else tors[0],
        "n_distinct": [len(t) for t in tors] if many else len(tors[0]),
        "n_total": ([sum(t.values()) for t in tors] if many
                    else sum(tors[0].values())),
        "method": "topological torsion descriptors (Nilakantan et al. 1987)",
    }
    if reference is not None:
        ref = topological_torsions(reference[0], reference[1], common_types)
        sims = [torsion_similarity(ref, t) for t in tors]
        payload["reference_torsions"] = ref
        payload["similarity"] = sims if many else sims[0]
        payload["ranking"] = sorted(range(len(sims)),
                                    key=lambda i: -sims[i])
    if activities is not None:
        payload["trend"] = trend_vector(tors, activities, permutations, seed)
    return RichResult(payload=payload)


def cheatsheet():
    return ("toptor: topological torsion (Nilakantan 1987). Four "
            "consecutively bonded HEAVY atoms, each coded (NPI, TYPE, "
            "NBR); NBR excludes the torsion itself -- total branches minus "
            "1 at the ends, minus 2 in the middle. Pi electrons stand in "
            "for bond types on purpose: it makes every torsion in benzene "
            "the same descriptor, where explicit bond types would give "
            "two. Each undirected path counted once, canonical direction. "
            "Similarity S = 2D/(d_i + d_j); trend vector "
            "T = (1/N) sum (a_i - A) S_i with a 40-fold randomisation "
            "test.")


# compact alias per ledger/NAMING.md
topological_torsion = toptor
