"""Scaffold hopping: find a molecule that keeps the pharmacophore and
throws away the scaffold.

The medicinal-chemistry problem is that a lead compound is two things at
once. It is a PHARMACOPHORE -- an arrangement of donors, acceptors,
charges and greasy patches in space, which is what the protein actually
recognises -- and it is a SCAFFOLD, the ring-and-linker skeleton that
holds them there, which is what the patent covers and what the
metabolism attacks. A scaffold hop keeps the first and replaces the
second. Schneider and colleagues' contribution was to make that
searchable: describe a molecule by its pharmacophore alone, in a form
that carries no memory of the skeleton, and then go looking for
molecules that match the description and do not match the skeleton.

Three pieces:

  THE DESCRIPTOR. CATS -- a topological pharmacophore correlation
  vector. Each atom is assigned pharmacophore types; for every ordered
  pair of types and every topological distance up to a limit, count the
  atom pairs of those types that many bonds apart. The result says
  "there is a donor five bonds from an acceptor" without saying what is
  in between, which is exactly the information a hop must preserve and
  the information a scaffold comparison must ignore.

  THE SCAFFOLD. Bemis and Murcko's framework: strip side chains by
  repeatedly deleting terminal atoms that are not in a ring, until only
  the ring systems and the linkers between them remain. A molecule with
  no rings has no framework, and that is reported as an empty scaffold
  rather than as the whole molecule.

  THE HOP. A candidate is a hop when its descriptor is close and its
  scaffold is different. Both halves are necessary and the module
  reports them separately, because a candidate that is merely similar is
  an analogue, and a candidate that is merely different is a different
  molecule.

WHAT IS THIS MODULE'S OWN. The five CATS types are Schneider's; the
rules that assign an atom to them are stated in the original work as
SMARTS patterns, and the ones here are this module's operationalisation
of the same five categories, written out in ``atom_types`` so a reader
can disagree with a specific rule rather than with a black box.

Scaffold identity is decided by a Weisfeiler-Leman colour refinement
over the framework subgraph. That is a graph INVARIANT, not a canonical
form: two isomorphic scaffolds always agree, and two different
scaffolds almost always disagree but are not guaranteed to. The module
says "different by this invariant", never "different", and the
refinement depth is a parameter.

References
  Schneider, G., Neidhart, W., Giller, T. and Schmid, G. (1999)
    "'Scaffold-hopping' by topological pharmacophore search: a
    contribution to virtual screening design." Angewandte Chemie
    International Edition 38(19), 2894-2896.
    doi:10.1002/(SICI)1521-3773(19991004)38:19<2894::AID-ANIE2894>3.0.CO;2-F
    Where CATS and the term scaffold hopping are introduced.
  Bemis, G.W. and Murcko, M.A. (1996) "The properties of known drugs. 1.
    Molecular frameworks." Journal of Medicinal Chemistry 39(15),
    2887-2893. doi:10.1021/jm9602928. The framework definition used
    here.
  Weisfeiler, B. and Leman, A.A. (1968) "The reduction of a graph to
    canonical form and the algebra which appears therein."
    Nauchno-Technicheskaya Informatsia 2(9), 12-16.
  Shervashidze, N., Schweitzer, P., van Leeuwen, E.J., Mehlhorn, K. and
    Borgwardt, K.M. (2011) "Weisfeiler-Lehman graph kernels." Journal
    of Machine Learning Research 12, 2539-2561. The refinement as a
    practical graph invariant.
"""

import math

from . import _w3num as _w
from .avalon import parse_smiles, _adjacency, _bfs_dist, implicit_h, _fnv
from ._richresult import RichResult

__all__ = ["scaffold_hop", "atom_types", "cats", "murcko_scaffold",
           "scaffold_signature", "similarity", "cheatsheet"]

# Schneider's five categories, in a fixed order so the vector's layout
# is a property of the module and not of a dictionary's iteration.
TYPES = ("A", "D", "L", "N", "P")
_METRICS = ("tanimoto", "euclidean", "cosine")
_SCALINGS = ("type", "count", "none")


def atom_types(smiles):
    """Assign each atom its pharmacophore categories.

    An atom may have several, or none. The rules, written out so they
    can be argued with:

      D  donor       nitrogen or oxygen carrying at least one hydrogen.
      A  acceptor    nitrogen or oxygen. An atom can be both, which is
                     correct: a hydroxyl is a donor and an acceptor.
      P  positive    an aliphatic nitrogen with no neighbouring
                     carbonyl carbon -- an amine rather than an amide,
                     which is the distinction that decides whether it
                     is protonated at physiological pH.
      N  negative    the oxygens of a carboxyl group: a carbon bearing
                     both a doubly bonded oxygen and a singly bonded
                     one.
      L  lipophilic  carbon with no nitrogen or oxygen neighbour, and
                     sulfur and the halogens.
    """
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    adj = _adjacency(n, bonds)
    nh = implicit_h(el, arom, chg, hexp, bonds)
    # A carbon is a carbonyl carbon when it holds a doubly bonded
    # oxygen, and a carboxyl carbon when it also holds a single-bonded
    # one. Both are read off the bond orders, not guessed from the
    # element.
    dbl_o = [0] * n
    sng_o = [0] * n
    for i in range(n):
        for v, o, k in adj[i]:
            if el[v] == "O":
                if o == 2:
                    dbl_o[i] += 1
                elif o == 1:
                    sng_o[i] += 1
    out = []
    for i in range(n):
        t = []
        e = el[i]
        if e in ("N", "O"):
            t.append("A")
            if nh[i] > 0:
                t.append("D")
        if e == "N" and not arom[i]:
            amide = False
            for v, o, k in adj[i]:
                if el[v] == "C" and dbl_o[v] > 0:
                    amide = True
            if not amide:
                t.append("P")
        if e == "O":
            for v, o, k in adj[i]:
                if el[v] == "C" and dbl_o[v] > 0 and sng_o[v] > 0:
                    t.append("N")
        if e == "C":
            het = False
            for v, o, k in adj[i]:
                if el[v] in ("N", "O"):
                    het = True
            if not het:
                t.append("L")
        elif e in ("S", "F", "Cl", "Br", "I"):
            t.append("L")
        out.append(sorted(set(t)))
    return out


def _pairs():
    """The fifteen unordered type pairs, in a fixed order."""
    out = []
    for a in range(len(TYPES)):
        for b in range(a, len(TYPES)):
            out.append((TYPES[a], TYPES[b]))
    return out


def cats(smiles, maxdist=9, scaling="type"):
    """The CATS correlation vector: type pairs by topological distance.

    Laid out as pair-major, distance-minor, so entry
    ``p * (maxdist + 1) + d`` is the count of pairs of the p-th type
    combination exactly d bonds apart. Distance zero is an atom with
    itself, which is how an atom carrying two types registers at all.

    Scaling routes, all three of which are defensible and which give
    different answers, so the choice is the caller's:

      ``type``   divide each entry by the number of atoms carrying the
                 two types. Schneider's scaling; it stops a large
                 molecule dominating simply by being large.
      ``count``  divide by the total number of atom pairs counted.
      ``none``   raw counts, which is what you want if you intend to
                 compare absolute frequencies.
    """
    if scaling not in _SCALINGS:
        raise ValueError("the scaling is type, count or none")
    maxdist = int(maxdist)
    if maxdist < 0:
        raise ValueError("a distance limit below zero counts nothing")
    ty = atom_types(smiles)
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    D = _bfs_dist(_adjacency(n, bonds), n)
    P = _pairs()
    idx = {}
    for p in range(len(P)):
        idx[P[p]] = p
    v = [0.0] * (len(P) * (maxdist + 1))
    have = {}
    for t in TYPES:
        have[t] = 0
    for i in range(n):
        for t in ty[i]:
            have[t] += 1
    for i in range(n):
        for j in range(i, n):
            d = D[i][j]
            if d < 0 or d > maxdist:
                continue
            for a in ty[i]:
                for b in ty[j]:
                    x, y = (a, b) if a <= b else (b, a)
                    v[idx[(x, y)] * (maxdist + 1) + d] += 1.0
    if scaling == "type":
        for p in range(len(P)):
            s = have[P[p][0]] + have[P[p][1]]
            if s > 0:
                for d in range(maxdist + 1):
                    v[p * (maxdist + 1) + d] /= float(s)
            else:
                for d in range(maxdist + 1):
                    v[p * (maxdist + 1) + d] = 0.0
    elif scaling == "count":
        tot = _w.csum(v)
        if tot > 0:
            for q in range(len(v)):
                v[q] /= tot
    return v


def similarity(a, b, metric="tanimoto"):
    """How close two CATS vectors are.

    ``tanimoto`` is the continuous form, sum of minima over sum of
    maxima: one exactly when the vectors are equal, zero when they
    share no dimension. ``euclidean`` is reported as a similarity
    ``1/(1+d)`` so that every route points the same way -- larger is
    closer -- and ``cosine`` ignores magnitude entirely.
    """
    if len(a) != len(b):
        raise ValueError("two descriptors of different lengths cannot "
                         "be compared")
    if metric == "tanimoto":
        lo = _w.csum(a[i] if a[i] < b[i] else b[i] for i in range(len(a)))
        hi = _w.csum(a[i] if a[i] > b[i] else b[i] for i in range(len(a)))
        return (lo / hi) if hi > 0 else 0.0
    if metric == "euclidean":
        d = math.sqrt(_w.csum((a[i] - b[i]) * (a[i] - b[i])
                              for i in range(len(a))))
        return 1.0 / (1.0 + d)
    if metric == "cosine":
        num = _w.csum(a[i] * b[i] for i in range(len(a)))
        na = math.sqrt(_w.csum(x * x for x in a))
        nb = math.sqrt(_w.csum(x * x for x in b))
        return (num / (na * nb)) if na > 0 and nb > 0 else 0.0
    raise ValueError("the metric is tanimoto, euclidean or cosine")


def murcko_scaffold(smiles):
    """The Bemis-Murcko framework: ring systems and the linkers between.

    Strip side chains by repeatedly deleting any atom that is not in a
    ring and has at most one remaining neighbour. What survives is the
    rings plus every atom on a path between two of them, which is the
    framework. A molecule with no rings loses everything, and the empty
    scaffold is returned rather than the whole molecule -- an acyclic
    lead has no skeleton to hop away from and saying so is the useful
    answer.
    """
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    from .avalon import ring_bonds
    rings, inring = ring_bonds(n, bonds, closures)
    keep = [True] * n
    changed = True
    while changed:
        changed = False
        deg = [0] * n
        for a, b, o in bonds:
            if keep[a] and keep[b]:
                deg[a] += 1
                deg[b] += 1
        for i in range(n):
            if keep[i] and not inring[i] and deg[i] <= 1:
                keep[i] = False
                changed = True
    atoms = [i for i in range(n) if keep[i]]
    keptb = [(a, b, o) for a, b, o in bonds if keep[a] and keep[b]]
    return atoms, keptb


def scaffold_signature(smiles, rounds=3):
    """A Weisfeiler-Leman colour of the framework, as a fingerprint.

    Each surviving atom starts coloured by its element, aromaticity and
    charge; a round replaces the colour by a hash of it together with
    the sorted multiset of its neighbours' colours and the bond orders
    reaching them. The signature is the sorted multiset of final
    colours.

    An INVARIANT, not a canonical form: isomorphic scaffolds always
    agree; different ones agree only in the rare cases the refinement
    cannot separate. Everything below says "different by this
    invariant" and never "different".
    """
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    atoms, keptb = murcko_scaffold(smiles)
    if not atoms:
        return []
    pos = {}
    for k in range(len(atoms)):
        pos[atoms[k]] = k
    m = len(atoms)
    nb = [[] for _ in range(m)]
    for a, b, o in keptb:
        nb[pos[a]].append((pos[b], o))
        nb[pos[b]].append((pos[a], o))
    col = [_fnv("%s|%d|%d" % (el[atoms[k]], arom[atoms[k]],
                              chg[atoms[k]])) for k in range(m)]
    for _ in range(int(rounds)):
        nxt = []
        for k in range(m):
            around = sorted("%d:%d" % (o, col[v]) for v, o in nb[k])
            nxt.append(_fnv("%d|%s" % (col[k], ",".join(around))))
        col = nxt
    return sorted(col)


def scaffold_hop(lead_smiles, scaffold_db, maxdist=9, scaling="type",
                 metric="tanimoto", rounds=3, threshold=0.0):
    """Rank candidates by pharmacophore, and say which ones are hops.

    Parameters
    ----------
    lead_smiles : str
        The lead compound.
    scaffold_db : sequence
        Candidate molecules, as SMILES.
    maxdist, scaling : int, str
        The descriptor; see ``cats``.
    metric : str
        The comparison; see ``similarity``.
    rounds : int
        Weisfeiler-Leman refinement depth for the scaffold invariant.
    threshold : float
        Candidates below this similarity are ranked but not called
        hops. Zero calls every different-scaffold candidate a hop,
        which is the honest default: the cut-off is a decision about
        the screening campaign, not a property of the method.

    Returns
    -------
    RichResult
        The ranked candidates, each with its similarity, whether its
        scaffold differs from the lead's, and whether it is a hop.

    References
    ----------
    Schneider et al. (1999) Angew. Chem. Int. Ed. 38(19), 2894-2896;
    Bemis and Murcko (1996) J. Med. Chem. 39(15), 2887-2893.
    """
    lead = cats(lead_smiles, maxdist, scaling)
    lsig = scaffold_signature(lead_smiles, rounds)
    latoms, lbonds = murcko_scaffold(lead_smiles)
    rows = []
    for q in range(len(scaffold_db)):
        sm = scaffold_db[q]
        v = cats(sm, maxdist, scaling)
        s = similarity(lead, v, metric)
        sig = scaffold_signature(sm, rounds)
        atoms, kb = murcko_scaffold(sm)
        diff = sig != lsig
        rows.append({
            "index": q,
            "smiles": sm,
            "similarity": s,
            "scaffold_differs": diff,
            "scaffold_size": len(atoms),
            "is_hop": bool(diff and s >= threshold),
        })
    # Ranked by similarity, ties broken by the order they were given in,
    # so the ranking is a function of the input and not of a sort's
    # internal state.
    order = sorted(range(len(rows)),
                   key=lambda i: (-rows[i]["similarity"], i))
    ranked = [rows[i] for i in order]
    return RichResult(payload={
        "lead": lead,
        "lead_scaffold": latoms,
        "lead_scaffold_size": len(latoms),
        "lead_signature": lsig,
        "ranked": ranked,
        "similarity": [r["similarity"] for r in ranked],
        "is_hop": [r["is_hop"] for r in ranked],
        "order": [r["index"] for r in ranked],
        "n_candidates": len(rows),
        "n_hops": sum(1 for r in rows if r["is_hop"]),
        "n_dim": len(lead),
        "maxdist": int(maxdist),
        "scaling": scaling,
        "metric": metric,
        "rounds": int(rounds),
        "threshold": float(threshold),
        "method": "CATS topological pharmacophore search with a "
                  "Bemis-Murcko scaffold test",
    })


def cheatsheet():
    return ("scfhop: scaffold hopping. CATS pharmacophore correlation "
            "vector for what to keep, Bemis-Murcko framework for what "
            "to change; a hop is close by the first and different by "
            "the second")
