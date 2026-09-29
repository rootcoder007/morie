# morie.fn -- function file (rootcoder007/morie)
"""SMARTS substructure search on SMILES graphs, Wildman-Crippen atom-typed logP and molar refractivity, and
the PAINS pan-assay interference filters (Baell and Holloway 2010) in their RDKit SMARTS form."""

from __future__ import annotations

from ._richresult import RichResult
from .avalon import parse_smiles

__all__ = ["smarts_match", "clogp_estimate", "pains_filter"]

_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]
_Z = {s: i + 1 for i, s in enumerate(_SYMBOLS)}
_AROM_SYM = {"c": 6, "n": 7, "o": 8, "s": 16, "p": 15, "b": 5, "se": 34, "as": 33}
# valences allowed for implicit hydrogens, by the isoelectronic neutral element
_VAL = {1: [1], 5: [3], 6: [4], 7: [3], 8: [2], 9: [1], 14: [4], 15: [3, 5], 16: [2, 4, 6], 17: [1], 35: [1], 53: [1]}


# ---------------------------------------------------------------- molecule graph
def _bridges(n, bonds):
    """Bond k is a ring bond iff its ends stay connected without it."""
    out = []
    for k, (a, b, _o) in enumerate(bonds):
        adj = [[] for _ in range(n)]
        for j, (u, v, _w) in enumerate(bonds):
            if j != k:
                adj[u].append(v)
                adj[v].append(u)
        seen, stack = {a}, [a]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(b in seen)
    return out


def _rings(n, bonds, ringb):
    """Smallest set of smallest rings: shortest cycle through each ring bond, kept when GF(2)-independent."""
    adj = [[] for _ in range(n)]
    for k, (a, b, _o) in enumerate(bonds):
        adj[a].append((b, k))
        adj[b].append((a, k))
    cands = []
    for k, (a, b, _o) in enumerate(bonds):
        if not ringb[k]:
            continue
        prev = {a: (-1, -1)}
        q = [a]
        h = 0
        while h < len(q) and b not in prev:
            u = q[h]
            h += 1
            for v, kk in adj[u]:
                if kk != k and v not in prev:
                    prev[v] = (u, kk)
                    q.append(v)
        path, edges, v = [b], [k], b
        while v != a:
            u, kk = prev[v]
            edges.append(kk)
            path.append(u)
            v = u
        cands.append((len(path), path[::-1], edges))
    cands.sort(key=lambda c: c[0])
    rings, pivots = [], {}
    for _, path, edges in cands:
        vec = 0
        for kk in edges:
            vec ^= 1 << kk
        while vec:
            top = vec.bit_length() - 1
            if top in pivots:
                vec ^= pivots[top]
            else:
                pivots[top] = vec
                rings.append((path, sorted(edges)))
                break
    return rings


def _hcount(z, arom, chg, hexp, bonds):
    n = len(z)
    used = [0] * n
    for a, b, o in bonds:
        w = 1 if o == 4 else o
        used[a] += w
        used[b] += w
    out = []
    for i in range(n):
        if hexp[i] >= 0:
            out.append(hexp[i])
            continue
        vals = _VAL.get(z[i] - chg[i]) if z[i] in (5, 6, 7, 8, 15, 16) else (_VAL.get(z[i]) if chg[i] == 0 else None)
        if vals is None:
            out.append(0)
            continue
        tot = used[i] + (1 if arom[i] and z[i] not in (8, 16) else 0)
        h = 0
        for v in vals:
            if v >= tot:
                h = v - tot
                break
        out.append(h)
    return out


_MORE_EN = (7, 8, 9, 15, 16, 17, 33, 34, 35, 51, 52, 53)


def _electrons(i, z, chg, nbr, inring_bond, hs):
    """Pi electrons an atom gives a ring (RDKit default model), or None when it cannot be aromatic."""
    dbl = [(j, k) for j, k, o in nbr[i] if o == 2]
    deg = len(nbr[i]) + hs[i]
    if any(o == 3 for _, _, o in nbr[i]) or len(dbl) > 1 or deg > 3:
        return None
    if dbl:
        j, k = dbl[0]
        if inring_bond[k]:
            return 1
        if z[i] != 6:
            return None
        return 0 if z[j] in _MORE_EN else 1
    if z[i] == 6:
        return 2 if chg[i] == -1 else (0 if chg[i] == 1 else None)
    if z[i] == 7:
        return 2 if (chg[i] == 0 and deg == 3) or chg[i] == -1 else None
    if z[i] in (8, 16, 34):
        return 2 if chg[i] == 0 and deg == 2 else None
    if z[i] == 5:
        return 0 if deg == 3 else None
    return None


def _kekulize(n, z, arom, chg, hs, bonds, ringb):
    """Bond orders for aromatic input: a perfect matching of double bonds over the atoms that need one."""
    order = [o for _, _, o in bonds]
    cand = [arom[a] and arom[b] and ringb[k] and o in (1, 4) for k, (a, b, o) in enumerate(bonds)]
    cur = list(hs)
    for k, (a, b, o) in enumerate(bonds):
        w = 1 if cand[k] or o == 4 else o
        cur[a] += w
        cur[b] += w
    need = [False] * n
    for i in range(n):
        if arom[i]:
            vals = _VAL.get(z[i] - chg[i]) if z[i] in (5, 6, 7, 8, 15, 16) else _VAL.get(z[i])
            need[i] = bool(vals) and cur[i] not in vals and (cur[i] + 1) in vals
    adj = [[] for _ in range(n)]
    for pref in (4, 1):
        for k, (a, b, o) in enumerate(bonds):
            if cand[k] and o == pref:
                adj[a].append((b, k))
                adj[b].append((a, k))
    mate = [-1] * n
    dbl = set()
    todo = [i for i in range(n) if need[i]]

    def solve(t):
        while t < len(todo) and mate[todo[t]] >= 0:
            t += 1
        if t == len(todo):
            return True
        u = todo[t]
        for v, k in adj[u]:
            if need[v] and mate[v] < 0:
                mate[u], mate[v] = v, u
                dbl.add(k)
                if solve(t + 1):
                    return True
                mate[u], mate[v] = -1, -1
                dbl.discard(k)
        return False

    if not solve(0):
        raise ValueError("cannot kekulize the aromatic system")
    for k, (_a, _b, o) in enumerate(bonds):
        if k in dbl:
            order[k] = 2
        elif cand[k] or o == 4:
            order[k] = 1
    return order


def _molecule(smiles, explicit_h=False):
    el, arom, chg, hexp, bonds, _closures = parse_smiles(smiles)
    n = len(el)
    z = [_Z[e] for e in el]
    arom = [bool(a) for a in arom]
    chg = list(chg)
    hs = _hcount(z, arom, chg, hexp, bonds)
    ringb = _bridges(n, bonds)
    rs = _rings(n, bonds, ringb)
    rings = [r for r, _ in rs]
    ring_edges = [e for _, e in rs]
    # RDKit's pipeline: kekulize what was written aromatic, then perceive aromaticity from scratch
    order = _kekulize(n, z, arom, chg, hs, bonds, ringb)
    arom = [False] * n
    nbr = [[] for _ in range(n)]
    for k, (a, b, _o) in enumerate(bonds):
        nbr[a].append((b, k, order[k]))
        nbr[b].append((a, k, order[k]))
    ec = [_electrons(i, z, chg, nbr, ringb, hs) for i in range(n)]
    # connected unions of up to six SSSR rings (sharing a bond; beyond pairs no atom in three of them)
    # obeying 4n + 2; as RDKit, only the
    # perimeter bonds of a union (in exactly one of its rings) become aromatic
    nr = len(rings)
    rset = [set(r) for r in rings]
    eset = [set(e) for e in ring_edges]
    fused = [[b for b in range(nr) if b != a and eset[a] & eset[b]] for a in range(nr)]
    seen = set()
    frontier = [frozenset([a]) for a in range(nr)]
    for size in range(1, 7):
        nxt = []
        for sub in frontier:
            if sub in seen:
                continue
            seen.add(sub)
            atoms = set().union(*(rset[a] for a in sub))
            if len(sub) > 2 and any(sum(1 for a in sub if v in rset[a]) > 2 for v in atoms):
                tot = None
            else:
                tot = 0
            for v in atoms if tot is not None else ():
                if ec[v] is None:
                    tot = None
                    break
                tot += ec[v]
            if tot is not None and tot % 4 == 2:
                for v in atoms:
                    arom[v] = True
                cnt = {}
                for a in sub:
                    for k in eset[a]:
                        cnt[k] = cnt.get(k, 0) + 1
                for k, c in cnt.items():
                    if c == 1:
                        order[k] = 4
            if size < 6:
                for a in sub:
                    for b in fused[a]:
                        if b not in sub:
                            nxt.append(sub | {b})
        frontier = nxt
    if explicit_h:
        el = list(el)
        bonds = list(bonds)
        for i in range(n):
            for _ in range(hs[i]):
                el.append("H")
                z.append(1)
                arom.append(False)
                chg.append(0)
                bonds.append((i, len(z) - 1, 1))
                order.append(1)
                ringb.append(False)
            hs[i] = 0
        hs = hs + [0] * (len(z) - n)
    N = len(z)
    nbr = [[] for _ in range(N)]
    for k, (a, b, _o) in enumerate(bonds):
        nbr[a].append((b, k))
        nbr[b].append((a, k))
    nring = [0] * N
    smallest = [0] * N
    for r in rings:
        for v in r:
            nring[v] += 1
            if smallest[v] == 0 or len(r) < smallest[v]:
                smallest[v] = len(r)
    rbcount = [sum(1 for _, k in nbr[i] if ringb[k]) for i in range(N)]
    val2 = []
    for i in range(N):
        s = 2 * hs[i]
        for _, k in nbr[i]:
            s += 3 if order[k] == 4 else 2 * order[k]
        val2.append(s)
    return {
        "z": z,
        "arom": arom,
        "chg": list(chg),
        "h": hs,
        "nbr": nbr,
        "order": order,
        "ringbond": ringb,
        "nring": nring,
        "smallest": smallest,
        "rbcount": rbcount,
        "val2": val2,
        "n": N,
    }


# ---------------------------------------------------------------- SMARTS parsing
def _match_close(s, i, op, cl):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == op:
            depth += 1
        elif s[j] == cl:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced SMARTS")


def _bracket_end(s, i):
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c in "[(":
            depth += 1
        elif c in "])":
            depth -= 1
            if depth == 0 and c == "]":
                return j
        j += 1
    raise ValueError("unclosed bracket atom")


def _num(s, i):
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    return (int(s[i:j]) if j > i else None), j


def _atom_prim(s, i, first):
    c = s[i]
    if c == "$":
        j = _match_close(s, i + 1, "(", ")")
        return ("rec", _parse(s[i + 2 : j])), j + 1
    if c == "*":
        return ("true",), i + 1
    if c == "#":
        v, j = _num(s, i + 1)
        return ("z", v), j
    if c.isdigit():
        _, j = _num(s, i)
        return ("true",), j
    if c in "+-":
        sg = 1 if c == "+" else -1
        v, j = _num(s, i + 1)
        if v is not None:
            return ("chg", sg * v), j
        q = sg
        while j < len(s) and s[j] == c:
            q += sg
            j += 1
        return ("chg", q), j
    if c == "@":
        j = i + 1
        while j < len(s) and s[j] == "@":
            j += 1
        return ("true",), j
    if c == "H" and first and (i + 1 >= len(s) or s[i + 1] in "]+-;&,"):
        return ("elem", 1, False), i + 1
    if c in "HDXvRrx":
        v, j = _num(s, i + 1)
        if c == "H":
            return ("h", 1 if v is None else v), j
        if c in "DXv":
            return (c, 1 if v is None else v), j
        if c == "R":
            return (("inring",) if v is None else ("R", v)), j
        if c == "r":
            return (("inring",) if v is None else ("r", v)), j
        return (("x", 1, "ge") if v is None else ("x", v)), j
    if c == "a":
        return ("arom",), i + 1
    if c == "A":
        return ("aliph",), i + 1
    if c.islower():
        for L in (2, 1):
            t = s[i : i + L]
            if len(t) == L and t in _AROM_SYM:
                return ("elem", _AROM_SYM[t], True), i + L
        raise ValueError(f"unknown SMARTS primitive {c!r}")
    if c.isupper():
        if i + 1 < len(s) and s[i + 1].islower() and s[i : i + 2] in _Z:
            return ("elem", _Z[s[i : i + 2]], False), i + 2
        if c in _Z:
            return ("elem", _Z[c], False), i + 1
    raise ValueError(f"unknown SMARTS primitive {c!r}")


def _logic(s, prim):
    """Recursive descent over ; , & ! with implicit and; returns the expression tree."""
    pos = [0]

    def peek():
        return s[pos[0]] if pos[0] < len(s) else ""

    def unary():
        if peek() == "!":
            pos[0] += 1
            return ("not", unary())
        node, pos[0] = prim(s, pos[0], pos[0] == 0)
        return node

    def andh():
        items = [unary()]
        while peek() and peek() not in ",;":
            if peek() == "&":
                pos[0] += 1
            items.append(unary())
        return items[0] if len(items) == 1 else ("and", items)

    def comma():
        items = [andh()]
        while peek() == ",":
            pos[0] += 1
            items.append(andh())
        return items[0] if len(items) == 1 else ("or", items)

    def semi():
        items = [comma()]
        while peek() == ";":
            pos[0] += 1
            items.append(comma())
        return items[0] if len(items) == 1 else ("and", items)

    out = semi()
    if pos[0] != len(s):
        raise ValueError(f"cannot parse SMARTS expression {s!r}")
    return out


def _bond_prim(s, i, _first):
    c = s[i]
    m = {"-": ("single",), "/": ("single",), "\\": ("single",), "=": ("double",), "#": ("triple",), ":": ("aromatic",)}
    if c in m:
        return m[c], i + 1
    if c == "~":
        return ("true",), i + 1
    if c == "@":
        return ("ringbond",), i + 1
    raise ValueError(f"unknown SMARTS bond {c!r}")


_DEFAULT_BOND = ("or", [("single",), ("aromatic",)])
_BOND_CHARS = set("-=#:~@!/\\,;&")


def _parse(sm):
    """SMARTS -> (atom predicates, bonds [(a, b, pred)], component starts)."""
    atoms, bonds, starts = [], [], []
    stack, prev, pend = [], -1, None
    rings = {}
    i = 0
    while i < len(sm):
        c = sm[i]
        if c == "(":
            stack.append(prev)
            i += 1
        elif c == ")":
            prev = stack.pop()
            i += 1
        elif c == ".":
            prev = -1
            i += 1
        elif c in _BOND_CHARS:
            j = i
            while j < len(sm) and sm[j] in _BOND_CHARS:
                j += 1
            pend = _logic(sm[i:j], _bond_prim)
            i = j
        elif c.isdigit() or c == "%":
            if c == "%":
                lab = sm[i + 1 : i + 3]
                i += 3
            else:
                lab = c
                i += 1
            if lab in rings:
                a, bp = rings.pop(lab)
                bonds.append((a, prev, pend or bp or _DEFAULT_BOND))
            else:
                rings[lab] = (prev, pend)
            pend = None
        else:
            if c == "[":
                j = _bracket_end(sm, i)
                pred = _logic(sm[i + 1 : j], _atom_prim)
                i = j + 1
            else:
                if sm[i : i + 2] in ("Cl", "Br"):
                    pred = ("elem", _Z[sm[i : i + 2]], False)
                    i += 2
                elif c in "BCNOSPFI":
                    pred = ("elem", _Z[c], False)
                    i += 1
                elif c in "cnospb":
                    pred = ("elem", _AROM_SYM[c], True)
                    i += 1
                elif c == "*":
                    pred = ("true",)
                    i += 1
                elif c == "a":
                    pred = ("arom",)
                    i += 1
                elif c == "A":
                    pred = ("aliph",)
                    i += 1
                else:
                    raise ValueError(f"unexpected SMARTS character {c!r}")
            atoms.append(pred)
            cur = len(atoms) - 1
            if prev >= 0:
                bonds.append((prev, cur, pend or _DEFAULT_BOND))
            else:
                starts.append(cur)
            prev = cur
            pend = None
    if rings or stack:
        raise ValueError("unclosed ring or branch in SMARTS")
    return {"atoms": atoms, "bonds": bonds, "starts": starts}


def _merge_hs(q):
    """RDKit's mergeHs: a query hydrogen [#1] singly bonded to one atom becomes 'at least k H' on that atom."""

    def is_h(p):
        return p == ("z", 1) or p == ("elem", 1, False)

    atoms = [_merge_pred(p) for p in q["atoms"]]
    deg = [0] * len(atoms)
    for a, b, _ in q["bonds"]:
        deg[a] += 1
        deg[b] += 1
    drop = set()
    add = {}
    for a, b, bp in q["bonds"]:
        for h, o in ((a, b), (b, a)):
            if is_h(atoms[h]) and deg[h] == 1 and not is_h(atoms[o]) and bp in (_DEFAULT_BOND, ("single",)):
                drop.add(h)
                add[o] = add.get(o, 0) + 1
    if not drop:
        return {"atoms": atoms, "bonds": q["bonds"], "starts": q["starts"]}
    keep = [i for i in range(len(atoms)) if i not in drop]
    idx = {o: k for k, o in enumerate(keep)}
    new_atoms = [("and", [atoms[i], ("hge", add[i])]) if i in add else atoms[i] for i in keep]
    new_bonds = [(idx[a], idx[b], bp) for a, b, bp in q["bonds"] if a not in drop and b not in drop]
    return {"atoms": new_atoms, "bonds": new_bonds, "starts": sorted({idx[s] for s in q["starts"] if s in idx} | {0})}


def _merge_pred(p):
    if p[0] == "rec":
        return ("rec", _merge_hs(p[1]))
    if p[0] in ("and", "or"):
        return (p[0], [_merge_pred(x) for x in p[1]])
    if p[0] == "not":
        return ("not", _merge_pred(p[1]))
    return p


# ---------------------------------------------------------------- matching
def _atom_ok(p, m, i, cache):
    t = p[0]
    if t == "and":
        return all(_atom_ok(x, m, i, cache) for x in p[1])
    if t == "or":
        return any(_atom_ok(x, m, i, cache) for x in p[1])
    if t == "not":
        return not _atom_ok(p[1], m, i, cache)
    if t == "true":
        return True
    if t == "elem":
        return m["z"][i] == p[1] and m["arom"][i] == p[2]
    if t == "z":
        return m["z"][i] == p[1]
    if t == "arom":
        return m["arom"][i]
    if t == "aliph":
        return not m["arom"][i]
    if t == "h":
        return m["h"][i] + sum(1 for j, _ in m["nbr"][i] if m["z"][j] == 1) == p[1]
    if t == "hge":
        return m["h"][i] + sum(1 for j, _ in m["nbr"][i] if m["z"][j] == 1) >= p[1]
    if t == "D":
        return len(m["nbr"][i]) == p[1]
    if t == "X":
        return len(m["nbr"][i]) + m["h"][i] == p[1]
    if t == "v":
        return m["val2"][i] == 2 * p[1]
    if t == "chg":
        return m["chg"][i] == p[1]
    if t == "inring":
        return m["nring"][i] > 0 or m["rbcount"][i] > 0
    if t == "R":
        return (m["rbcount"][i] == 0) if p[1] == 0 else m["nring"][i] == p[1]
    if t == "r":
        return m["smallest"][i] == p[1]
    if t == "x":
        return m["rbcount"][i] >= 1 if len(p) > 2 else m["rbcount"][i] == p[1]
    if t == "rec":
        key = (id(p[1]), i)
        if key not in cache:
            cache[key] = _embed(p[1], m, i, cache)
        return cache[key]
    raise ValueError(f"unknown predicate {t}")


def _bond_ok(p, m, k):
    t = p[0]
    if t == "and":
        return all(_bond_ok(x, m, k) for x in p[1])
    if t == "or":
        return any(_bond_ok(x, m, k) for x in p[1])
    if t == "not":
        return not _bond_ok(p[1], m, k)
    if t == "true":
        return True
    o = m["order"][k]
    if t == "single":
        return o == 1
    if t == "double":
        return o == 2
    if t == "triple":
        return o == 3
    if t == "aromatic":
        return o == 4
    if t == "ringbond":
        return m["ringbond"][k]
    raise ValueError(f"unknown bond predicate {t}")


def _embed(q, m, anchor, cache):
    """Is there an embedding of query q (first atom mapped to anchor, or anywhere when anchor is None)?"""
    na = len(q["atoms"])
    qn = [[] for _ in range(na)]
    for a, b, bp in q["bonds"]:
        qn[a].append((b, bp))
        qn[b].append((a, bp))
    # visiting order: query atoms in index order; each non-start atom has an earlier neighbour
    parent = [None] * na
    for v in range(na):
        earlier = [(u, bp) for u, bp in qn[v] if u < v]
        if earlier:
            parent[v] = earlier[0]
    mp = [-1] * na
    used = set()

    def ok_at(v, i):
        if i in used or not _atom_ok(q["atoms"][v], m, i, cache):
            return False
        for u, bp in qn[v]:
            if u < v and mp[u] >= 0:
                k = next((kk for j, kk in m["nbr"][i] if j == mp[u]), None)
                if k is None or not _bond_ok(bp, m, k):
                    return False
        return True

    def rec(v):
        if v == na:
            return True
        if parent[v] is not None:
            cands = [j for j, _ in m["nbr"][mp[parent[v][0]]]]
        elif v == 0 and anchor is not None:
            cands = [anchor]
        else:
            cands = range(m["n"])
        for i in cands:
            if ok_at(v, i):
                mp[v] = i
                used.add(i)
                if rec(v + 1):
                    mp[v] = -1
                    used.discard(i)
                    return True
                mp[v] = -1
                used.discard(i)
        return False

    return rec(0)


def smarts_match(smiles, smarts, *, merge_hs: bool = False) -> RichResult:
    r"""SMARTS substructure search on a SMILES molecule (hydrogens implicit; which atoms can anchor the pattern).

    The molecule is read with :func:`morie.fn.avalon.parse_smiles`,
    implicit hydrogens are filled to the lowest allowed valence of the
    (isoelectronic) element, ring bonds are the non-bridges, and rings
    written in Kekule form are perceived aromatic by Hueckel's 4n + 2 rule
    on single rings and fused pairs (ring carbons with an exocyclic double
    bond to N, O or S give no electron). Supported SMARTS: organic and
    bracket atoms with ``# a A H D X v R r x`` charges, ``!``, ``&``,
    ``,`` and ``;`` logic, recursive ``$()``, bonds ``- = # : ~ @`` with
    logic, branches, ring closures and ``.``. ``merge_hs`` folds query
    hydrogens into "at least k H" counts as RDKit's ``mergeHs``. Returns
    ``matched`` and ``anchors`` (0-based atoms that can be the pattern's
    first atom).

    References
    ----------
    Daylight Chemical Information Systems. *SMARTS - A Language for
    Describing Molecular Patterns*. Daylight Theory Manual, chapter 4.
    Ullmann, J. R. (1976). An algorithm for subgraph isomorphism. *Journal
    of the ACM*, 23, 31-42.

    Examples
    --------
    >>> smarts_match("CC(=O)Oc1ccccc1C(=O)O", "[CX3](=O)[OX2H1]").anchors
    [10]
    >>> smarts_match("C1=CC=CC=C1O", "c[OH]").matched
    True
    """
    m = _molecule(smiles)
    q = _parse(str(smarts))
    if merge_hs:
        q = _merge_hs(q)
    cache = {}
    anchors = [i for i in range(m["n"]) if _embed(q, m, i, cache)]
    return RichResult(payload={"matched": bool(anchors), "anchors": anchors, "n_atoms": m["n"]})


# ---------------------------------------------------------------- Wildman-Crippen
# (type, SMARTS, logP, MR): the RDKit Crippen.cpp table (Wildman and Crippen 1999), in its matching order
_CRIPPEN = [
    ("C1", "[CH4]", 0.1441, 2.503), ("C1", "[CH3]C", 0.1441, 2.503), ("C1", "[CH2](C)C", 0.1441, 2.503),
    ("C2", "[CH](C)(C)C", 0.0, 2.433), ("C2", "[C](C)(C)(C)C", 0.0, 2.433),
    ("C3", "[CH3][N,O,P,S,F,Cl,Br,I]", -0.2035, 2.753), ("C3", "[CH2X4]([N,O,P,S,F,Cl,Br,I])[A;!#1]", -0.2035, 2.753),
    ("C4", "[CH1X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])[A;!#1]", -0.2051, 2.731),
    ("C4", "[CH0X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])([A;!#1])[A;!#1]", -0.2051, 2.731),
    ("C5", "[C]=[!C;A;!#1]", -0.2783, 5.007),
    ("C6", "[CH2]=C", 0.1551, 3.513), ("C6", "[CH1](=C)[A;!#1]", 0.1551, 3.513),
    ("C6", "[CH0](=C)([A;!#1])[A;!#1]", 0.1551, 3.513), ("C6", "[C](=C)=C", 0.1551, 3.513),
    ("C7", "[CX2]#[A;!#1]", 0.0017, 3.888), ("C8", "[CH3]c", 0.08452, 2.464), ("C9", "[CH3]a", -0.1444, 2.412),
    ("C10", "[CH2X4]a", -0.0516, 2.488), ("C11", "[CHX4]a", 0.1193, 2.582), ("C12", "[CH0X4]a", -0.0967, 2.576),
    ("C13", "[cH0]-[A;!C;!N;!O;!S;!F;!Cl;!Br;!I;!#1]", -0.5443, 4.041), ("C14", "[c][#9]", 0.0, 3.257),
    ("C15", "[c][#17]", 0.245, 3.564), ("C16", "[c][#35]", 0.198, 3.18), ("C17", "[c][#53]", 0.0, 3.104),
    ("C18", "[cH]", 0.1581, 3.35), ("C19", "[c](:a)(:a):a", 0.2955, 4.346), ("C20", "[c](:a)(:a)-a", 0.2713, 3.904),
    ("C21", "[c](:a)(:a)-C", 0.136, 3.509), ("C22", "[c](:a)(:a)-N", 0.4619, 4.067),
    ("C23", "[c](:a)(:a)-O", 0.5437, 3.853), ("C24", "[c](:a)(:a)-S", 0.1893, 2.673),
    ("C25", "[c](:a)(:a)=[C,N,O]", -0.8186, 3.135), ("C26", "[C](=C)(a)[A;!#1]", 0.264, 4.305),
    ("C26", "[C](=C)(c)a", 0.264, 4.305), ("C26", "[CH1](=C)a", 0.264, 4.305), ("C26", "[C]=c", 0.264, 4.305),
    ("C27", "[CX4][A;!C;!N;!O;!P;!S;!F;!Cl;!Br;!I;!#1]", 0.2148, 2.693), ("CS", "[#6]", 0.08129, 3.243),
    ("H1", "[#1][#6,#1]", 0.123, 1.057), ("H2", "[#1]O[CX4,c]", -0.2677, 1.395),
    ("H2", "[#1]O[!#6;!#7;!#8;!#16]", -0.2677, 1.395), ("H2", "[#1][!#6;!#7;!#8]", -0.2677, 1.395),
    ("H3", "[#1][#7]", 0.2142, 0.9627), ("H3", "[#1]O[#7]", 0.2142, 0.9627),
    ("H4", "[#1]OC=[#6,#7,O,S]", 0.298, 1.805), ("H4", "[#1]O[O,S]", 0.298, 1.805), ("HS", "[#1]", 0.1125, 1.112),
    ("N1", "[NH2+0][A;!#1]", -1.019, 2.262), ("N2", "[NH+0]([A;!#1])[A;!#1]", -0.7096, 2.173),
    ("N3", "[NH2+0]a", -1.027, 2.827), ("N4", "[NH1+0]([!#1;A,a])a", -0.5188, 3.0),
    ("N5", "[NH+0]=[!#1;A,a]", 0.08387, 1.757), ("N6", "[N+0](=[!#1;A,a])[!#1;A,a]", 0.1836, 2.428),
    ("N7", "[N+0]([A;!#1])([A;!#1])[A;!#1]", -0.3187, 1.839), ("N8", "[N+0](a)([!#1;A,a])[A;!#1]", -0.4458, 2.819),
    ("N8", "[N+0](a)(a)a", -0.4458, 2.819), ("N9", "[N+0]#[A;!#1]", 0.01508, 1.725),
    ("N10", "[NH3,NH2,NH;+,+2,+3]", -1.95, 0.0), ("N11", "[n+0]", -0.3239, 2.202), ("N12", "[n;+,+2,+3]", -1.119, 0.0),
    ("N13", "[NH0;+,+2,+3]([A;!#1])([A;!#1])([A;!#1])[A;!#1]", -0.3396, 0.2604),
    ("N13", "[NH0;+,+2,+3](=[A;!#1])([A;!#1])[!#1;A,a]", -0.3396, 0.2604),
    ("N13", "[NH0;+,+2,+3](=[#6])=[#7]", -0.3396, 0.2604), ("N14", "[N;+,+2,+3]#[A;!#1]", 0.2887, 3.359),
    ("N14", "[N;-,-2,-3]", 0.2887, 3.359), ("N14", "[N;+,+2,+3](=[N;-,-2,-3])=N", 0.2887, 3.359),
    ("NS", "[#7]", -0.4806, 2.134),
    ("O1", "[o]", 0.1552, 1.08), ("O2", "[OH,OH2]", -0.2893, 0.8238), ("O3", "[O]([A;!#1])[A;!#1]", -0.0684, 1.085),
    ("O4", "[O](a)[!#1;A,a]", -0.4195, 1.182), ("O5", "[O]=[#7,#8]", 0.0335, 3.367),
    ("O5", "[OX1;-,-2,-3][#7]", 0.0335, 3.367), ("O6", "[OX1;-,-2,-2][#16]", -0.3339, 0.7774),
    ("O6", "[O;-0]=[#16;-0]", -0.3339, 0.7774), ("O12", "[O-]C(=O)", -1.326, 0.0),
    ("O7", "[OX1;-,-2,-3][!#1;!N;!S]", -1.189, 0.0), ("O8", "[O]=c", 0.1788, 3.135),
    ("O9", "[O]=[CH]C", -0.1526, 0.0), ("O9", "[O]=C(C)([A;!#1])", -0.1526, 0.0),
    ("O9", "[O]=[CH][N,O]", -0.1526, 0.0), ("O9", "[O]=[CH2]", -0.1526, 0.0), ("O9", "[O]=[CX2]=O", -0.1526, 0.0),
    ("O10", "[O]=[CH]c", 0.1129, 0.2215), ("O10", "[O]=C([C,c])[a;!#1]", 0.1129, 0.2215),
    ("O10", "[O]=C(c)[A;!#1]", 0.1129, 0.2215), ("O11", "[O]=C([!#1;!#6])[!#1;!#6]", 0.4833, 0.389),
    ("OS", "[#8]", -0.1188, 0.6865),
    ("F", "[#9-0]", 0.4202, 1.108), ("Cl", "[#17-0]", 0.6895, 5.853), ("Br", "[#35-0]", 0.8456, 8.927),
    ("I", "[#53-0]", 0.8857, 14.02), ("Hal", "[#9,#17,#35,#53;-]", -2.996, 0.0),
    ("Hal", "[#53;+,+2,+3]", -2.996, 0.0), ("Hal", "[+;#3,#11,#19,#37,#55]", -2.996, 0.0),
    ("P", "[#15]", 0.8612, 6.92), ("S2", "[S;-,-2,-3,-4,+1,+2,+3,+5,+6]", -0.0024, 7.365),
    ("S2", "[S-0]=[N,O,P,S]", -0.0024, 7.365), ("S1", "[S;A]", 0.6482, 7.591), ("S3", "[s;a]", 0.6237, 6.691),
    ("Me1", "[#3,#11,#19,#37,#55]", -0.3808, 5.754), ("Me1", "[#4,#12,#20,#38,#56]", -0.3808, 5.754),
    ("Me1", "[#5,#13,#31,#49,#81]", -0.3808, 5.754), ("Me1", "[#14,#32,#50,#82]", -0.3808, 5.754),
    ("Me1", "[#33,#51,#83]", -0.3808, 5.754), ("Me1", "[#34,#52,#84]", -0.3808, 5.754),
    ("Me2", "[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30]", -0.0025, 0.0),
    ("Me2", "[#39,#40,#41,#42,#43,#44,#45,#46,#47,#48]", -0.0025, 0.0),
    ("Me2", "[#72,#73,#74,#75,#76,#77,#78,#79,#80]", -0.0025, 0.0),
]  # fmt: skip
_CRIPPEN_Q = None


def clogp_estimate(smiles) -> RichResult:
    r"""Wildman-Crippen logP and molar refractivity: a sum of atomic contributions over typed atoms, hydrogens included.

    Hydrogens are made explicit and every atom takes the type of the first
    SMARTS in the Wildman-Crippen table (RDKit's ``Crippen.cpp`` order)
    whose first atom it can be; ``logP = sum_i a_{t(i)}`` and ``MR = sum_i
    r_{t(i)}``. Returns ``logp``, ``mr``, the per-atom ``types`` (heavy
    atoms first, then hydrogens in the order of their heavy atom) and
    ``contributions``.

    References
    ----------
    Wildman, S. A. and Crippen, G. M. (1999). Prediction of physicochemical
    parameters by atomic contributions. *Journal of Chemical Information and
    Computer Sciences*, 39, 868-873.

    Examples
    --------
    >>> r = clogp_estimate("c1ccccc1O")
    >>> round(r.logp, 4), round(r.mr, 4)
    (1.3922, 28.1068)
    """
    global _CRIPPEN_Q
    if _CRIPPEN_Q is None:
        _CRIPPEN_Q = [_parse(s) for _, s, _, _ in _CRIPPEN]
    m = _molecule(smiles, explicit_h=True)
    cache = {}
    types, contrib = [], []
    lp, mr = 0.0, 0.0
    for i in range(m["n"]):
        t = None
        for (name, _s, a, r), q in zip(_CRIPPEN, _CRIPPEN_Q):
            if _embed(q, m, i, cache):
                t = (name, a, r)
                break
        if t is None:
            t = ("unassigned", 0.0, 0.0)
        types.append(t[0])
        contrib.append(t[1])
        lp += t[1]
        mr += t[2]
    return RichResult(payload={"logp": lp, "mr": mr, "types": types, "contributions": contrib})


def pains_filter(smiles) -> RichResult:
    r"""PAINS substructure alerts (Baell and Holloway 2010): the 480 SMARTS of the RDKit/WEHI conversion.

    Each pattern of families A (16 alerts), B (55) and C (409) is matched
    with query hydrogens merged into "at least k H" counts on their
    neighbours (as RDKit's ``FilterCatalog``). Returns ``flagged``, the
    matching alert ``names`` (in table order) and ``n_alerts``.

    References
    ----------
    Baell, J. B. and Holloway, G. A. (2010). New substructure filters for
    removal of pan assay interference compounds (PAINS) from screening
    libraries and for their exclusion in bioassays. *Journal of Medicinal
    Chemistry*, 53, 2719-2740.
    Saubern, S., Guha, R. and Baell, J. B. (2011). KNIME workflow to assess
    PAINS filters in SMARTS format. *Molecular Informatics*, 30, 847-850.

    Examples
    --------
    >>> pains_filter("Oc1ccc(CC)cc1O").names
    ['catechol_A(92)']
    >>> pains_filter("CCO").flagged
    False
    """
    m = _molecule(smiles)
    cache = {}
    names = []
    for name, sm in _PAINS:
        q = _pains_q(sm)
        if _embed(q, m, None, cache):
            names.append(name)
    return RichResult(payload={"flagged": bool(names), "names": names, "n_alerts": len(names)})


# PAINS SMARTS (name, pattern): wehi_pains.csv of the RDKit distribution (Saubern, Guha and Baell 2011)
_PAINS = [
    ('anil_di_alk_F(14)', 'c:1:c:c(:c:c:c:1-[#6;X4]-c:2:c:c:c(:c:c:2)-[#7&H2,$([#7;!H0]-[#6;X4]),$([#7](-[#6X4])-[#6X4])])-[#7&H2,$([#7;!H0]-[#6;X4]),$([#7](-[#6X4])-[#6X4])]'),
    ('hzone_anil(14)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#6]=[#7]-[#7]-[#1]'),
    ('het_5_pyrazole_OH(14)', 'c1(nn(c([c;!H0,$(c-[#6;!H0])]1)-[#8]-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#1])-[#6;X4]'),
    ('het_thio_666_A(13)', 'c:2(:c:1-[#16]-c:3:c(-[#7;!H0,$([#7]-[CH3]),$([#7]-[#6;!H0;!H1]-[#6;!H0])](-c:1:c(:c(:c:2-[#1])-[#1])-[#1])):[c;!H0,$(c~[#7](-[#1])-[#6;X4]),$(c(cc)(cc)~[#6]:[#6])](:[c;!H0,$(c(cc)(cc)~[#6]:[#6])]:[c;!H0,$(c-[#7](-[#1])-[#1]),$(c-[#8]-[#6;X4])]:c:3-[#1]))-[#1]'),
    ('styrene_A(13)', '[#6]-2-[#6]-c:1:c(:c:c:c:c:1)-[#6](-c:3:c:c:c:c:c-2:3)=[#6]-[#6]'),
    ('ene_rhod_C(13)', '[#16]-1-[#6](=[#7]-[#6]:[#6])-[#7;!H0,$([#7]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#8]),$([#7]-[#6]:[#6])]-[#6](=[#8])-[#6]-1=[#6](-[#1])-[$([#6]:[#6]:[#6]-[#17]),$([#6]:[!#6&!#1])]'),
    ('dhp_amino_CN_A(13)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#6]=[#6])-[#8]-1)-[#6](-[#1])-[#1]'),
    ('cyano_imine_C(12)', '[#8]=[#16](=[#8])-[#6](-[#6]#[#7])=[#7]-[#7]-[#1]'),
    ('thio_urea_A(12)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2'),
    ('thiophene_amino_B(12)', 'c:1:c(:c:c:c:c:1)-[#7](-[#1])-c:2:c(:c(:c(:s:2)-[$([#6]=[#8]),$([#6]#[#7]),$([#6](-[#8]-[#1])=[#6])])-[#7])-[$([#6]#[#7]),$([#6](:[#7]):[#7])]'),
    ('keto_keto_beta_B(12)', '[#6;X4]-1-[#6](=[#8])-[#7]-[#7]-[#6]-1=[#8]'),
    ('keto_phenone_A(11)', 'c:1:c-3:c(:c:c:c:1)-[#6]:2:[#7]:[!#1]:[#6]:[#6]:[#6]:2-[#6]-3=[#8]'),
    ('cyano_pyridone_C(11)', '[#6]-1(-[#6](=[#6](-[#6]#[#7])-[#6](~[#8])~[#7]~[#6]-1~[#8])-[#6](-[#1])-[#1])=[#6](-[#1])-[#6]:[#6]'),
    ('thiaz_ene_C(11)', '[#6]-,:1(=,:[#6](-!@[#6]=[#7])-,:[#16]-,:[#6](-,:[#7]-,:1)=[#8])-[$([F,Cl,Br,I]),$([#7+](:[#6]):[#6])]'),
    ('hzone_thiophene_A(11)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1]):[!#6&!#1]:[#6;!H0,$([#6]-[OH]),$([#6]-[#6;H2,H3])](:[#6]:2-[#6](-[#1])=[#7]-[#7](-[#1])-[$([#6]:1:[#7]:[#6]:[#6](-[#1]):[#16]:1),$([#6]:[#6](-[#1]):[#6]-[#1]),$([#6]:[#7]:[#6]:[#7]:[#6]:[#7]),$([#6]:[#7]:[#7]:[#7]:[#7])])'),
    ('ene_quin_methide(10)', '[!#1]:[!#1]-[#6;!H0,$([#6]-[#6]#[#7])]=[#6]-1-[#6]=,:[#6]-[#6](=[$([#8]),$([#7;!R])])-[#6]=,:[#6]-1'),
    ('het_thio_676_A(10)', 'c:1:c:c-2:c(:c:c:1)-[#6]-[#6](-c:3:c(-[#16]-2):c(:c(-[#1]):[c;!H0,$(c-[#8]),$(c-[#16;X2]),$(c-[#6;X4]),$(c-[#7;H2,H3,$([#7!H0]-[#6;X4]),$([#7](-[#6;X4])-[#6;X4])])](:c:3-[#1]))-[#1])-[#7;H2,H3,$([#7;!H0](-[#6])-[#6;X4]),$([#7](-[#6])(-[#6;X4])-[#6;X4])]'),
    ('ene_five_het_G(10)', '[#6]-1(=[#8])-[#6](=[#6](-[#1])-[$([#6]:1:[#6]:[#6]:[#6]:[#6]:[#6]:1),$([#6]:1:[#6]:[#6]:[#6]:[!#6&!#1]:1)])-[#7]=[#6](-[!#1]:[!#1]:[!#1])-[$([#16]),$([#7]-[!#1]:[!#1])]-1'),
    ('acyl_het_A(9)', '[#7+](:[!#1]:[!#1]:[!#1])-[!#1]=[#8]'),
    ('anil_di_alk_G(9)', '[#6;X4]-[#7](-[#6;X4])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6]2=,:[#7][#6]:[#6]:[!#1]2)-[#1])-[#1]'),
    ('dhp_keto_A(9)', '[#7;!H0,$([#7]-[#6;X4])]-1-[#6]=,:[#6](-[#6](=[#8])-[#6]:[#6]:[#6])-[#6](-[#6])-[#6](=[#6]-1-[#6](-[#1])(-[#1])-[#1])-[$([#6]=[#8]),$([#6]#[#7])]'),
    ('thio_urea_B(9)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2'),
    ('anil_alk_bim(9)', 'c:1:3:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2)-[#1]):n:c(-[#1]):n:3-[#6]'),
    ('imine_imine_A(9)', 'c:1:c:c-2:c(:c:c:1)-[#7]=[#6]-[#6]-2=[#7;!R]'),
    ('thio_urea_C(9)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6](=[#8])-[#6]-,:2:[!#1]:[!#6&!#1]:[#6]:[#6]-,:2'),
    ('imine_one_fives_B(9)', '[#7;!R]=[#6]-2-[#6](=[#8])-c:1:c:c:c:c:c:1-[#16]-2'),
    ('dhp_amino_CN_B(9)', '[$([#7](-[#1])-[#1]),$([#8]-[#1])]-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-c:1:c(:n(-[#6]):n:c:1)-[#8]-2'),
    ('anil_OC_no_alk_A(8)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:n:c:1-[#1])-[#8]-c:2:c:c:c:c:c:2)-[#1])-[#1]'),
    ('het_thio_66_one(8)', '[#6](=[#8])-[#6]-1=[#6]-[#7]-c:2:c(-[#16]-1):c:c:c:c:2'),
    ('styrene_B(8)', 'c:1:c:c-2:c(:c:c:1)-[#6](-c:3:c(-[$([#16;X2]),$([#6;X4])]-2):c:c:[c;!H0,$(c-[#17]),$(c-[#6;X4])](:c:3))=[#6]-[#6]'),
    ('het_thio_5_A(8)', '[#6](-[#1])(-[#1])-[#16;X2]-c:1:n:c(:c(:n:1-!@[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2)-[#1]'),
    ('anil_di_alk_ene_A(8)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2=[#6](-[#1])-c:1:c(:c:c:c:c:1)-[#16;X2]-c:3:c-2:c:c:c:c:3'),
    ('ene_rhod_D(8)', '[#16]-1-[#6](=!@[#7;!H0,$([#7]-[#7](-[#1])-[#6]:[#6])])-[#7;!H0,$([#7]-[#6]:[#7]:[#6]:[#6]:[#16])]-[#6](=[#8])-[#6]-1=[#6](-[#1])-[#6]:[#6]-[$([#17]),$([#8]-[#6]-[#1])]'),
    ('ene_rhod_E(8)', '[#16]-1-[#6](=[#8])-[#7]-[#6](=[#16])-[#6]-1=[#6](-[#1])-[#6]:[#6]'),
    ('anil_OH_alk_A(8)', 'c:1:c(:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#7](-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#1])-[#1])-[#1]'),
    ('pyrrole_C(8)', 'n1(-[#6;X4])c(c(-[#1])c(c1-[#6]:[#6])-[#1])-[#6](-[#1])-[#1]'),
    ('thio_urea_D(8)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-c:2:c:c:c:c:c:2'),
    ('thiaz_ene_D(8)', '[#7](-c:1:c:c:c:c:c:1)-c2[n+]c(cs2)-c:3:c:c:c:c:c:3'),
    ('ene_rhod_F(8)', 'n:1:c:c:c(:c:1-[#6](-[#1])-[#1])-[#6](-[#1])=[#6]-2-[#6](=[#8])-[#7]-[#6](=[!#6&!#1])-[#7]-2'),
    ('thiaz_ene_E(8)', '[#6]-,:1(=,:[#6](-[#6](-[#1])(-[#6])-[#6])-,:[#16]-,:[#6](-,:[#7;!H0,$([#7]-[#6;!H0;!H1])]-,:1)=[#8])-[#16]-[#6;R]'),
    ('het_65_B(7)', '[!#1]:,-1:[!#1]-,:2:[!#1](:[!#1]:[!#1]:[!#1]:,-1)-,:[#7](-[#1])-,:[#7](-,:[#6]-,:2=[#8])-[#6]'),
    ('keto_keto_beta_C(7)', 'c:1:c:c-2:c(:c:c:1)-[#6](=[#6](-[#6]-2=[#8])-[#6])-[#8]-[#1]'),
    ('het_66_A(7)', 'c:2:c:c:1:n:n:c(:n:c:1:c:c:2)-[#6](-[#1])(-[#1])-[#6]=[#8]'),
    ('thio_urea_E(7)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:n:c:c:c:c:2'),
    ('thiophene_amino_C(7)', '[#6](-[#1])-[#6](-[#1])(-[#1])-c:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6]-[#6]-[#6]=[#8])-[$([#6](=[#8])-[#8]),$([#6]#[#7])])-[#6](-[#1])-[#1]'),
    ('hzone_phenone(7)', '[#6](-c:1:c(:c(:[c;!H0,$(c-[#6;X4])]:c:c:1-[#1])-[#1])-[#1])(-c:2:c(:c(:[c;!H0,$(c-[#17])](:c(:c:2-[#1])-[#1]))-[#1])-[#1])=[$([#7]-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]),$([#7]-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]),$([#7]-[#7](-[#1])-[#6](=[#7]-[#1])-[#7](-[#1])-[#1]),$([#6](-[#1])-[#7])]'),
    ('ene_rhod_G(7)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c(:c:c:c:1)-[#6]:[!#1]:[#6]-[#6](-[#1])=[#6]-2-[#6](=[!#6&!#1])-[#7]-[#6](=[!#6&!#1])-[!#6&!#1]-2'),
    ('ene_cyano_B(7)', '[#6]-1(=[#6]-[#6](-c:2:c:c(:c(:n:c-1:2)-[#7](-[#1])-[#1])-[#6]#[#7])=[#6])-[#6]#[#7]'),
    ('dhp_amino_CN_C(7)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#6]:[#6])-[#8]-1)-[#6]#[#7]'),
    ('het_5_A(7)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6]=[#8])-[#6;X4]-[#6]-2=[#8]'),
    ('ene_five_het_H(6)', '[#7]-1=[#6]-[#6](-[#6](-[#7]-1)=[#16])=[#6]'),
    ('thio_amide_A(6)', 'c1(coc(c1-[#1])-[#6](=[#16])-[#7]-2-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[!#1]-[#6](-[#1])(-[#1])-[#6]-2(-[#1])-[#1])-[#1]'),
    ('ene_cyano_C(6)', '[#6]=[#6](-[#6]#[#7])-[#6](=[#7]-[#1])-[#7]-[#7]'),
    ('hzone_furan_A(6)', 'c:1(:c(:c(:[c;!H0,$(c-[#6;!H0;!H1])](:o:1))-[#1])-[#1])-[#6;!H0,$([#6]-[#6;!H0;!H1])]=[#7]-[#7](-[#1])-c:2:n:c:c:s:2'),
    ('anil_di_alk_H(6)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-[#6]:2:[#6]:[!#1]:[#6]:[#6]:[#6]:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('het_65_C(6)', 'n2c1ccccn1c(c2-[$([#6](-[!#1])=[#6](-[#1])-[#6]:[#6]),$([#6]:[#8]:[#6])])-[#7]-[#6]:[#6]'),
    ('thio_urea_F(6)', '[#6]-1-[#7](-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#7]-1-[#1]'),
    ('ene_five_het_I(6)', 'c:1(:c:c:c:o:1)-[#6](-[#1])=!@[#6]-3-[#6](=[#8])-c:2:c:c:c:c:c:2-[!#6&!#1]-3'),
    ('keto_keto_gamma(5)', '[#8]=[#6]-1-[#6;X4]-[#6]-[#6](=[#8])-c:2:c:c:c:c:c-1:2'),
    ('quinone_B(5)', 'c:1:c:c-2:c(:c:c:1)-[#6](-c3cccc4noc-2c34)=[#8]'),
    ('het_6_pyridone_OH(5)', '[#8](-[#1])-c:1:n:c(:c:c:c:1)-[#8]-[#1]'),
    ('hzone_naphth_A(5)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(:c(:c:2-[#1])-[#1])-[#6]=[#7]-[#7](-[#1])-[$([#6]:[#6]),$([#6]=[#16])])-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('thio_ester_A(5)', '[#6]-,:1=,:[#6](-,:[#16]-,:[#6](-,:[#6]=,:[#6]-,:1)=[#16])-,:[#7]'),
    ('ene_misc_A(5)', '[#6]-1=[#6]-[#6](-[#8]-[#6]-1-[#8])(-[#8])-[#6]'),
    ('cyano_pyridone_D(5)', '[#8]=[#6]-,:1-,:[#6](=,:[#6]-,:[#6](=,:[#7]-,:[#7]-,:1)-,:[#6]=[#8])-[#6]#[#7]'),
    ('het_65_Db(5)', 'c3cn1c(nc(c1-[#7]-[#6])-c:2:c:c:c:c:n:2)cc3'),
    ('het_666_A(5)', '[#7]-2-c:1:c:c:c:c:c:1-[#6](=[#7])-c:3:c-2:c:c:c:c:3'),
    ('diazox_sulfon_B(5)', 'c:1:c(:c:c:c:c:1)-[#7]-2-[#6](-[#1])-[#6](-[#1])-[#7](-[#6](-[#1])-[#6]-2-[#1])-[#16](=[#8])(=[#8])-c:3:c:c:c:c:4:n:s:n:c:3:4'),
    ('anil_NH_alk_A(5)', 'c:1(:c(:c-,:2:c(:c(:c:1-[#1])-[#1])-,:[#7](-,:[#6](-,:[#7]-,:2-[#1])=[#8])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('sulfonamide_C(5)', 'c:1(:c(:c-3:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-c:2:c:c:c(:c:c:2)-[!#6&!#1])-[#1])-[#8]-[#6](-[#8]-3)(-[#1])-[#1])-[#1])-[#1]'),
    ('het_thio_N_55(5)', '[#6](-[#1])-[#6]:2:[#7]:[#7](-c:1:c:c:c:c:c:1):[#16]:3:[!#6&!#1]:[!#1]:[#6]:[#6]:2:3'),
    ('keto_keto_beta_D(5)', '[#8]=[#6]-[#6]=[#6](-[#1])-[#8]-[#1]'),
    ('ene_rhod_H(5)', '[#7]-,:1-,:2-,:[#6](=,:[#7]-,:[#6](=[#8])-,:[#6](=,:[#7]-,:1)-[#6](-[#1])-[#1])-,:[#16]-,:[#6](=[#6](-[#1])-[#6]:[#6])-,:[#6]-,:2=[#8]'),
    ('imine_ene_A(5)', '[#6]:[#6]-[#6](-[#1])=[#6](-[#1])-[#6](-[#1])=[#7]-[#7](-[#6;X4])-[#6;X4]'),
    ('het_thio_656a(5)', 'c:1:3:c(:c:c:c:c:1):c:2:n:n:c(-[#16]-[#6](-[#1])(-[#1])-[#6]=[#8]):n:c:2:n:3-[#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1]'),
    ('pyrrole_D(5)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#1])-[#1])-[#1]'),
    ('pyrrole_E(5)', 'n2(-[#6]:1:[!#1]:[!#6&!#1]:[!#1]:[#6]:1-[#1])c(c(-[#1])c(c2-[#6;X4])-[#1])-[#6;X4]'),
    ('thio_urea_G(5)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6]([#7;R])[#7;R]'),
    ('anisol_A(5)', 'c:1(:c(:c(:c(:c(:[c;!H0,$(c-[#6](-[#1])-[#1])]:1)-[#1])-[#8]-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[$([#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]),$([#6](-[#1])(-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#1])])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('pyrrole_F(5)', 'n2(-[#6]:1:[#6](-[#6]#[#7]):[#6]:[#6]:[!#6&!#1]:1)c(c(-[#1])c(c2)-[#1])-[#1]'),
    ('dhp_amino_CN_D(5)', '[#7](-[#1])(-[#1])-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-c:1:c(:c:c:s:1)-[#8]-2'),
    ('thiazole_amine_A(4)', '[#7](-[#1])-c:1:n:c(:c:s:1)-c:2:c:n:c(-[#7](-[#1])-[#1]):s:2'),
    ('het_6_imidate_A(4)', '[#7]=[#6]-,:1-,:[#7](-[#1])-,:[#6](=,:[#6](-[#7]-[#1])-,:[#7]=,:[#7]-,:1)-[#7]-[#1]'),
    ('anil_OC_no_alk_B(4)', 'c:1:c(:c:2:c(:c:c:1):c:c:c:c:2)-[#8]-c:3:c(:c(:c(:c(:c:3-[#1])-[#1])-[#7]-[#1])-[#1])-[#1]'),
    ('styrene_C(4)', 'c:1:c:c-2:c(:c:c:1)-[#6]-[#16]-c3c(-[#6]-2=[#6])ccs3'),
    ('azulene(4)', 'c:2:c:c:c:1:c(:c:c:c:1):c:c:2'),
    ('furan_acid_A(4)', 'c:1(:c(:c(:c(:o:1)-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6]:[#6])-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('cyano_pyridone_E(4)', '[!#1]:[#6]-[#6]-,:1=,:[#6](-[#1])-,:[#6](=,:[#6](-[#6]#[#7])-,:[#6](=[#8])-,:[#7]-,:1-[#1])-[#6]:[#8]'),
    ('anil_alk_thio(4)', '[#6]-1-3=[#6](-[#6](-[#7]-c:2:c:c:c:c:c-1:2)(-[#6])-[#6])-[#16]-[#16]-[#6]-3=[!#1]'),
    ('anil_di_alk_I(4)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#6](=[#8])-c:2:c:c:c:c:c:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('het_thio_6_furan(4)', '[#6](-[#1])(-[#1])-[#16;X2]-c:1:n:n:c(:c(:n:1)-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-c:3:c(:c(:c(:o:3)-[#1])-[#1])-[#1]'),
    ('anil_di_alk_ene_B(4)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2=[#6]-c:1:c(:c:c:c:c:1)-[#6]-2(-[#1])-[#1]'),
    ('imine_one_B(4)', '[#7](-[#1])(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6](=[#8])-[#6](-[#1])-[#1])-[#7](-[#1])-[$([#7]-[#1]),$([#6]:[#6])]'),
    ('anil_OC_alk_A(4)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1]):o:c:3:c(-[#1]):c(:c(-[#8]-[#6](-[#1])-[#1]):c(:c:2:3)-[#1])-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('ene_five_het_J(4)', '[#16]=[#6]-,:1-,:[#7](-[#1])-,:[#6]=,:[#6]-,:[#6]-2=,:[#6]-,:1-[#6](=[#8])-[#8]-[#6]-2=[#6]-[#1]'),
    ('pyrrole_G(4)', 'n2(-c:1:c(:c:c(:c(:c:1)-[#1])-[$([#7](-[#1])-[#1]),$([#6]:[#7])])-[#1])c(c(-[#1])c(c2-[#1])-[#1])-[#1]'),
    ('ene_five_het_K(4)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])=[#6]-2-[#6](=[#8])-[!#6&!#1]-[#6]=,:[!#1]-2)-[#1])-[#1]'),
    ('cyano_ene_amine_B(4)', '[#6]=[#6]-[#6](-[#6]#[#7])(-[#6]#[#7])-[#6](-[#6]#[#7])=[#6]-[#7](-[#1])-[#1]'),
    ('thio_ester_B(4)', '[#6]:[#6]-[#6](=[#16;X1])-[#16;X2]-[#6](-[#1])-[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('ene_five_het_L(4)', '[#8]=[#6]-3-[#6](=!@[#6](-[#1])-c:1:c:n:c:c:1)-c:2:c:c:c:c:c:2-[#7]-3'),
    ('hzone_thiophene_B(4)', 'c:1(:[c;!H0,$(c-[#6;!H0;!H1])](:c(:c(:s:1)-[#1])-[#1]))-[#6](-[#1])=[#7]-[#7](-[#1])-c:2:c:c:c:c:c:2'),
    ('dhp_amino_CN_E(4)', '[#6](-[#1])(-[#1])-[#16;X2]-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](-[#6]#[#7])-[#6](=[#8])-[#7]-1'),
    ('het_5_B(4)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#7](-[#1])-[#6]=[#8])-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('imine_imine_B(3)', '[#6]:[#6]-[#6](-[#1])=[#6](-[#1])-[#6](-[#1])=[#7]-[#7]=[#6]'),
    ('thiazole_amine_B(3)', 'c:1(:c:c:c(:c:c:1)-[#6](-[#1])-[#1])-c:2:c(:s:c(:n:2)-[#7](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]'),
    ('imine_ene_one_A(3)', '[#6]-2(-[#6]=[#7]-c:1:c:c:c:c:c:1-[#7]-2)=[#6](-[#1])-[#6]=[#8]'),
    ('diazox_A(3)', '[#8](-c:1:c:c:c:c:c:1)-c:3:c:c:2:n:o:n:c:2:c:c:3'),
    ('ene_one_A(3)', '[!#1]:1:[!#1]:[!#1]:[!#1](:[!#1]:[!#1]:1)-[#6](-[#1])=[#6](-[#1])-[#6](-[#7]-c:2:c:c:c:3:c(:c:2):c:c:c(:n:3)-[#7](-[#6])-[#6])=[#8]'),
    ('anil_OC_no_alk_C(3)', '[#7](-[#1])(-[#1])-c:1:c(:c:c:c:n:1)-[#8]-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('thiazol_SC_A(3)', '[#6]-[#16;X2]-c:1:n:c(:c:s:1)-[#1]'),
    ('het_666_B(3)', 'c:1:c-3:c(:c:c:c:1)-[#7](-c:2:c:c:c:c:c:2-[#8]-3)-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('furan_A(3)', 'c:1(:c(:c(:c(:o:1)-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#8]-[#1])-[#6]#[#6]-[#6;X4]'),
    ('colchicine_A(3)', '[#6]-1(-[#6](=[#6]-[#6]=[#6]-[#6]=[#6]-1)-[#7]-[#1])=[#7]-[#6]'),
    ('thiophene_C(3)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])=[#6]-[#6](=[#8])-c:1:c(-[#16;X2]):s:c(:c:1)-[$([#6]#[#7]),$([#6]=[#8])]'),
    ('anil_OC_alk_B(3)', 'c:1:c(:c:c:c:c:1)-[#7]-2-[#6](=[#8])-[#6](=[#6](-[F,Cl,Br,I])-[#6]-2=[#8])-[#7](-[#1])-[#6]:3:[#6]:[#6]:[#6](-[#8]-[#6](-[#1])-[#1]):[#6]:[#6]:3'),
    ('het_thio_66_A(3)', 'c:1-2:c(:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]=[#6]-2-[#16;X2]-[#6](-[#1])(-[#1])-[#6](=[#8])-c:3:c:c:c:c:c:3'),
    ('rhod_sat_B(3)', '[#7]-2(-c:1:c:c:c:c:c:1-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#6](-[#1])(-[#1])-[!#1]:[!#1]:[!#1]:[!#1]:[!#1])-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('ene_rhod_I(3)', '[#7]-2(-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](=[#6](-[#1])-c:1:c:c:c:c(:c:1)-[Br])-[#6]-2=[#8]'),
    ('keto_thiophene(3)', 'c:1(:c(:c:2:c(:s:1):c:c:c:c:2)-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('imine_imine_C(3)', '[#7](-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])=[#7]-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#6](-[#1])-[#1])-[#6]:[#6]'),
    ('het_65_pyridone_A(3)', '[#6]:2(:[#6](-[#6](-[#1])-[#1]):[#6]-,:1:[#6](-,:[#7]=,:[#6;!H0,$([#6]-[#16]-[#6](-[#1])-[#1])](-,:[#7](-,:[#6]-,:1=[!#6&!#1;X1])-[#6](-[#1])-[$([#6](=[#8])-[#8]),$([#6]:[#6])])):[!#6&!#1;X2]:2)-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('thiazole_amine_C(3)', 'c:1(:n:c(:c(-[#1]):s:1)-[!#1]:[!#1]:[!#1](-[$([#8]-[#6](-[#1])-[#1]),$([#6](-[#1])-[#1])]):[!#1]:[!#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c(-[#1]):c(:c(-[#1]):o:2)-[#1]'),
    ('het_thio_pyr_A(3)', 'n:1:c(:c(:c(:c(:c:1-[#16]-[#6]-[#1])-[#6]#[#7])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])-[#1])-[#1])-[#6]:[#6]'),
    ('melamine_A(3)', 'c:1:4:c(:n:c(:n:c:1-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-[#7](-[#1])-c:3:c:[c;!H0,$(c-[#6](-[#1])-[#1]),$(c-[#16;X2]),$(c-[#8]-[#6]-[#1]),$(c-[#7;X3])](:[c;!H0,$(c-[#6](-[#1])-[#1]),$(c-[#16;X2]),$(c-[#8]-[#6]-[#1]),$(c-[#7;X3])](:c:[c;!H0,$(c-[#6](-[#1])-[#1]),$(c-[#16;X2]),$(c-[#8]-[#6]-[#1]),$(c-[#7;X3])]:3))):c:c:c:c:4'),
    ('anil_NH_alk_B(3)', '[#7](-[#1])(-[#6]:1:[#6]:[#6]:[!#1]:[#6]:[#6]:1)-c:2:c:c:c(:c:c:2)-[#7](-[#1])-[#6]-[#1]'),
    ('rhod_sat_C(3)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#7]-[#6]=[#8])-[#16]-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('thiophene_amino_D(3)', '[#6]=[#6]-[#6](=[#8])-[#7]-c:1:c(:c(:c(:s:1)-[#6](=[#8])-[#8])-[#6]-[#1])-[#6]#[#7]'),
    ('anil_OC_alk_C(3)', '[#8;!H0,$([#8]-[#6](-[#1])-[#1])]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:n:c:c:n:2'),
    ('het_thio_65_A(3)', '[#6](-[#1])(-[#1])-[#16;X2]-c3nc1c(n(nc1-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2)nn3'),
    ('het_thio_656b(3)', '[#6]-[#6](=[#8])-[#6](-[#1])(-[#1])-[#16;X2]-c:3:n:n:c:2:c:1:c(:c(:c(:c(:c:1:n(:c:2:n:3)-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('thiazole_amine_D(3)', 's:1:c(:[n+](-[#6](-[#1])-[#1]):c(:c:1-[#1])-[#6])-[#7](-[#1])-c:2:c:c:c:c:c:2[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('thio_urea_H(3)', '[#6]-,:2(=[#16])-,:[#7](-[#6](-[#1])(-[#1])-c:1:c:c:c:o:1)-,:[#6](=,:[#7]-,:[#7]-,:2-[#1])-[#6]:[#6]'),
    ('cyano_pyridone_F(3)', '[#7]-,:2(-c:1:c:c:c:c:c:1)-,:[#6](=[#8])-,:[#6](=,:[#6]-,:[#6](=,:[#7]-,:2)-[#6]#[#7])-[#6]#[#7]'),
    ('rhod_sat_D(3)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6])-[#6]-2=[#8]'),
    ('ene_rhod_J(3)', '[#6](-[#1])(-[#1])-[#7]-2-[#6](=[$([#16]),$([#7])])-[!#6&!#1]-[#6](=[#6]-1-[#6](=[#6](-[#1])-[#6]:[#6]-[#7]-1-[#6](-[#1])-[#1])-[#1])-[#6]-2=[#8]'),
    ('imine_phenol_A(3)', '[#6]=[#7;!R]-c:1:c:c:c:c:c:1-[#8]-[#1]'),
    ('thio_carbonate_B(3)', '[#8]=[#6]-,:2-,:[#16]-,:c:1:c(:c(:c:c:c:1)-[#8]-[#6](-[#1])-[#1])-,:[#8]-,:2'),
    ('het_thio_N_5A(3)', '[#7]=,:[#6]-,:1-,:[#7]=,:[#6]-,:[#7]-,:[#16]-,:1'),
    ('het_thio_N_65A(3)', '[#7]-,:2-,:[#16]-,:[#6]-1=,:[#6](-[#6]:[#6]-[#7]-[#6]-1)-,:[#6]-,:2=[#16]'),
    ('anil_di_alk_J(3)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])=[#7]-[#7]=[#6](-[#6])-[#6]:[#6])-[#1])-[#1]'),
    ('pyrrole_H(3)', 'n1-2cccc1-[#6]=[#7](-[#6])-[#6]-[#6]-2'),
    ('ene_cyano_D(3)', '[#6](-[#6]#[#7])(-[#6]#[#7])=[#6](-[#16])-[#16]'),
    ('cyano_cyano_B(3)', '[#6]-1(-[#6]#[#7])(-[#6]#[#7])-[#6](-[#1])(-[#6](=[#8])-[#6])-[#6]-1-[#1]'),
    ('ene_five_het_M(3)', '[#6]-1=,:[#6]-[#6](-[#6](-[$([#8]),$([#16])]-1)=[#6]-[#6]=[#8])=[#8]'),
    ('cyano_ene_amine_C(3)', '[#6]:[#6]-[#6](=[#8])-[#7](-[#1])-[#6](=[#8])-[#6](-[#6]#[#7])=[#6](-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('thio_urea_I(3)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#7]=[#6]-c:2:c:n:c:c:2'),
    ('dhp_amino_CN_F(3)', '[#7](-[#1])(-[#1])-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-c:1:c:c:c:s:1)-[#6](=[#6](-[#6](-[#1])-[#1])-[#8]-2)-[#6](=[#8])-[#8]-[#6]'),
    ('anthranil_acid_B(3)', 'c:1:c-3:c(:c:c(:c:1)-[#6](=[#8])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#6](=[#8])-[#8]-[#1])-[#6](-[#7](-[#6]-3=[#8])-[#6](-[#1])-[#1])=[#8]'),
    ('diazox_B(3)', '[Cl]-c:2:c:c:1:n:o:n:c:1:c:c:2'),
    ('thio_aldehyd_A(3)', '[#6]-[#6](=[#16])-[#1]'),
    ('thio_amide_B(2)', '[#6;X4]-[#7](-[#1])-[#6](-[#6]:[#6])=[#6](-[#1])-[#6](=[#16])-[#7](-[#1])-c:1:c:c:c:c:c:1'),
    ('imidazole_B(2)', '[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#16]-[#6](-[#1])(-[#1])-c1cn(cn1)-[#1]'),
    ('thiazole_amine_E(2)', '[#8]=[#6]-[#7](-[#1])-c:1:c(-[#6]:[#6]):n:c(-[#6](-[#1])(-[#1])-[#6]#[#7]):s:1'),
    ('thiazole_amine_F(2)', '[#6](-[#1])-[#7](-[#1])-c:1:n:c(:c:s:1)-c2cnc3n2ccs3'),
    ('thio_ester_C(2)', '[#7]-,:1-,:[#6](=[#8])-,:[#6](=,:[#6](-[#6])-,:[#16]-,:[#6]-,:1=[#16])-[#1]'),
    ('ene_one_B(2)', '[#6](-[#16])(-[#7])=[#6](-[#1])-[#6]=[#6](-[#1])-[#6]=[#8]'),
    ('quinone_C(2)', '[#8]=[#6]-3-c:1:c(:c:c:c:c:1)-[#6]-,:2=,:[#6](-[#8]-[#1])-,:[#6](=[#8])-,:[#7]-,:c:4:c-,:2:c-3:c:c:c:4'),
    ('keto_naphthol_A(2)', 'c:1:2:c:c:c:c(:c:1:c(:c:c:c:2)-[$([#8]-[#1]),$([#7](-[#1])-[#1])])-[#6](-[#6])=[#8]'),
    ('thio_amide_C(2)', '[#6](-[#1])(-c:1:c:c:c:c:c:1)(-c:2:c:c:c:c:c:2)-[#6](=[#16])-[#7]-[#1]'),
    ('phthalimide_misc(2)', '[#7]-2(-[#6](=[#8])-c:1:c(:c(:c(:c(:c:1-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1])-[#6]-2=[#8])-c:3:c(:c:c(:c(:c:3)-[#1])-[#8])-[#1]'),
    ('sulfonamide_D(2)', 'c:1:c:c(:c:c:c:1-[#7](-[#1])-[#16](=[#8])=[#8])-[#7](-[#1])-[#16](=[#8])=[#8]'),
    ('anil_NH_alk_C(2)', '[#6](-[#1])-[#7](-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6]-[#1]'),
    ('het_65_E(2)', 's1c(c(c-,:2c1-,:[#7](-[#1])-,:[#6](-,:[#6](=,:[#6]-,:2-[#1])-[#6](=[#8])-[#8]-[#1])=[#8])-[#7](-[#1])-[#1])-[#6](=[#8])-[#7]-[#1]'),
    ('hzide_naphth(2)', 'c:2(:c:1:c(:c(:c(:c(:c:1:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#7](-[#1])-[#6]=[#8])-[#1])-[#1])-[#1]'),
    ('anisol_B(2)', '[#6](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6;X4])-[#1]'),
    ('thio_carbam_ene(2)', '[#6]-1=[#6]-[#7]-[#6](-[#16]-[#6;X4]-1)=[#16]'),
    ('thio_amide_D(2)', '[#6](-[#7](-[#6]-[#1])-[#6]-[#1]):[#6]-[#7](-[#1])-[#6](=[#16])-[#6]-[#1]'),
    ('het_65_Da(2)', 'n2nc(c1cccc1c2-[#6])-[#6]'),
    ('thiophene_D(2)', 's:1:c(:c(-[#1]):c(:c:1-[#6](=[#8])-[#7](-[#1])-[#7]-[#1])-[#8]-[#6](-[#1])-[#1])-[#1]'),
    ('het_thio_6_ene(2)', '[#6]-1:[#6]-[#7]=[#6]-[#6](=[#6]-[#7]-[#6])-[#16]-1'),
    ('cyano_keto_A(2)', '[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6]#[#7])-[#6](=[#8])-[#6]'),
    ('anthranil_acid_C(2)', 'c2(c(-[#7](-[#1])-[#1])n(-c:1:c:c:c:c:c:1-[#6](=[#8])-[#8]-[#1])nc2-[#6]=[#8])-[$([#6]#[#7]),$([#6]=[#16])]'),
    ('naphth_amino_C(2)', 'c:2:c:1:c:c:c:c-,:3:c:1:c(:c:c:2)-,:[#7](-,:[#7]=,:[#6]-,:3)-[#1]'),
    ('naphth_amino_D(2)', 'c:2:c:1:c:c:c:c-,:3:c:1:c(:c:c:2)-,:[#7]-,:[#7]=,:[#7]-,:3'),
    ('thiazole_amine_G(2)', 'c1csc(n1)-[#7]-[#7]-[#16](=[#8])=[#8]'),
    ('het_66_B(2)', 'c:1:c:c:c:2:c(:c:1):n:c(:n:c:2)-[#7](-[#1])-[#6]-3=[#7]-[#6](-[#6]=[#6]-[#7]-3-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('coumarin_A(2)', 'c:1-,:3:c(:c(:c(:c(:c:1)-[#8]-[#6]-[#1])-[#1])-[#1])-,:c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-,:[#6](=[#8])-,:[#8]-,:3'),
    ('anthranil_acid_D(2)', 'c:12:c(:c:c:c:n:1)c(c(-[#6](=[#8])~[#8;X1])s2)-[#7](-[#1])-[#1]'),
    ('het_66_C(2)', 'c:1:2:n:c(:c(:n:c:1:[#6]:[#6]:[#6]:[!#1]:2)-[#6](-[#1])=[#6](-[#8]-[#1])-[#6])-[#6](-[#1])=[#6](-[#8]-[#1])-[#6]'),
    ('thiophene_amino_E(2)', 'c1csc(c1-[#7](-[#1])-[#1])-[#6](-[#1])=[#6](-[#1])-c2cccs2'),
    ('het_6666_A(2)', 'c:2:c:c:1:n:c:3:c(:n:c:1:c:c:2):c:c:c:4:c:3:c:c:c:c:4'),
    ('sulfonamide_E(2)', '[#6]:[#6]-[#7](-[#1])-[#16](=[#8])(=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('anil_di_alk_K(2)', 'c:1:c:c(:c:c:c:1-[#7](-[#1])-[#1])-[#7](-[#6;X3])-[#6;X3]'),
    ('het_5_C(2)', '[#7]-2=[#6](-c:1:c:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#6](-[#8]-[#1])(-[#6](-[#9])(-[#9])-[#9])-[#7]-2-[$([#6]:[#6]:[#6]:[#6]:[#6]:[#6]),$([#6](=[#16])-[#6]:[#6]:[#6]:[#6]:[#6]:[#6])]'),
    ('ene_six_het_B(2)', 'c:1:c(:c:c:c:c:1)-[#6](=[#8])-[#6](-[#1])=[#6]-,:3-,:[#6](=[#8])-,:[#7](-[#1])-,:[#6](=[#8])-,:[#6](=[#6](-[#1])-c:2:c:c:c:c:c:2)-,:[#7]-,:3-[#1]'),
    ('steroid_A(2)', '[#8]=[#6]-4-[#6]-[#6]-[#6]-3-[#6]-2-[#6](=[#8])-[#6]-[#6]-1-[#6]-[#6]-[#6]-[#6]-1-[#6]-2-[#6]-[#6]-[#6]-3=[#6]-4'),
    ('het_565_A(2)', 'c:1:2:c:3:c(:c(-[#8]-[#1]):c(:c:1:c(:c:n:2-[#6])-[#6]=[#8])-[#1]):n:c:n:3'),
    ('thio_imine_ium(2)', '[#6;X4]-[#7+](-[#6;X4]-[#8]-[#1])=[#6]-[#16]-[#6]-[#1]'),
    ('anthranil_acid_E(2)', '[#6]-3(=[#8])-[#6](=[#6](-[#1])-[#7](-[#1])-c:1:c:c:c:c:c:1-[#6](=[#8])-[#8]-[#1])-[#7]=[#6](-c:2:c:c:c:c:c:2)-[#8]-3'),
    ('hzone_furan_B(2)', 'c:1(:c(:c(:[c;!H0,$(c-[#6;!H0;!H1])](:o:1))-[#1])-[#1])-[#6;!H0,$([#6]-[#6;!H0;!H1])]=[#7]-[#7](-[#1])-c:2:c:c:n:c:c:2'),
    ('thiophene_E(2)', 'c:1(:c(:c(:[c;!H0,$(c-[#6;!H0;!H1])](:s:1))-[#1])-[#1])-[#6;!H0,$([#6]-[#6;!H0;!H1])]-[#6](=[#8])-[#7](-[#1])-c:2:n:c:c:s:2'),
    ('ene_misc_B(2)', '[#6]:[#6]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6]=[#8])-[#7]-2-[#6](=[#8])-[#6]-1(-[#1])-[#6](-[#1])(-[#1])-[#6]=[#6]-[#6](-[#1])(-[#1])-[#6]-1(-[#1])-[#6]-2=[#8]'),
    ('het_thio_5_B(2)', '[#6]-1(-[#6]=[#8])(-[#6]:[#6])-[#16;X2]-[#6]=[#7]-[#7]-1-[#1]'),
    ('thiophene_amino_F(2)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-c:2:c:c:c:c:c:2)-[#6]#[#7])-[#6]:3:[!#1]:[!#1]:[!#1]:[!#1]:[!#1]:3'),
    ('anil_OC_alk_D(2)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2-[$([#6](-[#1])-[#1]),$([#8]-[#6](-[#1])-[#1])]'),
    ('tert_butyl_A(2)', '[#6](-[#1])(-[#1])(-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-c:1:c(:c:c(:c(:c:1-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6](-[#1])-[#7])-[#1]'),
    ('thio_urea_J(2)', 'c:1(:c(:o:c:c:1)-[#6]-[#1])-[#6]=[#7]-[#7](-[#1])-[#6](=[#16])-[#7]-[#1]'),
    ('het_thio_65_B(2)', '[#7](-[#1])-c1nc(nc2nnc(n12)-[#16]-[#6])-[#7](-[#1])-[#6]'),
    ('coumarin_B(2)', 'c:1-,:2:c(:c:c:c:c:1-[#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1])-,:[#6](=,:[#6](-[#6](=[#8])-[#7](-[#1])-[#6]:[#6])-,:[#6](=[#8])-,:[#8]-,:2)-[#1]'),
    ('thio_urea_K(2)', '[#6]-,:2(=[#16])-,:[#7]-,:1-,:[#6]=,:[#6]-,:[#7]=,:[#7]-,:[#6]-,:1=,:[#7]-,:[#7]-,:2-[#1]'),
    ('thiophene_amino_G(2)', '[#6]:[#6]:[#6]:[#6]:[#6]:[#6]-c:1:c:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6])-[#6](=[#8])-[#8]-[#1]'),
    ('anil_NH_alk_D(2)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c:c:1-[#7](-[#1])-[#6](-[#1])(-[#6])-[#6](-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('het_thio_5_C(2)', '[#16]=[#6]-,:2-,:[#7](-[#1])-,:[#7]=,:[#6](-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-,:[#8]-,:2'),
    ('thio_keto_het(2)', '[#16]=[#6]-c:1:c:c:c:2:c:c:c:c:n:1:2'),
    ('het_thio_N_5B(2)', '[#6]~1~[#6](~[#7]~[#7]~[#6](~[#6](-[#1])-[#1])~[#6](-[#1])-[#1])~[#7]~[#16]~[#6]~1'),
    ('quinone_D(2)', '[#6]-1(-[#6]=,:[#6]-[#6]=,:[#6]-[#6]-1=[!#6&!#1])=[!#6&!#1]'),
    ('anil_di_alk_furan_B(2)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(-[#1]):c(:c(:o:1)-[#6](-[#1])=[#6]-[#6]#[#7])-[#1]'),
    ('ene_six_het_C(2)', '[#8]=[#6]-1-[#6]:[#6]-[#6](-[#1])(-[#1])-[#7]-[#6]-1=[#6]-[#1]'),
    ('het_55_A(2)', '[#6]:[#6]-[#7]:2:[#7]:[#6]:1-[#6](-[#1])(-[#1])-[#16;X2]-[#6](-[#1])(-[#1])-[#6]:1:[#6]:2-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])=[#6]-[#1]'),
    ('het_thio_65_C(2)', 'n:1:c(:n(:c:2:c:1:c:c:c:c:2)-[#6](-[#1])-[#1])-[#16]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6](-[#1])-[#6](-[#1])=[#6]-[#1]'),
    ('hydroquin_A(2)', 'c:1(:c:c(:c(:c:c:1)-[#8]-[#1])-[#6](=!@[#6]-[#7])-[#6]=[#8])-[#8]-[#1]'),
    ('anthranil_acid_F(2)', 'c:1(:c:c(:c(:c:c:1)-[#7](-[#1])-[#6](=[#8])-[#6]:[#6])-[#6](=[#8])-[#8]-[#1])-[#8]-[#1]'),
    ('pyrrole_I(2)', 'n2(-[#6](-[#1])-[#1])c-1c(-[#6]:[#6]-[#6]-1=[#8])cc2-[#6](-[#1])-[#1]'),
    ('thiophene_amino_H(2)', '[#6](-[#1])-[#7](-[#1])-c:1:c(:c(:c(:s:1)-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('imine_one_fives_C(2)', '[#6]:[#6]-[#7;!R]=[#6]-2-[#6](=[!#6&!#1])-c:1:c:c:c:c:c:1-[#7]-2'),
    ('keto_phenone_zone_A(2)', 'c:1:c:c:c:c:c:1-[#6](=[#8])-[#7](-[#1])-[#7]=[#6]-3-c:2:c:c:c:c:c:2-c:4:c:c:c:c:c-3:4'),
    ('dyes7A(2)', 'c:1:c(:c:c:c:c:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])=[#6](-[#1])-[#6]=!@[#6](-[#1])-[#6](-[#1])=[#6]-[#6]=@[#7]-c:2:c:c:c:c:c:2'),
    ('het_pyridiniums_B(2)', '[#6]:1:2:[!#1]:[#7+](:[!#1]:[#6;!H0,$([#6]-[*])](:[!#1]:1:[#6]:[#6]:[#6]:[#6]:2))~[#6]:[#6]'),
    ('het_5_D(2)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#16]-[#6])-[#6]-2=[#8]'),
    ('thiazole_amine_H(1)', 'c:1:c:c:c(:c:c:1-[#7](-[#1])-c2nc(c(-[#1])s2)-c:3:c:c:c(:c:c:3)-[#6](-[#1])(-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('thiazole_amine_I(1)', '[#6](-[#1])(-[#1])-[#7](-[#1])-[#6]=[#7]-[#7](-[#1])-c1nc(c(-[#1])s1)-[#6]:[#6]'),
    ('het_thio_N_5C(1)', '[#6]:[#6]-[#7](-[#1])-[#6](=[#8])-c1c(snn1)-[#7](-[#1])-[#6]:[#6]'),
    ('sulfonamide_F(1)', '[#8]=[#16](=[#8])(-[#6]:[#6])-[#7](-[#1])-c1nc(cs1)-[#6]:[#6]'),
    ('thiazole_amine_J(1)', '[#8]=[#16](=[#8])(-[#6]:[#6])-[#7](-[#1])-[#7](-[#1])-c1nc(cs1)-[#6]:[#6]'),
    ('het_65_F(1)', 's2c:1:n:c:n:c(:c:1c(c2-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7]-[#7]=[#6]-c3ccco3'),
    ('keto_keto_beta_E(1)', '[#6](=[#8])-[#6](-[#1])=[#6](-[#8]-[#1])-[#6](-[#8]-[#1])=[#6](-[#1])-[#6](=[#8])-[#6]'),
    ('ene_five_one_B(1)', 'c:2(:c:1-[#6](-[#6](-[#6](-c:1:c(:c(:c:2-[#1])-[#1])-[#1])(-[#1])-[#1])=[#8])=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]'),
    ('keto_keto_beta_zone(1)', '[#6]:[#6]-[#7](-[#1])-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#1])-[#6]:[#6]'),
    ('thio_urea_L(1)', '[#6;X4]-[#16;X2]-[#6](=[#7]-[!#1]:[!#1]:[!#1]:[!#1])-[#7](-[#1])-[#7]=[#6]'),
    ('het_thio_urea_ene(1)', '[#6]-1(=[#7]-[#7](-[#6](-[#16]-1)=[#6](-[#1])-[#6]:[#6])-[#6]:[#6])-[#6]=[#8]'),
    ('cyano_amino_het_A(1)', 'c:1(:c(:c:2:c(:n:c:1-[#7](-[#1])-[#1]):c:c:c(:c:2-[#7](-[#1])-[#1])-[#6]#[#7])-[#6]#[#7])-[#6]#[#7]'),
    ('tetrazole_hzide(1)', '[!#1]:1:[!#1]:[!#1]:[!#1](:[!#1]:[!#1]:1)-[#6](-[#1])=[#6](-[#1])-[#6](-[#7](-[#1])-[#7](-[#1])-c2nnnn2-[#6])=[#8]'),
    ('imine_naphthol_A(1)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(:c(:c:2-[#1])-[#1])-[#6](=[#7]-[#6]:[#6])-[#6](-[#1])-[#1])-[#8]-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('misc_anisole_A(1)', 'c:1(:c(:c:2:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1]):c(:c(:c(:c:2-[#7](-[#1])-[#6](-[#1])(-[#1])-[#1])-[#1])-c:3:c(:c(:c(:c(:c:3-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('het_thio_665(1)', 'c:1:c:c-2:c(:c:c:1)-[#16]-c3c(-[#7]-2)cc(s3)-[#6](-[#1])-[#1]'),
    ('anil_di_alk_L(1)', 'c:1:c:c:c-2:c(:c:1)-[#6](-[#6](-[#7]-2-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-4-[#6](-c:3:c:c:c:c:c:3-[#6]-4=[#8])=[#8])(-[#1])-[#1])(-[#1])-[#1]'),
    ('colchicine_B(1)', 'c:1(:c:c:c(:c:c:1)-[#6]-,:3=,:[#6]-,:[#6](-,:c2cocc2-,:[#6](=,:[#6]-,:3)-[#8]-[#1])=[#8])-[#16]-[#6](-[#1])-[#1]'),
    ('misc_aminoacid_A(1)', '[#6;X4]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#16]-[#6](-[#1])(-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1]'),
    ('imidazole_amino_A(1)', 'n:1:c(:n(:c(:c:1-c:2:c:c:c:c:c:2)-c:3:c:c:c:c:c:3)-[#7]=!@[#6])-[#7](-[#1])-[#1]'),
    ('phenol_sulfite_A(1)', '[#6](-c:1:c:c:c(:c:c:1)-[#8]-[#1])(-c:2:c:c:c(:c:c:2)-[#8]-[#1])-[#8]-[#16](=[#8])=[#8]'),
    ('het_66_D(1)', 'c:2:c:c:1:n:c(:c(:n:c:1:c:c:2)-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]:[#6])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]:[#6]'),
    ('misc_anisole_B(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c(-[#6](-[#1])-[#1])c:c:2'),
    ('tetrazole_A(1)', '[#6](-[#1])(-[#1])-c1nnnn1-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#1])-[#1]'),
    ('het_65_G(1)', '[#6]-2(=[#7]-c1c(c(nn1-[#6](-[#6]-2(-[#1])-[#1])=[#8])-[#7](-[#1])-[#1])-[#7](-[#1])-[#1])-[#6]'),
    ('misc_trityl_A(1)', '[#6](-[#6]:[#6])(-[#6]:[#6])(-[#6]:[#6])-[#16]-[#6]:[#6]-[#6](=[#8])-[#8]-[#1]'),
    ('misc_pyridine_OC(1)', '[#8]=[#6](-c:1:c(:c(:n:c(:c:1-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('het_6_hydropyridone(1)', '[#7]-1=[#6](-[#7](-[#6](-[#6](-[#6]-1(-[#1])-[#6]:[#6])(-[#1])-[#1])=[#8])-[#1])-[#7]-[#1]'),
    ('misc_stilbene(1)', '[#6]-1(=[#6](-[#6](-[#6](-[#6](-[#6]-1(-[#1])-[#1])(-[#1])-[#6](=[#8])-[#6])(-[#1])-[#6](=[#8])-[#8]-[#1])(-[#1])-[#1])-[#6]:[#6])-[#6]:[#6]'),
    ('misc_imidazole(1)', '[#6](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[Cl])-[#1])-[#1])(-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[Cl])-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c3nc(c(n3-[#6](-[#1])(-[#1])-[#1])-[#1])-[#1]'),
    ('anil_NH_no_alk_A(1)', 'n:1:c(:c(:c(:c(:c:1-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('het_6_imidate_B(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#8]-[#1])-[#6]-,:2=,:[#6](-,:[#8]-,:[#6](-,:[#7]=,:[#7]-,:2)=[#7])-[#7](-[#1])-[#1]'),
    ('anil_alk_B(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('styrene_anil_A(1)', 'c:1:c:c-3:c(:c:c:1)-c:2:c:c:c(:c:c:2-[#6]-3=[#6](-[#1])-[#6])-[#7](-[#1])-[#1]'),
    ('misc_aminal_acid(1)', 'c:1:c:c-2:c(:c:c:1)-[#7](-[#6](-[#8]-[#6]-2)(-[#6](=[#8])-[#8]-[#1])-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])-[#1]'),
    ('anil_no_alk_D(1)', 'n:1:c(:c(:c(:c(:c:1-[#7](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('anil_alk_C(1)', '[#7](-[#1])(-c:1:c:c:c:c:c:1)-[#6](-[#6])(-[#6])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('misc_anisole_C(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6]-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])-[#1])-[#6]:[#6]'),
    ('het_465_misc(1)', 'c:1-2:c:c-3:c(:c:c:1-[#8]-[#6]-[#8]-2)-[#6]-[#6]-3'),
    ('anthranil_acid_G(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('anil_di_alk_M(1)', 'c:1(:c:4:c(:n:c(:c:1-[#6](-[#1])(-[#1])-[#7]-3-c:2:c(:c(:c(:c(:c:2-[#6](-[#1])(-[#1])-[#6]-3(-[#1])-[#1])-[#1])-[#1])-[#1])-[#1])-[#1]):c(:c(:c(:c:4-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('anthranil_acid_H(1)', 'c:1:c(:c2:c(:c:c:1)c(c(n2-[#1])-[#6]:[#6])-[#6]:[#6])-[#6](=[#8])-[#8]-[#1]'),
    ('thio_urea_M(1)', '[#6]:[#6]-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-c:1:c(:c(:c(:c(:c:1-[F,Cl,Br,I])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('thiazole_amine_K(1)', 'n:1:c3:c(:c:c2:c:1nc(s2)-[#7])sc(n3)-[#7]'),
    ('het_thio_5_imine_A(1)', '[#7]=[#6]-1-[#16]-[#6](=[#7])-[#7]=[#6]-1'),
    ('thio_amide_E(1)', 'c:1:c(:n:c:c:c:1)-[#6](=[#16])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#8]-[#6](-[#1])-[#1]'),
    ('het_thio_676_B(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#6](-c:3:c(-[#16]-[#6]-2(-[#1])-[#1]):c(:c(-[#1]):c(:c:3-[#1])-[#1])-[#1])-[#8]-[#6]:[#6])-[#1])-[#1])-[#1])-[#1]'),
    ('sulfonamide_G(1)', '[#6](-[#1])(-[#1])(-[#1])-c:1:c(:c(:c(:c(:n:1)-[#7](-[#1])-[#16](-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])(=[#8])=[#8])-[#1])-[#1])-[#1]'),
    ('thio_thiomorph_Z(1)', '[#6](=[#8])(-[#7]-1-[#6]-[#6]-[#16]-[#6]-[#6]-1)-c:2:c(:c(:c(:c(:c:2-[#16]-[#6](-[#1])-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('naphth_ene_one_A(1)', 'c:1:c:c:3:c:2:c(:c:1)-[#6](-[#6]=[#6](-c:2:c:c:c:3)-[#8]-[#6](-[#1])-[#1])=[#8]'),
    ('naphth_ene_one_B(1)', 'c:1-3:c:2:c(:c(:c:c:1)-[#7]):c:c:c:c:2-[#6](-[#6]=[#6]-3-[#6](-[F])(-[F])-[F])=[#8]'),
    ('amino_acridine_A(1)', 'c:1:c:c:c:c:2:c:1:c:c:3:c(:n:2):n:c:4:c(:c:3-[#7]):c:c:c:c:4'),
    ('keto_phenone_B(1)', 'c:1:c-3:c(:c:c:c:1)-[#6]-2=[#7]-[!#1]=[#6]-[#6]-[#6]-2-[#6]-3=[#8]'),
    ('hzone_acid_A(1)', 'c:1-3:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#7]-[#7](-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1])-c:4:c-3:c(:c(:c(:c:4-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('sulfonamide_H(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#16](=[#8])(=[#8])-[#7](-[#1])-c:2:n:n:c(:c(:c:2-[#1])-[#1])-[#1]'),
    ('het_565_indole(1)', 'c2(c(-[#1])n(-[#6](-[#1])-[#1])c:3:c(:c(:c:1n(c(c(c:1:c2:3)-[#1])-[#1])-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1]'),
    ('pyrrole_J(1)', 'c1(c-2c(c(n1-[#6](-[#8])=[#8])-[#6](-[#1])-[#1])-[#16]-[#6](-[#1])(-[#1])-[#16]-2)-[#6](-[#1])-[#1]'),
    ('pyrazole_amino_B(1)', 's1ccnc1-c2c(n(nc2-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('pyrrole_K(1)', 'c1(c(c(c(n1-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('anthranil_acid_I(1)', 'c:1(:c(:c(:c(:o:1)-[#6])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:2:[#6](-[#1]):[#6](-[#1]):[#6](-[#1]):[#6](-[#1]):[#6]:2-[#6](=[#8])-[#8]-[#1]'),
    ('thio_amide_F(1)', '[!#1]:[#6]-[#6](=[#16])-[#7](-[#1])-[#7](-[#1])-[#6]:[!#1]'),
    ('ene_one_C(1)', '[#6]-1(=[#8])-[#6](-[#6](-[#6]#[#7])=[#6](-[#1])-[#7])-[#6](-[#7])-[#6]=[#6]-1'),
    ('het_65_H(1)', 'c2(c-,:1n(-,:[#6](-,:[#6]=,:[#6]-,:[#7]-,:1)=[#8])nc2-c3cccn3)-[#6]#[#7]'),
    ('cyano_imine_D(1)', '[#8]=[#6]-1-[#6](=[#7]-[#7]-[#6]-[#6]-1)-[#6]#[#7]'),
    ('cyano_misc_A(1)', 'c:2(:c:1:c:c:c:c:c:1:n:n:c:2)-[#6](-[#6]:[#6])-[#6]#[#7]'),
    ('ene_misc_C(1)', 'c:1:c:c-2:c(:c:c:1)-[#6]=[#6]-[#6](-[#7]-2-[#6](=[#8])-[#7](-[#1])-c:3:c:c(:c(:c:c:3)-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('het_66_E(1)', 'c:2:c:c:1:n:c(:c(:n:c:1:c:c:2)-c:3:c:c:c:c:c:3)-c:4:c:c:c:c:c:4-[#8]-[#1]'),
    ('keto_keto_beta_F(1)', '[#6](-[#1])(-[#1])-[#6](-[#8]-[#1])=[#6](-[#6](=[#8])-[#6](-[#1])-[#1])-[#6](-[#1])-[#6]#[#6]'),
    ('misc_naphthimidazole(1)', 'c:1:c:4:c(:c:c2:c:1nc(n2-[#1])-[#6]-[#8]-[#6](=[#8])-c:3:c:c(:c:c(:c:3)-[#7](-[#1])-[#1])-[#7](-[#1])-[#1]):c:c:c:c:4'),
    ('naphth_ene_one_C(1)', 'c:2(:c:1:c:c:c:c-3:c:1:c(:c:c:2)-[#6]=[#6]-[#6]-3=[#7])-[#7]'),
    ('keto_phenone_C(1)', 'c:2(:c:1:c:c:c:c:c:1:c-3:c(:c:2)-[#6](-c:4:c:c:c:c:c-3:4)=[#8])-[#8]-[#1]'),
    ('coumarin_C(1)', '[#6]-,:2(-,:[#6]=,:[#7]-,:c:1:c:c(:c:c:c:1-,:[#8]-,:2)-[Cl])=[#8]'),
    ('thio_est_cyano_A(1)', '[#6]-1=[#6]-[#7](-[#6](-c:2:c-1:c:c:c:c:2)(-[#6]#[#7])-[#6](=[#16])-[#16])-[#6]=[#8]'),
    ('het_65_imidazole(1)', 'c2(nc:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])n2-[#6])-[#7](-[#1])-[#6](-[#7](-[#1])-c:3:c(:c:c:c:c:3-[#1])-[#1])=[#8]'),
    ('anthranil_acid_J(1)', '[#7](-[#1])(-[#6]:[#6])-c:1:c(-[#6](=[#8])-[#8]-[#1]):c:c:c(:n:1)-,:[#6]:[#6]'),
    ('colchicine_het(1)', 'c:1-,:3:c(:c:c:c:c:1)-,:[#16]-,:[#6](=[#7]-[#7]=[#6]-,:2-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:2)-,:[#7]-,:3-[#6](-[#1])-[#1]'),
    ('ene_misc_D(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#6](-[#6])-[#16]-[#6]-2(-[#1])-[#1])-[#6]'),
    ('indole_3yl_alk_B(1)', 'c:12:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])c(c(-[#6]:[#6])n2-!@[#6]:[#6])-[#6](-[#1])-[#1]'),
    ('anil_OH_no_alk_A(1)', '[#7](-[#1])(-[#1])-c:1:c:c:c(:c:c:1-[#8]-[#1])-[#16](=[#8])(=[#8])-[#8]-[#1]'),
    ('thiazole_amine_L(1)', 's:1:c:c:c(:c:1-[#1])-c:2:c:s:c(:n:2)-[#7](-[#1])-[#1]'),
    ('pyrazole_amino_A(1)', 'c1c(-[#7](-[#1])-[#1])nnc1-c2c(-[#6](-[#1])-[#1])oc(c2-[#1])-[#1]'),
    ('het_thio_N_5D(1)', 'n1nscc1-c2nc(no2)-[#6]:[#6]'),
    ('anil_alk_indane(1)', 'c:1(:c:c-3:c(:c:c:1)-[#7]-[#6]-4-c:2:c:c:c:c:c:2-[#6]-[#6]-3-4)-[#6;X4]'),
    ('anil_di_alk_N(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#6](-[#1])-[#6]-3-[#6](-[#6]#[#7])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#7]-2-3)-[#1]'),
    ('het_666_C(1)', 'c:2-,:3:c(:c:c:1:c:c:c:c:c:1:c:2)-,:[#7](-[#6](-[#1])-[#1])-,:[#6](=[#8])-,:[#6](=,:[#7]-,:3)-[#6]:[#6]-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('ene_one_D(1)', '[#6](-[#8]-[#1]):[#6]-[#6](=[#8])-[#6](-[#1])=[#6](-[#6])-[#6]'),
    ('anil_di_alk_indol(1)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]):c(:c(-[#1]):n:2-[#1])-[#16](=[#8])=[#8]'),
    ('anil_no_alk_indol_A(1)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1]):c(:c(-[#1]):n:2-[#6](-[#1])-[#1])-[#1]'),
    ('dhp_amino_CN_G(1)', '[#16;X2]-1-[#6]=[#6](-[#6]#[#7])-[#6](-[#6])(-[#6]=[#8])-[#6](=[#6]-1-[#7](-[#1])-[#1])-[$([#6]=[#8]),$([#6]#[#7])]'),
    ('anil_di_alk_dhp(1)', '[#7]-2-[#6]=[#6](-[#6]=[#8])-[#6](-c:1:c:c:c(:c:c:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#6]~3=,:[#6]-2~[#7]~[#6](~[#16])~[#7]~[#6]~3~[#7]'),
    ('anthranil_amide_A(1)', 'c:1:c(:c:c:c:c:1)-[#6](=[#8])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#6](=[#8])-[#7](-[#1])-[#7](-[#1])-c:3:n:c:c:s:3'),
    ('hzone_anthran_Z(1)', 'c:1:c:2:c(:c:c:c:1):c(:c:3:c(:c:2):c:c:c:c:3)-[#6]=[#7]-[#7](-[#1])-c:4:c:c:c:c:c:4'),
    ('ene_one_amide_A(1)', 'c:1:c(:c:c:c:c:1)-[#6](-[#1])-[#7]-[#6](=[#8])-[#6](-[#7](-[#1])-[#6](-[#1])-[#1])=[#6](-[#1])-[#6](=[#8])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])-[#1]'),
    ('het_76_A(1)', 's:1:c(:c(-[#1]):c(:c:1-[#6]-3=[#7]-c:2:c:c:c:c:c:2-[#6](=[#7]-[#7]-3-[#1])-c:4:c:c:n:c:c:4)-[#1])-[#1]'),
    ('thio_urea_N(1)', 'o:1:c(:c(-[#1]):c(:c:1-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7](-[#6]-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2)-[#1])-[#1]'),
    ('anil_di_alk_coum(1)', 'c:1:c(:c:c:c:c:1)-[#7](-[#6]-[#1])-[#6](-[#1])-[#6](-[#1])-[#6](-[#1])-[#7](-[#1])-[#6](=[#8])-[#6]-,:2=,:[#6](-,:[#8]-,:[#6](-,:[#6](=,:[#6]-,:2-[#6](-[#1])-[#1])-[#1])=[#8])-[#6](-[#1])-[#1]'),
    ('ene_one_amide_B(1)', 'c2-3:c:c:c:1:c:c:c:c:c:1:c2-[#6](-[#1])-[#6;X4]-[#7]-[#6]-3=[#6](-[#1])-[#6](=[#8])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('het_thio_656c(1)', 'c:1:c(:c:c:c:c:1)-[#6]-4=[#7]-[#7]:2:[#6](:[#7+]:c:3:c:2:c:c:c:c:3)-[#16]-[#6;X4]-4'),
    ('het_5_ene(1)', '[#6]-2(=[#8])-[#6](=[#6](-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#7]=[#6](-c:1:c:c:c:c:c:1)-[#8]-2'),
    ('thio_imide_A(1)', 'c:1:c(:c:c:c:c:1)-[#7]-2-[#6](=[#8])-[#6](=[#6](-[#1])-[#6]-2=[#8])-[#16]-c:3:c:c:c:c:c:3'),
    ('dhp_amidine_A(1)', '[#7]-,:1(-[#1])-,:[#7]=,:[#6](-[#7]-[#1])-,:[#16]-,:[#6](=,:[#6]-,:1-,:[#6]:[#6])-,:[#6]:[#6]'),
    ('thio_urea_O(1)', 'c:1(:c(:c-3:c(:c(:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])-c:2:c(:c(:c(:o:2)-[#6]-[#1])-[#1])-[#1])-[#1])-[#8]-[#6](-[#8]-3)(-[#1])-[#1])-[#1])-[#1]'),
    ('anil_di_alk_O(1)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-c:2:c:c:c:c:c:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('thio_urea_P(1)', '[#8]=[#6]-!@n:1:c:c:c-,:2:c:1-,:[#7](-[#1])-,:[#6](=[#16])-,:[#7]-,:2-[#1]'),
    ('het_pyraz_misc(1)', '[#6](-[F])(-[F])-[#6](=[#8])-[#7](-[#1])-c:1:c(-[#1]):n(-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6]:[#6]):n:c:1-[#1]'),
    ('diazox_C(1)', '[#7]-2=[#7]-[#6]:1:[#7]:[!#6&!#1]:[#7]:[#6]:1-[#7]=[#7]-[#6]:[#6]-2'),
    ('diazox_D(1)', '[#6]-2(-[#1])(-[#8]-[#1])-[#6]:1:[#7]:[!#6&!#1]:[#7]:[#6]:1-[#6](-[#1])(-[#8]-[#1])-[#6]=[#6]-2'),
    ('misc_cyclopropane(1)', '[#6]-1(-[#6](-[#1])(-[#1])-[#6]-1(-[#1])-[#1])(-[#6](=[#8])-[#7](-[#1])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])(-[#1])-[#8])-[#16](=[#8])(=[#8])-[#6]:[#6]'),
    ('imine_ene_one_B(1)', '[#6]-1:[#6]-[#6](=[#8])-[#6]=[#6]-1-[#7]=[#6](-[#1])-[#7](-[#6;X4])-[#6;X4]'),
    ('coumarin_D(1)', 'c:1:c:c(:c:c-,:2:c:1-,:[#6](=,:[#6](-[#1])-,:[#6](=[#8])-,:[#8]-,:2)-c:3:c:c:c:c:c:3)-[#8]-[#6](-[#1])(-[#1])-[#6]:[#8]:[#6]'),
    ('misc_furan_A(1)', 'c:1:c(:o:c(:c:1-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#8]-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-c:2:c:c-3:c(:c:c:2)-[#8]-[#6](-[#8]-3)(-[#1])-[#1]'),
    ('rhod_sat_E(1)', '[#7]-4(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#7](-[#1])-c:2:c:c:c:c:3:c:c:c:c:c:2:3)-[#6]-4=[#8]'),
    ('rhod_sat_imine_A(1)', '[#7]-3(-[#6](=[#8])-c:1:c:c:c:c:c:1)-[#6](=[#7]-c:2:c:c:c:c:c:2)-[#16]-[#6](-[#1])(-[#1])-[#6]-3=[#8]'),
    ('rhod_sat_F(1)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#1])-[#6]-2=[#16]'),
    ('het_thio_5_imine_B(1)', '[#7]-1(-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#6]:[#6])-[#6](=[#7]-[#6]:[#6])-[#6]-1=[#7]-[#6]:[#6]'),
    ('het_thio_5_imine_C(1)', '[#16]-1-[#6](=[#7]-[#7]-[#1])-[#16]-[#6](=[#7]-[#6]:[#6])-[#6]-1=[#7]-[#6]:[#6]'),
    ('ene_five_het_N(1)', '[#6]-2(=[#8])-[#6](=[#6](-[#1])-c:1:c(:c:c:c(:c:1)-[F,Cl,Br,I])-[#8]-[#6](-[#1])-[#1])-[#7]=[#6](-[#16]-[#6](-[#1])-[#1])-[#16]-2'),
    ('thio_carbam_A(1)', '[#6](-[#1])(-[#1])-[#16]-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('misc_anilide_A(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6])-[#1])-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('misc_anilide_B(1)', 'c:1(:c(:c(:c(:c(:c:1-[#6](-[#1])-[#1])-[#1])-[Br])-[#1])-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('mannich_B(1)', 'c:1-2:c(:c:c:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#7](-[#6]:[#6]-[#8]-[#6](-[#1])-[#1])-[#6]-2(-[#1])-[#1])-[#1])-[#1]'),
    ('mannich_catechol_A(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2(-[#1])-[#1])-[#1])-[#8])-[#8])-[#1]'),
    ('anil_alk_D(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('het_65_I(1)', 'n:1:2:c:c:c(:c:c:1:c:c(:c:2-[#6](=[#8])-[#6]:[#6])-[#6]:[#6])-[#6](~[#8])~[#8]'),
    ('misc_urea_A(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#6](=[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](-[#6;X4])(-[#6;X4])-[#7](-[#1])-[#6](=[#8])-[#7](-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('imidazole_C(1)', '[#6]-3(-[#1])(-n:1:c(:n:c(:c:1-[#1])-[#1])-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[Br])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c:4:c-3:c(:c(:c(:c:4-[#1])-[#1])-[#1])-[#1]'),
    ('styrene_imidazole_A(1)', '[#6](=[#6](-[#1])-[#6](-[#1])(-[#1])-n:1:c(:n:c(:c:1-[#1])-[#1])-[#1])(-[#6]:[#6])-[#6]:[#6]'),
    ('thiazole_amine_M(1)', 'c:1(:n:c(:c(-[#1]):s:1)-c:2:c:c:n:c:c:2)-[#7](-[#1])-[#6]:[#6]-[#6](-[#1])-[#1]'),
    ('misc_pyrrole_thiaz(1)', 'c:1(:n:c(:c(-[#1]):s:1)-c:2:c:c:c:c:c:2)-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](-[#1])(-[#1])-c:3:c:c:c:n:3-[#1]'),
    ('pyrrole_L(1)', 'n:1(-[#1]):c(:c(-[#6](-[#1])-[#1]):c(:c:1-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](=[#8])-[#8]-[#6](-[#1])-[#1]'),
    ('het_thio_65_D(1)', 'c:2(:n:c:1:c(:c(:c:c(:c:1-[#1])-[F,Cl,Br,I])-[#1]):n:2-[#1])-[#16]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('ene_misc_E(1)', 'c:1(:c(:c-2:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1])-[#6]=[#6]-[#6](-[#1])-[#16]-2)-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('thio_cyano_A(1)', '[#7]-1(-[#1])-[#6](=[#16])-[#6](-[#1])(-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6]-1-[#6]:[#6])-[#1]'),
    ('cyano_amino_het_B(1)', 'n:1:c(:c(:c(:c(:c:1-[#16;X2]-c:2:c:c:c:c:c:2-[#7](-[#1])-[#1])-[#6]#[#7])-c:3:c:c:c:c:c:3)-[#6]#[#7])-[#7](-[#1])-[#1]'),
    ('cyano_pyridone_G(1)', '[#7]-,:2(-c:1:c:c:c(:c:c:1)-[#8]-[#6](-[#1])-[#1])-,:[#6](=[#8])-,:[#6](=,:[#6]-,:[#6](=,:[#7]-,:2)-n:3:c:n:c:c:3)-[#6]#[#7]'),
    ('het_65_J(1)', 'o:1:c(:c:c:2:c:1:c(:c(:c(:c:2-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](~[#8])~[#8]'),
    ('ene_one_yne_A(1)', '[#6]#[#6]-[#6](=[#8])-[#6]#[#6]'),
    ('anil_OH_no_alk_B(1)', 'c:2(:c:1:c(:c(:c(:c(:c:1:c(:c(:c:2-[#8]-[#1])-[#6]=[#8])-[#1])-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('hzone_acyl_misc_A(1)', 'c:1(:c(:c(:[c;!H0,$(c-[#6;!H0;!H1])](:o:1))-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6;!H0,$([#6]-[#6;!H0!H1])]-c:2:c:c:c:c(:c:2)-[*]-[*]-[*]-c:3:c:c:c:o:3'),
    ('thiophene_F(1)', '[#16](=[#8])(=[#8])-[#7](-[#1])-c:1:c(:c(:c(:s:1)-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#7]-[#1]'),
    ('anil_OC_alk_E(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#8]-[#1])-[#6](-[#1])-[#1]'),
    ('anil_OC_alk_F(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#6]=[#8])-[#16]'),
    ('het_65_K(1)', 'n1nnnc2cccc12'),
    ('het_65_L(1)', 'c:1-,:2:c(-[#1]):s:c(:c:1-,:[#6](=[#8])-,:[#7]-,:[#7]=,:[#6]-,:2-[#7](-[#1])-[#1])-[#6]=[#8]'),
    ('coumarin_E(1)', 'c:1-,:3:c(:c:2:c(:c:c:1-[Br]):o:c:c:2)-,:[#6](=,:[#6]-,:[#6](=[#8])-,:[#8]-,:3)-[#1]'),
    ('coumarin_F(1)', 'c:1-,:3:c(:c:c:c:c:1)-,:[#6](=,:[#6](-[#6](=[#8])-[#7](-[#1])-c:2:n:o:c:c:2-[Br])-,:[#6](=[#8])-,:[#8]-,:3)-[#1]'),
    ('coumarin_G(1)', 'c:1-,:2:c(:c:c(:c:c:1-[F,Cl,Br,I])-[F,Cl,Br,I])-,:[#6](=,:[#6](-[#6](=[#8])-[#7](-[#1])-[#1])-,:[#6](=[#7]-[#1])-,:[#8]-,:2)-[#1]'),
    ('coumarin_H(1)', 'c:1-,:3:c(:c:c:c:c:1)-,:[#6](=,:[#6](-[#6](=[#8])-[#7](-[#1])-c:2:n:c(:c:s:2)-[#6]:[#16]:[#6]-[#1])-,:[#6](=[#8])-,:[#8]-,:3)-[#1]'),
    ('het_thio_67_A(1)', '[#6](-[#1])(-[#1])-[#16;X2]-c:2:n:n:c:1-[#6]:[#6]-[#7]=[#6]-[#8]-c:1:n:2'),
    ('sulfonamide_I(1)', '[#16](=[#8])(=[#8])(-c:1:c:n(-[#6](-[#1])-[#1]):c:n:1)-[#7](-[#1])-c:2:c:n(:n:c:2)-[#6](-[#1])(-[#1])-[#6]:[#6]-[#8]-[#6](-[#1])-[#1]'),
    ('het_65_mannich(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#8]-2)-[#6](-[#1])(-[#1])-[#7]-3-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]-3)-[#1])-[#1])-[#1]'),
    ('anil_alk_A(1)', '[#6](-[#1])(-[#1])-[#8]-[#6]:[#6]-[#6](-[#1])(-[#1])-[#7](-[#1])-c:2:c(:c(:c:1:n(:c(:n:c:1:c:2-[#1])-[#1])-[#6]-[#1])-[#1])-[#1]'),
    ('het_5_inium(1)', '[#7]-,:4(-c:1:c:c:c:c:c:1)-,:[#6](=,:[#7+](-c:2:c:c:c:c:c:2)-,:[#6](=[#7]-c:3:c:c:c:c:c:3)-,:[#7]-,:4)-[#1]'),
    ('anil_di_alk_P(1)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:1:s:c(:n:c:1:c:2)-[#16]-[#6](-[#1])-[#1]'),
    ('thio_urea_Q(1)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(-[#1]):c(:c:2-[#1])-[#1])-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6]:[#6]:[#6])-[#1])-[#1])-[#1])-[#1]'),
    ('thio_pyridine_A(1)', '[#6]:1(:[#7]:[#6](:[#7]:[!#1]:[#7]:1)-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-[#16]-[#6;X4]'),
    ('melamine_B(1)', 'n:1:c(:n:c(:n:c:1-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#6]-[#1])-[#6]=[#8]'),
    ('misc_phthal_thio_N(1)', 'c:1(:n:s:c(:n:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](=[#8])-c:2:c:c:c:c:c:2-[#6](=[#8])-[#8]-[#1])-c:3:c:c:c:c:c:3'),
    ('hzone_acyl_misc_B(1)', 'n:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6](-[#1])-c:2:c:c:c:c:c:2-[#8]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('tert_butyl_B(1)', '[#6](-[#1])(-[#1])(-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#8]-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c(:c(:c:2-[#1])-[#1])-[#8]-[#1])-[#1]'),
    ('diazox_E(1)', '[#7](-[#1])(-[#1])-c:1:c(-[#7](-[#1])-[#1]):c(:c(-[#1]):c:2:n:o:n:c:1:2)-[#1]'),
    ('anil_NH_no_alk_B(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])=[#8])-[#1])-[#7](-[#1])-[#6](-[#1])-[#1])-[F,Cl,Br,I])-[#1]'),
    ('anil_no_alk_A(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#7]=[#6]-2-[#6](=[#6]~[#6]~[#6]=[#6]-2)-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('anil_no_alk_B(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-n:2:c:c:c:c:2)-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]'),
    ('thio_ene_amine_A(1)', '[#16]=[#6]-[#6](-[#6](-[#1])-[#1])=[#6](-[#6](-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('het_55_B(1)', '[#6]-1:[#6]-[#8]-[#6]-2-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]-[#6]-1-2'),
    ('cyanamide_A(1)', '[#8]-[#6](=[#8])-[#6](-[#1])(-[#1])-[#16;X2]-[#6](=[#7]-[#6]#[#7])-[#7](-[#1])-c:1:c:c:c:c:c:1'),
    ('ene_one_one_A(1)', '[#8]=[#6]-[#6]-1=[#6](-[#16]-[#6](=[#6](-[#1])-[#6])-[#16]-1)-[#6]=[#8]'),
    ('ene_six_het_D(1)', '[#8]=[#6]-1-[#7]-[#7]-[#6](=[#7]-[#6]-1=[#6]-[#1])-[!#1]:[!#1]'),
    ('ene_cyano_E(1)', '[#8]=[#6]-[#6](-[#1])=[#6](-[#6]#[#7])-[#6]'),
    ('ene_cyano_F(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c(:c(:c(:c(:c:1-[#8]-[#1])-[#1])-c:2:c(-[#1]):c(:c(:o:2)-[#6](-[#1])=[#6](-[#6]#[#7])-c:3:n:c:c:n:3)-[#1])-[#1])-[#1]'),
    ('hzone_furan_C(1)', 'c:1:c(:c:c:c:c:1)-[#7](-c:2:c:c:c:c:c:2)-[#7]=[#6](-[#1])-[#6]:3:[#6](:[#6](:[#6](:[!#1]:3)-c:4:c:c:c:c(:c:4)-[#6](=[#8])-[#8]-[#1])-[#1])-[#1]'),
    ('anil_no_alk_C(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-c:2:c(-[#1]):c(:c(-[#6](-[#1])-[#1]):o:2)-[#6]=[#8])-[#1])-[#1]'),
    ('hzone_acid_D(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c:c(:c:c:1)-[#7]-[#7]=[#6](-[#1])-[#6]:2:[#6](:[#6](:[#6](:[!#1]:2)-c:3:c:c:c:c:c:3)-[#1])-[#1]'),
    ('hzone_furan_E(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c:c:c(:c:1)-[#6]:[!#1]:[#6]-[#6]=[#7]-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#8]'),
    ('het_6_pyridone_NH2(1)', '[#8](-[#1])-[#6]:1:[#6](:[#6]:[!#1]:[#6](:[#7]:1)-[#7](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]'),
    ('imine_one_fives_D(1)', '[#6]-1(=[!#6&!#1])-[#6](-[#7]=[#6]-[#16]-1)=[#8]'),
    ('pyrrole_M(1)', 'n2(-c:1:c:c:c:c:c:1)c(c(-[#1])c(c2-[#6]=[#7]-[#8]-[#1])-[#1])-[#1]'),
    ('pyrrole_N(1)', 'n2(-[#6](-[#1])-c:1:c(:c(:c:c(:c:1-[#1])-[#1])-[#1])-[#1])c(c(-[#1])c(c2-[#6]-[#1])-[#1])-[#6]-[#1]'),
    ('pyrrole_O(1)', 'n1(-[#6](-[#1])-[#1])c(c(-[#6](=[#8])-[#6])c(c1-[#6]:[#6])-[#6])-[#6](-[#1])-[#1]'),
    ('ene_cyano_G(1)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])=[#6](-[#6]#[#7])-c:2:n:c:c:s:2)-[#1])-[#1]'),
    ('sulfonamide_J(1)', 'n3(-c:1:c:c:c:c:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-c:2:c:c:c:s:2)c(c(-[#1])c(c3-[#1])-[#1])-[#1]'),
    ('misc_pyrrole_benz(1)', 'n2(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6]:[#6])c(c(-[#1])c(c2-[#1])-[#1])-[#1]'),
    ('thio_urea_R(1)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6](-[#1])=[#6](-[#1])-[#6]=[#8]'),
    ('ene_one_one_B(1)', '[#6]-1(-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6]-[#6](-[#1])(-[#1])-[#6]-1=[#8])=[#6](-[#7]-[#1])-[#6]=[#8]'),
    ('dhp_amino_CN_H(1)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#16]-[#6;X4]-[#16]-1'),
    ('het_66_anisole(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-c:2:c:c:n:c:3:c(:c:c:c(:c:2:3)-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('thiazole_amine_N(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#7](-[#1])-c:2:n:c(:c:s:2)-c:3:c:c:c(:c:c:3)-[#8]-[#6](-[#1])-[#1]'),
    ('het_pyridiniums_C(1)', '[#6]~1~3~[#7](-[#6]:[#6])~[#6]~[#6]~[#6]~[#6]~1~[#6]~2~[#7]~[#6]~[#6]~[#6]~[#7+]~2~[#7]~3'),
    ('het_5_E(1)', '[#7]-3(-c:2:c:1:c:c:c:c:c:1:c:c:c:2)-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6]-3=[#8]'),
    ('thiaz_ene_A(128)', '[#6]-,:1(=,:[#6;!H0,$([#6]-[#6;!H0;!H1]),$([#6]-[#6]=[#8])]-,:[#16]-,:[#6](-,:[#7;!H0,$([#7]-[#6;!H0]),$([#7]-[#6]:[#6])]-,:1)=[#7;!R])-[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('pyrrole_A(118)', 'n2(-[#6]:1:[!#1]:[#6]:[#6]:[#6]:[#6]:1)c(cc(c2-[#6;X4])-[#1])-[#6;X4]'),
    ('catechol_A(92)', 'c:1:c:c(:c(:c:c:1)-[#8]-[#1])-[#8]-[#1]'),
    ('ene_five_het_B(90)', '[#6]-1(=[#6])-[#6](-[#7]=[#6]-[#16]-1)=[#8]'),
    ('imine_one_fives(89)', '[#6]-1=[!#1]-[!#6&!#1]-[#6](-[#6]-1=[!#6&!#1;!R])=[#8]'),
    ('ene_five_het_C(85)', '[#6]-1(-[#6](-[#6]=[#6]-[!#6&!#1]-1)=[#6])=[!#6&!#1]'),
    ('hzone_pipzn(79)', '[#6]-[#7]-1-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])-[#6]-1(-[#1])-[#1])-[#7]=[#6](-[#1])-[#6]:[!#1]'),
    ('keto_keto_beta_A(68)', 'c:1-2:c(:c:c:c:c:1)-[#6](=[#8])-[#6;X4]-[#6]-2=[#8]'),
    ('hzone_pyrrol(64)', 'n1(-[#6])c(c(-[#1])c(c1-[#6]=[#7]-[#7])-[#1])-[#1]'),
    ('ene_one_ene_A(57)', '[#6]=!@[#6](-[!#1])-@[#6](=!@[!#6&!#1])-@[#6](=!@[#6])-[!#1]'),
    ('cyano_ene_amine_A(56)', '[#6](-[#6]#[#7])(-[#6]#[#7])-[#6](-[#7](-[#1])-[#1])=[#6]-[#6]#[#7]'),
    ('ene_five_one_A(55)', 'c:1-2:c(:c:c:c:c:1)-[#6](=[#8])-[#6](=[#6])-[#6]-2=[#8]'),
    ('cyano_pyridone_A(54)', '[#6]-,:1(=,:[!#1]-,:[!#1]=,:[!#1]-,:[#7](-,:[#6]-,:1=[#16])-[#1])-[#6]#[#7]'),
    ('anil_alk_ene(51)', 'c:1:c:c-2:c(:c:c:1)-[#6]-3-[#6](-[#6]-[#7]-2)-[#6]-[#6]=[#6]-3'),
    ('amino_acridine_A(46)', 'c:1:c:2:c(:c:c:c:1):n:c:3:c(:c:2-[#7]):c:c:c:c:3'),
    ('ene_five_het_D(46)', '[#6]-1(=[#6])-[#6](=[#8])-[#7]-[#7]-[#6]-1=[#8]'),
    ('thiophene_amino_Aa(45)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:s:1)-[!#1])-[!#1])-[#6]=[#8]'),
    ('ene_five_het_E(44)', '[#7]-[#6]=!@[#6]-2-[#6](=[#8])-c:1:c:c:c:c:c:1-[!#6&!#1]-2'),
    ('sulfonamide_A(43)', 'c:1(:c(:c(:c(:c(:c:1-[#8]-[#1])-[F,Cl,Br,I])-[#1])-[F,Cl,Br,I])-[#1])-[#16](=[#8])(=[#8])-[#7]'),
    ('thio_ketone(43)', '[#6]-[#6](=[#16])-[#6]'),
    ('sulfonamide_B(41)', 'c:1:c:c(:c:c:c:1-[#8]-[#1])-[#7](-[#1])-[#16](=[#8])=[#8]'),
    ('anil_no_alk(40)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[$([#8]),$([#7]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('thiophene_amino_Ab(40)', '[c;!H0,$(c-[#6](-[#1])-[#1]),$(c-[#6]:[#6])]:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6])-[#6](=[#8])-[#8])-[$([#6]:1:[#6]:[#6]:[#6]:[#6]:[#6]:1),$([#6]:1:[#16]:[#6]:[#6]:[#6]:1)]'),
    ('het_pyridiniums_A(39)', '[#7+]:1(:[#6]:[#6]:[!#1]:c:2:c:1:c(:[c;!H0,$(c-[#7])]:c:c:2)-[#1])-[$([#6](-[#1])(-[#1])-[#1]),$([#8;X1]),$([#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1]),$([#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#8]-[#1]),$([#6](-[#1])(-[#1])-[#6](=[#8])-[#6]),$([#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6]),$([#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#1])]'),
    ('anthranil_one_A(38)', 'c:1:c:c:c:c(:c:1-[#7&!H0;!H1,!$([#7]-[#6]=[#8])])-[#6](-[#6]:[#6])=[#8]'),
    ('cyano_imine_A(37)', '[#7](-[#1])-[#7]=[#6](-[#6]#[#7])-[#6]=[!#6&!#1;!R]'),
    ('diazox_sulfon_A(36)', '[#7](-c:1:c:c:c:c:c:1)-[#16](=[#8])(=[#8])-[#6]:2:[#6]:[#6]:[#6]:[#6]:3:[#7]:[$([#8]),$([#16])]:[#7]:[#6]:2:3'),
    ('hzone_anil_di_alk(35)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])=[#7]-[#7]-[$([#6](=[#8])-[#6](-[#1])(-[#1])-[#16]-[#6]:[#7]),$([#6](=[#8])-[#6](-[#1])(-[#1])-[!#1]:[!#1]:[#7]),$([#6](=[#8])-[#6]:[#6]-[#8]-[#1]),$([#6]:[#7]),$([#6](-[#1])(-[#1])-[#6](-[#1])-[#8]-[#1])])-[#1])-[#1]'),
    ('rhod_sat_A(33)', '[#7]-1-[#6](=[#16])-[#16]-[#6;X4]-[#6]-1=[#8]'),
    ('hzone_enamin(30)', '[#7](-[#1])-[#7]=[#6]-[#6;!H0,$(*(-[#6])-[#6])]=[#6](-[#6])-!@[$([#7]),$([#8]-[#1])]'),
    ('pyrrole_B(29)', 'n2(-[#6]:1:[!#1]:[#6]:[#6]:[#6]:[#6]:1)c(cc(c2-[#6]:[#6])-[#1])-[#6;X4]'),
    ('thiophene_hydroxy(28)', 's1ccc(c1)-[#8]-[#1]'),
    ('cyano_pyridone_B(27)', '[#6]-,:1(=,:[#6](-,:[#6](=[#8])-,:[#7]-,:[#6](=,:[#7]-,:1)-,:[!#6&!#1])-[#6]#[#7])-[#6]'),
    ('imine_one_sixes(27)', '[#6]-1(-[#6](=[#8])-[#7]-[#6](=[#8])-[#7]-[#6]-1=[#8])=[#7]'),
    ('dyes5A(27)', '[#6](-[#1])(-[#1])-[#7]([#6]:[#6])~[#6][#6]=,:[#6]-[#6]~[#6][#7]'),
    ('naphth_amino_A(25)', 'c:2:c:1:c:c:c:c-,:3:c:1:c(:c:c:2)-,:[#7]-,:[#6]=,:[#7]-,:3'),
    ('naphth_amino_B(25)', 'c:2:c:1:c:c:c:c-3:c:1:c(:c:c:2)-[#7](-[#6;X4]-[#7]-3-[#1])-[#1]'),
    ('ene_one_ester(24)', '[#6]-[#6](=[#8])-[#6](-[#1])=[#6](-[#7](-[#1])-[#6])-[#6](=[#8])-[#8]-[#6]'),
    ('thio_dibenzo(23)', '[#16]=[#6]-1-[#6]=,:[#6]-[!#6&!#1]-[#6]=,:[#6]-1'),
    ('cyano_cyano_A(23)', '[#6](-[#6]#[#7])(-[#6]#[#7])-[#6](-[$([#6]#[#7]),$([#6]=[#7])])-[#6]#[#7]'),
    ('hzone_acyl_naphthol(22)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(:c(:c:2-[#1])-[#8]-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6])-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('het_65_A(21)', '[#8]=[#6]-c2c1nc(-[#6](-[#1])-[#1])cc(-[#8]-[#1])n1nc2'),
    ('imidazole_A(19)', 'n:1:c(:n(:c(:c:1-c:2:c:c:c:c:c:2)-c:3:c:c:c:c:c:3)-[#1])-[#6]:[!#1]'),
    ('ene_cyano_A(19)', '[#6](-[#6]#[#7])(-[#6]#[#7])=[#6]-c:1:c:c:c:c:c:1'),
    ('anthranil_acid_A(19)', 'c:1(:c:c:c:c:c:1-[#7](-[#1])-[#7]=[#6])-[#6](=[#8])-[#8]-[#1]'),
    ('dyes3A(19)', '[#7+]([#6]:[#6])=,:[#6]-[#6](-[#1])=[#6]-[#7](-[#6;X4])-[#6]'),
    ('dhp_bis_amino_CN(19)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#7](-[#1])-[#1])-[#16]-1)-[#6]#[#7]'),
    ('het_6_tetrazine(18)', '[#7]~[#6]:1:[#7]:[#7]:[#6](:[$([#7]),$([#6]-[#1]),$([#6]-[#7]-[#1])]:[$([#7]),$([#6]-[#7])]:1)-[$([#7]-[#1]),$([#8]-[#6](-[#1])-[#1])]'),
    ('ene_one_hal(17)', '[#6]-[#6]=[#6](-[F,Cl,Br,I])-[#6](=[#8])-[#6]'),
    ('cyano_imine_B(17)', '[#6](-[#6]#[#7])(-[#6]#[#7])=[#7]-[#7](-[#1])-c:1:c:c:c:c:c:1'),
    ('thiaz_ene_B(17)', '[#6]-,:1(=,:[#6](-!@[#6](=[#8])-[#7]-[#6](-[#1])-[#1])-,:[#16]-,:[#6](-,:[#7]-,:1-,:[$([#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1]),$([#6]:[#6])])=[#16])-,:[$([#7]-[#6](=[#8])-[#6]:[#6]),$([#7](-[#1])-[#1])]'),
    ('ene_rhod_B(16)', '[#16]-1-[#6](=[#8])-[#7]-[#6](=[#8])-[#6]-1=[#6](-[#1])-[$([#6]-[#35]),$([#6]:[#6](-[#1]):[#6](-[F,Cl,Br,I]):[#6]:[#6]-[F,Cl,Br,I]),$([#6]:[#6](-[#1]):[#6](-[#1]):[#6]-[#16]-[#6](-[#1])-[#1]),$([#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]-[#8]-[#6](-[#1])-[#1]),$([#6]:1:[#6](-[#6](-[#1])-[#1]):[#7](-[#6](-[#1])-[#1]):[#6](-[#6](-[#1])-[#1]):[#6]:1)]'),
    ('thio_carbonate_A(15)', '[#8]-,:1-,:[#6](-,:[#16]-,:c:2:c-,:1:c:c:c(:c:2)-,:[$([#7]),$([#8])])=[$([#8]),$([#16])]'),
    ('anil_di_alk_furan_A(15)', '[#7](-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-c:1:c(:c(:c(:o:1)-[#6]=[#7]-[#7](-[#1])-[#6]=[!#6&!#1])-[#1])-[#1]'),
    ('ene_five_het_F(15)', 'c:1(:c:c:c:c:c:1)-[#6](-[#1])=!@[#6]-3-[#6](=[#8])-c:2:c:c:c:c:c:2-[#16]-3'),
    ('ene_six_het_A(483)', '[#6]-1(-[#6](~[!#6&!#1]~[#6]-[!#6&!#1]-[#6]-1=[!#6&!#1])~[!#6&!#1])=[#6;!R]-[#1]'),
    ('hzone_phenol_A(479)', 'c:1:c:c(:c(:c:c:1)-[#6]=[#7]-[#7])-[#8]-[#1]'),
    ('anil_di_alk_A(478)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c:c(:c(:[c;!H0,$(c-[#6](-[#1])-[#1]),$(c-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])](:c:1))-[#7])-[#1]'),
    ('indol_3yl_alk(461)', '[n;!H0,$(n-[#6;!H0;!H1])]:1(c(c(c:2:c:1:c:c:c:c:2-[#1])-[#6;X4]-[#1])-[$([#6](-[#1])-[#1]),$([#6]=,:[!#6&!#1]),$([#6](-[#1])-[#7]),$([#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](-[#1])-[#1])])'),
    ('quinone_A(370)', '[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1'),
    ('azo_A(324)', '[#7;!R]=[#7]'),
    ('imine_one_A(321)', '[#6]-[#6](=[!#6&!#1;!R])-[#6](=[!#6&!#1;!R])-[$([#6]),$([#16](=[#8])=[#8])]'),
    ('mannich_A(296)', '[#7]-[#6;X4]-c:1:c:c:c:c:c:1-[#8]-[#1]'),
    ('anil_di_alk_B(251)', 'c:1:c:c(:c:c:c:1-[#7](-[#6;X4])-[#6;X4])-[#6]=[#6]'),
    ('anil_di_alk_C(246)', 'c:1:c:c(:c:c:c:1-[#8]-[#6;X4])-[#7;$([#7!H0]-[#6;X4]),$([#7](-[#6;X4])-[#6;X4])]'),
    ('ene_rhod_A(235)', '[#7]-1-[#6](=[#16])-[#16]-[#6](=[#6])-[#6]-1=[#8]'),
    ('hzone_phenol_B(215)', 'c:1(:c:c:c(:c:c:1)-[#6]=[#7]-[#7])-[#8]-[#1]'),
    ('ene_five_het_A(201)', '[#6]-1(=[#6])-[#6]=[#7]-[!#6&!#1]-[#6]-1=[#8]'),
    ('anil_di_alk_D(198)', 'c:1:c:c(:c:c:c:1-[#7](-[#6;X4])-[#6;X4])-[#6;X4]-[$([#8]-[#1]),$([#6]=[#6]-[#1]),$([#7](-[#6X4])-[#6;X4])]'),
    ('imine_one_isatin(189)', '[#8]=[#6]-2-[#6](=!@[#7]-[#7])-c:1:c:c:c:c:c:1-[#7]-2'),
    ('anil_di_alk_E(186)', '[#6](-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:[c;!H0,$(c-[#6](-[#1])-[#1])](:c:1-[#1]))-[#6&!H0;!H1,$([#6]-[#6;!H0])])-[#1])-[#1]'),
]  # fmt: skip


_PAINS_Q = {}


def _pains_q(sm):
    if sm not in _PAINS_Q:
        _PAINS_Q[sm] = _merge_hs(_parse(sm))
    return _PAINS_Q[sm]


def cheatsheet() -> str:
    return (
        "smarts_match(smiles, smarts) -> anchors; clogp_estimate(smiles) -> Wildman-Crippen logP/MR; "
        "pains_filter(smiles) -> PAINS alert names."
    )
