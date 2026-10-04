# morie.fn -- function file (rootcoder007/morie)
"""Cheminformatics filters on SMILES: the 166 MACCS structural keys, the REOS-style nuisance filter of
rd_filters (property ranges plus 1251 substructure alerts in eight rule sets), and the Ertl-Schuffenhauer
synthetic accessibility score with RDKit-compatible Morgan environments. The SMARTS engine is a copy of
morie.fn.chemsmarts (batch G) extended with two-letter element symbols, isotopes, implicit-H counts and
match enumeration."""

from __future__ import annotations

import math

from ._richresult import RichResult
from .avalon import parse_smiles

__all__ = ["maccs_fingerprint", "reos_filter", "molecular_properties", "morgan_environments", "sa_score"]

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
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
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


def _molecule_core(el, arom, chg, hexp, bonds, explicit_h=False):
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
    kek = list(order)  # valences follow the Kekule form, as RDKit's
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
            tot = None if len(sub) > 2 and any(sum(1 for a in sub if v in rset[a]) > 2 for v in atoms) else 0
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
                kek.append(1)
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
            s += 2 * kek[k]
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
        "ring_bonds": ring_edges,
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
    if c.isupper() and i + 1 < len(s) and s[i + 1].islower() and s[i : i + 2] in _Z:
        return ("elem", _Z[s[i : i + 2]], False), i + 2
    if c == "$":
        j = _match_close(s, i + 1, "(", ")")
        return ("rec", _parse(s[i + 2 : j])), j + 1
    if c == "*":
        return ("true",), i + 1
    if c == "#":
        v, j = _num(s, i + 1)
        return ("z", v), j
    if c.isdigit():
        v, j = _num(s, i)
        return ("iso", v), j
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
    if c == "h":
        v, j = _num(s, i + 1)
        return ("himp", v), j
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
    if t == "iso":
        return m["iso"][i] == p[1]
    if t == "himp":
        return m["himp"][i] >= 1 if p[1] is None else m["himp"][i] == p[1]
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


def _all_matches(q, m, cache, cap=1000):
    """Distinct atom sets of all embeddings of query q (RDKit's uniquified GetSubstructMatches), at most cap."""
    na = len(q["atoms"])
    qn = [[] for _ in range(na)]
    for a, b, bp in q["bonds"]:
        qn[a].append((b, bp))
        qn[b].append((a, bp))
    parent = [None] * na
    for v in range(na):
        earlier = [(u, bp) for u, bp in qn[v] if u < v]
        if earlier:
            parent[v] = earlier[0]
    mp = [-1] * na
    used = set()
    found = set()
    out = []

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
        if len(out) >= cap:
            return
        if v == na:
            key = frozenset(mp)
            if key not in found:
                found.add(key)
                out.append(sorted(mp))
            return
        cands = [j for j, _ in m["nbr"][mp[parent[v][0]]]] if parent[v] is not None else range(m["n"])
        for i in cands:
            if ok_at(v, i):
                mp[v] = i
                used.add(i)
                rec(v + 1)
                mp[v] = -1
                used.discard(i)

    rec(0)
    return out


# ---------------------------------------------------------------- SMILES front end
_TWO_AROM = {"se": (34, "s"), "te": (52, "s"), "as": (33, "p")}


def _pretokenize(smiles):
    """Rewrite bracket atoms for the SMILES reader: element symbols it cannot read become C (the true atomic
    number is restored afterwards), every bracket atom gets an explicit H count, isotopes and atom classes are
    lifted out. Returns the rewritten SMILES, per-atom isotope, bracket flag and atomic-number fixes."""
    s = str(smiles)
    out = []
    iso = []
    brk = []
    zfix = {}
    i = 0
    while i < len(s):
        c = s[i]
        if c == "[":
            j = s.index("]", i)
            body = s[i + 1 : j]
            k = 0
            while k < len(body) and body[k].isdigit():
                k += 1
            iso.append(int(body[:k]) if k else 0)
            rest = body[k:]
            if ":" in rest:
                rest = rest[: rest.index(":")]
            if rest[:2] in _TWO_AROM:
                zfix[len(iso) - 1] = _TWO_AROM[rest[:2]][0]
                rest = _TWO_AROM[rest[:2]][1] + rest[2:]
            elif len(rest) > 1 and rest[0].isupper() and rest[1].islower() and rest[:2] not in ("Cl", "Br"):
                zfix[len(iso) - 1] = _Z[rest[:2]]
                rest = "C" + rest[2:]
            sym = 2 if rest[:2] in ("Cl", "Br") else 1
            if "H" not in rest[sym:]:
                rest = rest[:sym] + "H0" + rest[sym:]
            out.append("[" + rest + "]")
            brk.append(True)
            i = j + 1
        elif s[i : i + 2] in ("Cl", "Br"):
            out.append(s[i : i + 2])
            iso.append(0)
            brk.append(False)
            i += 2
        elif c.isalpha():
            out.append(c)
            iso.append(0)
            brk.append(False)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out), iso, brk, zfix


def _molecule(smiles, explicit_h=False):
    s2, iso, brk, zfix = _pretokenize(smiles)
    el, arom, chg, hexp, bonds, _closures = parse_smiles(s2)
    hs0 = _hcount([_Z[e] for e in el], [bool(a) for a in arom], list(chg), hexp, bonds)
    m = _molecule_core(el, arom, chg, hexp, bonds, explicit_h)
    for a, zz in zfix.items():
        m["z"][a] = zz
    n = len(el)
    m["iso"] = iso + [0] * (m["n"] - n)
    m["himp"] = [0 if brk[i] else hs0[i] for i in range(n)] + [0] * (m["n"] - n)
    m["heavy"] = n
    m["bonds"] = [(a, b) for a, b, _o in bonds]
    return m


def _count(m, q, cache):
    return len(_all_matches(q, m, cache))


def _has(m, q, cache):
    return any(_embed(q, m, i, cache) for i in range(m["n"]))


def _frags(m):
    n = m["heavy"]
    seen = [False] * n
    k = 0
    for s in range(n):
        if seen[s]:
            continue
        k += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v, _b in m["nbr"][u]:
                if v < n and not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return k


# ---------------------------------------------------------------- MACCS keys
# the public MACCS key definitions in RDKit's MACCSkeys.py (G. Landrum, updated with A. Dalke, 2011):
# (key, SMARTS, count) -- the key is set when the SMARTS has more than `count` distinct matches;
# key 1 (isotope) is undefined, 125 (more than one aromatic ring) and 166 (more than one fragment) are computed
_MACCS = [
    (2, "[#104]", 0), (3, "[#32,#33,#34,#50,#51,#52,#82,#83,#84]", 0),
    (4, "[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0), (5, "[Sc,Ti,Y,Zr,Hf]", 0),
    (6, "[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0), (7, "[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),
    (8, "[!#6;!#1]1~*~*~*~1", 0), (9, "[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0), (10, "[Be,Mg,Ca,Sr,Ba,Ra]", 0),
    (11, "*1~*~*~*~1", 0), (12, "[Cu,Zn,Ag,Cd,Au,Hg]", 0), (13, "[#8]~[#7](~[#6])~[#6]", 0),
    (14, "[#16]-[#16]", 0), (15, "[#8]~[#6](~[#8])~[#8]", 0), (16, "[!#6;!#1]1~*~*~1", 0),
    (17, "[#6]#[#6]", 0), (18, "[#5,#13,#31,#49,#81]", 0), (19, "*1~*~*~*~*~*~*~1", 0), (20, "[#14]", 0),
    (21, "[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0), (22, "*1~*~*~1", 0), (23, "[#7]~[#6](~[#8])~[#8]", 0),
    (24, "[#7]-[#8]", 0), (25, "[#7]~[#6](~[#7])~[#7]", 0), (26, "[#6]=;@[#6](@*)@*", 0), (27, "[I]", 0),
    (28, "[!#6;!#1]~[CH2]~[!#6;!#1]", 0), (29, "[#15]", 0), (30, "[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),
    (31, "[!#6;!#1]~[F,Cl,Br,I]", 0), (32, "[#6]~[#16]~[#7]", 0), (33, "[#7]~[#16]", 0), (34, "[CH2]=*", 0),
    (35, "[Li,Na,K,Rb,Cs,Fr]", 0), (36, "[#16R]", 0), (37, "[#7]~[#6](~[#8])~[#7]", 0),
    (38, "[#7]~[#6](~[#6])~[#7]", 0), (39, "[#8]~[#16](~[#8])~[#8]", 0), (40, "[#16]-[#8]", 0),
    (41, "[#6]#[#7]", 0), (42, "F", 0), (43, "[!#6;!#1;!H0]~*~[!#6;!#1;!H0]", 0),
    (44, "[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]", 0), (45, "[#6]=[#6]~[#7]", 0), (46, "Br", 0),
    (47, "[#16]~*~[#7]", 0), (48, "[#8]~[!#6;!#1](~[#8])(~[#8])", 0), (49, "[!+0]", 0),
    (50, "[#6]=[#6](~[#6])~[#6]", 0), (51, "[#6]~[#16]~[#8]", 0), (52, "[#7]~[#7]", 0),
    (53, "[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0), (54, "[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),
    (55, "[#8]~[#16]~[#8]", 0), (56, "[#8]~[#7](~[#8])~[#6]", 0), (57, "[#8R]", 0),
    (58, "[!#6;!#1]~[#16]~[!#6;!#1]", 0), (59, "[#16]!:*:*", 0), (60, "[#16]=[#8]", 0),
    (61, "*~[#16](~*)~*", 0), (62, "*@*!@*@*", 0), (63, "[#7]=[#8]", 0), (64, "*@*!@[#16]", 0),
    (65, "c:n", 0), (66, "[#6]~[#6](~[#6])(~[#6])~*", 0), (67, "[!#6;!#1]~[#16]", 0),
    (68, "[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0), (69, "[!#6;!#1]~[!#6;!#1;!H0]", 0),
    (70, "[!#6;!#1]~[#7]~[!#6;!#1]", 0), (71, "[#7]~[#8]", 0), (72, "[#8]~*~*~[#8]", 0), (73, "[#16]=*", 0),
    (74, "[CH3]~*~[CH3]", 0), (75, "*!@[#7]@*", 0), (76, "[#6]=[#6](~*)~*", 0), (77, "[#7]~*~[#7]", 0),
    (78, "[#6]=[#7]", 0), (79, "[#7]~*~*~[#7]", 0), (80, "[#7]~*~*~*~[#7]", 0), (81, "[#16]~*(~*)~*", 0),
    (82, "*~[CH2]~[!#6;!#1;!H0]", 0), (83, "[!#6;!#1]1~*~*~*~*~1", 0), (84, "[NH2]", 0),
    (85, "[#6]~[#7](~[#6])~[#6]", 0), (86, "[C;H2,H3][!#6;!#1][C;H2,H3]", 0), (87, "[F,Cl,Br,I]!@*@*", 0),
    (88, "[#16]", 0), (89, "[#8]~*~*~*~[#8]", 0),
    (90, "[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]", 0),
    (91, "[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),"
         "$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]", 0),
    (92, "[#8]~[#6](~[#7])~[#6]", 0), (93, "[!#6;!#1]~[CH3]", 0), (94, "[!#6;!#1]~[#7]", 0),
    (95, "[#7]~*~*~[#8]", 0), (96, "*1~*~*~*~*~1", 0), (97, "[#7]~*~*~*~[#8]", 0),
    (98, "[!#6;!#1]1~*~*~*~*~*~1", 0), (99, "[#6]=[#6]", 0), (100, "*~[CH2]~[#7]", 0),
    (101, "[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),"
          "$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),"
          "$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),"
          "$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),"
          "$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]", 0),
    (102, "[!#6;!#1]~[#8]", 0), (103, "Cl", 0), (104, "[!#6;!#1;!H0]~*~[CH2]~*", 0), (105, "*@*(@*)@*", 0),
    (106, "[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0), (107, "[F,Cl,Br,I]~*(~*)~*", 0),
    (108, "[CH3]~*~*~*~[CH2]~*", 0), (109, "*~[CH2]~[#8]", 0), (110, "[#7]~[#6]~[#8]", 0),
    (111, "[#7]~*~[CH2]~*", 0), (112, "*~*(~*)(~*)~*", 0), (113, "[#8]!:*:*", 0), (114, "[CH3]~[CH2]~*", 0),
    (115, "[CH3]~*~[CH2]~*", 0), (116, "[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]", 0),
    (117, "[#7]~*~[#8]", 0), (118, "[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]", 1), (119, "[#7]=*", 0),
    (120, "[!#6;R]", 1), (121, "[#7;R]", 0), (122, "*~[#7](~*)~*", 0), (123, "[#8]~[#6]~[#8]", 0),
    (124, "[!#6;!#1]~[!#6;!#1]", 0), (126, "*!@[#8]!@*", 0), (127, "*@*!@[#8]", 1),
    (128, "[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),"
          "$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]", 0),
    (129, "[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]", 0),
    (130, "[!#6;!#1]~[!#6;!#1]", 1), (131, "[!#6;!#1;!H0]", 1), (132, "[#8]~*~[CH2]~*", 0),
    (133, "*@*!@[#7]", 0), (134, "[F,Cl,Br,I]", 0), (135, "[#7]!:*:*", 0), (136, "[#8]=*", 1),
    (137, "[!C;!c;R]", 0), (138, "[!#6;!#1]~[CH2]~*", 1), (139, "[O;!H0]", 0), (140, "[#8]", 3),
    (141, "[CH3]", 2), (142, "[#7]", 1), (143, "*@*!@[#8]", 0), (144, "*!:*:*!:*", 0),
    (145, "*1~*~*~*~*~*~1", 1), (146, "[#8]", 2), (147, "[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]", 0),
    (148, "*~[!#6;!#1](~*)~*", 0), (149, "[C;H3,H4]", 1), (150, "*!@*@*!@*", 0), (151, "[#7;!H0]", 0),
    (152, "[#8]~[#6](~[#6])~[#6]", 0), (153, "[!#6;!#1]~[CH2]~*", 0), (154, "[#6]=[#8]", 0),
    (155, "*!@[CH2]!@*", 0), (156, "[#7]~*(~*)~*", 0), (157, "[#6]-[#8]", 0), (158, "[#6]-[#7]", 0),
    (159, "[#8]", 1), (160, "[C;H3,H4]", 0), (161, "[#7]", 0), (162, "a", 0), (163, "*1~*~*~*~*~*~1", 0),
    (164, "[#8]", 0), (165, "[R]", 0),
]  # fmt: skip
_MACCS_Q = {}


def maccs_fingerprint(smiles) -> RichResult:
    r"""The 166 public MACCS structural keys as SMARTS indicators (RDKit's MACCSkeys definitions).

    Key ``k`` is on when its SMARTS has more than the listed number of
    distinct matches (atom sets) in the molecule; key 125 counts SSSR rings
    whose bonds are all aromatic (on when more than one), key 166 counts
    fragments (on when more than one), and key 1 (isotope) is left undefined
    as in RDKit. The molecule is read as for the SMARTS engine (implicit
    hydrogens, Kekule-invariant aromaticity perception).

    Returns ``on_bits`` (sorted key numbers 1..166) and ``bits`` (a 0/1 list
    indexed 0..166, position 0 unused, as RDKit's bit vector).

    References
    ----------
    Durant, J. L., Leland, B. A., Henry, D. R. and Nourse, J. G. (2002).
    Reoptimization of MDL keys for use in drug discovery. *Journal of
    Chemical Information and Computer Sciences*, 42(6), 1273-1280.
    Landrum, G. RDKit ``Chem/MACCSkeys.py`` SMARTS definitions.

    Examples
    --------
    >>> maccs_fingerprint("CNO").on_bits
    [24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164]
    >>> maccs_fingerprint("CCC").on_bits
    [74, 114, 149, 155, 160]
    """
    m = _molecule(smiles)
    cache = {}
    bits = [0] * 167
    for key, sm, cnt in _MACCS:
        if key not in _MACCS_Q:
            _MACCS_Q[key] = _parse(sm)
        q = _MACCS_Q[key]
        hit = _has(m, q, cache) if cnt == 0 else _count(m, q, cache) > cnt
        bits[key] = 1 if hit else 0
    n_arom = 0
    for r in m["ring_bonds"]:
        if all(m["order"][k] == 4 for k in r):
            n_arom += 1
    bits[125] = 1 if n_arom > 1 else 0
    bits[166] = 1 if _frags(m) > 1 else 0
    return RichResult(payload={"on_bits": [k for k in range(167) if bits[k]], "bits": bits})


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


# ---------------------------------------------------------------- molecular properties
# average atomic weights of elements 1..104 as in RDKit's periodic table
_WEIGHT = [
    1.008, 4.003, 6.941, 9.012, 10.812, 12.011, 14.007, 15.999, 18.998, 20.18, 22.99, 24.305, 26.982, 28.086, 30.974,
    32.067, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.944, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546,
    65.39, 69.723, 72.61, 74.922, 78.96, 79.904, 83.8, 85.468, 87.62, 88.906, 91.224, 92.906, 95.94, 98.0, 101.07,
    102.906, 106.42, 107.868, 112.412, 114.818, 118.711, 121.76, 127.6, 126.904, 131.29, 132.905, 137.328, 138.906,
    140.116, 140.908, 144.24, 145.0, 150.36, 151.964, 157.25, 158.925, 162.5, 164.93, 167.26, 168.934, 173.04,
    174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.078, 196.967, 200.59, 204.383, 207.2, 208.98,
    209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.038, 231.036, 238.029, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0,
    252.0, 257.0, 258.0, 259.0, 262.0, 267.0,
]  # fmt: skip
# RDKit Lipinski.cpp (NumHBD, NumHBA, strict NumRotatableBonds)
_HBD = "[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"
_HBA = "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0X2,o,s;+0])]"
_ROTB = (
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CH3])"
    "&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])"
    "&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)"
    "&!$(C([CH3])([CH3])[CH3])&!$([CH3])]"
)
_PROP_Q = {}


def _q(sm):
    if sm not in _PROP_Q:
        _PROP_Q[sm] = _parse(sm)
    return _PROP_Q[sm]


def _tpsa(m):
    """Ertl topological polar surface area over N and O (RDKit MolSurf.cpp contributions)."""
    total = 0.0
    for i in range(m["heavy"]):
        z = m["z"][i]
        if z not in (7, 8):
            continue
        nb = ns = nd = nt = na = 0
        nh = m["h"][i]
        for j, k in m["nbr"][i]:
            if m["z"][j] == 1:
                nh += 1
                continue
            nb += 1
            o = m["order"][k]
            if o == 4:
                na += 1
            elif o == 1:
                ns += 1
            elif o == 2:
                nd += 1
            elif o == 3:
                nt += 1
        q = m["chg"][i]
        r3 = m["smallest"][i] == 3
        t = -1.0
        if z == 7:
            if nb == 1:
                if nh == 0 and q == 0 and nt == 1:
                    t = 23.79
                elif nh == 1 and q == 0 and nd == 1:
                    t = 23.85
                elif nh == 2 and q == 0 and ns == 1:
                    t = 26.02
                elif nh == 2 and q == 1 and nd == 1:
                    t = 25.59
                elif nh == 3 and q == 1 and ns == 1:
                    t = 27.64
            elif nb == 2:
                if nh == 0 and q == 0 and ns == 1 and nd == 1:
                    t = 12.36
                elif nh == 0 and q == 0 and nt == 1 and nd == 1:
                    t = 13.60
                elif nh == 1 and q == 0 and ns == 2 and r3:
                    t = 21.94
                elif nh == 1 and q == 0 and ns == 2:
                    t = 12.03
                elif nh == 0 and q == 1 and nt == 1 and ns == 1:
                    t = 4.36
                elif nh == 1 and q == 1 and nd == 1 and ns == 1:
                    t = 13.97
                elif nh == 2 and q == 1 and ns == 2:
                    t = 16.61
                elif nh == 0 and q == 0 and na == 2:
                    t = 12.89
                elif nh == 1 and q == 0 and na == 2:
                    t = 15.79
                elif nh == 1 and q == 1 and na == 2:
                    t = 14.14
            elif nb == 3:
                if nh == 0 and q == 0 and ns == 3 and r3:
                    t = 3.01
                elif nh == 0 and q == 0 and ns == 3:
                    t = 3.24
                elif nh == 0 and q == 0 and ns == 1 and nd == 2:
                    t = 11.68
                elif nh == 0 and q == 1 and ns == 2 and nd == 1:
                    t = 3.01
                elif nh == 1 and q == 1 and ns == 3:
                    t = 4.44
                elif nh == 0 and q == 0 and na == 3:
                    t = 4.41
                elif nh == 0 and q == 0 and ns == 1 and na == 2:
                    t = 4.93
                elif nh == 0 and q == 0 and nd == 1 and na == 2:
                    t = 8.39
                elif nh == 0 and q == 1 and na == 3:
                    t = 4.10
                elif nh == 0 and q == 1 and ns == 1 and na == 2:
                    t = 3.88
            elif nb == 4 and nh == 0 and ns == 4 and q == 1:
                t = 0.0
            if t < 0.0:
                t = max(30.5 - nb * 8.2 + nh * 1.5, 0.0)
        else:
            if nb == 1:
                if nh == 0 and q == 0 and nd == 1:
                    t = 17.07
                elif nh == 1 and q == 0 and ns == 1:
                    t = 20.23
                elif nh == 0 and q == -1 and ns == 1:
                    t = 23.06
            elif nb == 2:
                if nh == 0 and q == 0 and ns == 2 and r3:
                    t = 12.53
                elif nh == 0 and q == 0 and ns == 2:
                    t = 9.23
                elif nh == 0 and q == 0 and na == 2:
                    t = 13.14
            if t < 0.0:
                t = max(28.5 - nb * 8.6 + nh * 1.5, 0.0)
        total += t
    return total


def _crippen_logp(smiles):
    global _CRIPPEN_Q
    if _CRIPPEN_Q is None:
        _CRIPPEN_Q = [_parse(s) for _, s, _, _ in _CRIPPEN]
    m = _molecule(smiles, explicit_h=True)
    cache = {}
    lp = 0.0
    for i in range(m["n"]):
        for (_name, _s, a, _r), q in zip(_CRIPPEN, _CRIPPEN_Q):
            if _embed(q, m, i, cache):
                lp += a
                break
    return lp


def molecular_properties(smiles) -> RichResult:
    r"""The rd_filters descriptor set: MW, Crippen logP, HBD, HBA, TPSA, rotatable bonds, plus heavy atoms and charge.

    - ``MW``: average molecular weight (RDKit's atomic weights, hydrogens
      included);
    - ``LogP``: Wildman-Crippen atom-typed logP (RDKit ``Crippen.cpp``
      table, first matching type per atom, hydrogens explicit);
    - ``HBD`` / ``HBA``: distinct matches of RDKit's ``NumHBD`` and
      ``NumHBA`` SMARTS;
    - ``TPSA``: Ertl's polar surface area over N and O;
    - ``Rot``: bonds matched by RDKit's strict rotatable-bond SMARTS;
    - ``heavy_atoms`` and total formal ``charge``.

    References
    ----------
    Wildman, S. A. and Crippen, G. M. (1999). Prediction of physicochemical
    parameters by atomic contributions. *J. Chem. Inf. Comput. Sci.*, 39,
    868-873.
    Ertl, P., Rohde, B. and Selzer, P. (2000). Fast calculation of molecular
    polar surface area as a sum of fragment-based contributions. *J. Med.
    Chem.*, 43, 3714-3717.

    Examples
    --------
    >>> p = molecular_properties("CC(=O)Oc1ccccc1C(=O)O")
    >>> round(p.MW, 3), p.HBD, p.HBA, round(p.TPSA, 2), p.Rot
    (180.159, 1, 3, 63.6, 2)
    """
    m = _molecule(smiles)
    cache = {}
    mw = 0.0
    for i in range(m["heavy"]):
        mw += _WEIGHT[m["z"][i] - 1] + m["h"][i] * _WEIGHT[0]
    return RichResult(
        payload={
            "MW": mw,
            "LogP": _crippen_logp(smiles),
            "HBD": _count(m, _q(_HBD), cache),
            "HBA": _count(m, _q(_HBA), cache),
            "TPSA": _tpsa(m),
            "Rot": _count(m, _q(_ROTB), cache),
            "heavy_atoms": sum(1 for i in range(m["heavy"]) if m["z"][i] != 1),
            "charge": sum(m["chg"][: m["heavy"]]),
        }
    )


# ---------------------------------------------------------------- REOS / rd_filters
# alert_collection.csv of rd_filters (P. Walters, MIT licence): (rule set, description, SMARTS); every rule
# allows 0 matches. Sets: Glaxo, Dundee, BMS, PAINS, SureChEMBL, MLSMR, Inpharmatica, LINT.
_ALERTS = [
    ('Glaxo', 'R1 Reactive alkyl halides', '[Br,Cl,I][CX4;CH,CH2]'),
    ('Glaxo', 'R2 Acid halides', '[S,C](=[O,S])[F,Br,Cl,I]'),
    ('Glaxo', 'R3 Carbazides', 'O=CN=[N+]=[N-]'),
    ('Glaxo', 'R4 Sulphate esters', 'COS(=O)O[C,c]'),
    ('Glaxo', 'R5 Sulphonates', 'COS(=O)(=O)[C,c]'),
    ('Glaxo', 'R6 Acid anhydrides', 'C(=O)OC(=O)'),
    ('Glaxo', 'R7 Peroxides', 'OO'),
    ('Glaxo', 'R8 Pentafluorophenyl esters', 'C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)'),
    ('Glaxo', 'R9 Paranitrophenyl esters', 'C(=O)Oc1ccc(N(=O)~[OX1])cc1'),
    ('Glaxo', 'R10 esters of HOBT', 'C(=O)Onnn'),
    ('Glaxo', 'R11 Isocyanates & Isothiocyanates', 'N=C=[S,O]'),
    ('Glaxo', 'R12 Triflates', 'OS(=O)(=O)C(F)(F)F'),
    ('Glaxo', "R13 lawesson's reagent and derivatives", 'P(=S)(S)S'),
    ('Glaxo', 'R14 phosphoramides', 'NP(=O)(N)N'),
    ('Glaxo', 'R15 Aromatic azides', 'cN=[N+]=[N-]'),
    ('Glaxo', 'R16 beta carbonyl quaternary Nitrogen', 'C(=O)C[N+,n+]'),
    ('Glaxo', 'R17 acylhydrazide', '[N;R0][N;R0]C(=O)'),
    ('Glaxo', 'R18 Quaternary C, Cl, I, P or S', '[C+,Cl+,I+,P+,S+]'),
    ('Glaxo', 'R19 Phosphoranes', 'C=P'),
    ('Glaxo', 'R20 Chloramidines', '[Cl]C([C&R0])=N'),
    ('Glaxo', 'R21 Nitroso', '[N&D2](=O)'),
    ('Glaxo', 'R22 P/S Halides', '[P,S][Cl,Br,F,I]'),
    ('Glaxo', 'R23 Carbodiimide', 'N=C=N'),
    ('Glaxo', 'R24 Isonitrile', '[N+]#[C-]'),
    ('Glaxo', 'R25 Triacyloximes', 'C(=O)N(C(=O))OC(=O)'),
    ('Glaxo', 'R26 Cyanohydrins', 'N#CC[OH]'),
    ('Glaxo', 'R27 Acyl cyanides', 'N#CC(=O)'),
    ('Glaxo', 'R28 Sulfonyl cyanides', 'S(=O)(=O)C#N'),
    ('Glaxo', 'R29 Cyanophosphonates', 'P(OCC)(OCC)(=O)C#N'),
    ('Glaxo', 'R30 Azocyanamides', '[N;R0]=[N;R0]C#N'),
    ('Glaxo', 'R31 Azoalkanals', '[N;R0]=[N;R0]CC=O'),
    ('Glaxo', 'I1 Aliphatic methylene chains 7 or more long', '[CD2;R0][CD2;R0][CD2;R0][CD2;R0][CD2;R0][CD2;R0][CD2;R0]'),
    ('Glaxo', 'I2 Compounds with 4 or more acidic groups', '[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH]'),
    ('Glaxo', 'I3 Crown ethers', '[O;R1][C;R1][C;R1][O;R1][C;R1][C;R1][O;R1]'),
    ('Glaxo', 'I4 Disulphides', 'SS'),
    ('Glaxo', 'I5 Thiols', '[SH]'),
    ('Glaxo', 'I6 Epoxides, Thioepoxides, Aziridines', 'C1[O,S,N]C1'),
    ('Glaxo', 'I7 2,4,5 trihydroxyphenyl', 'c([OH])c([OH])c([OH])'),
    ('Glaxo', 'I8 2,3,4 trihydroxyphenyl', 'c([OH])c([OH])cc([OH])'),
    ('Glaxo', 'I9 Hydrazothiourea', 'N=NC(=S)N'),
    ('Glaxo', 'I10 Thiocyanate', 'SC#N'),
    ('Glaxo', 'I11 Benzylic quaternary Nitrogen', 'cC[N+]'),
    ('Glaxo', 'I12 Thioesters', 'C[O,S;R0][C;R0](=S)'),
    ('Glaxo', 'I13 Cyanamides', 'N[CH2]C#N'),
    ('Glaxo', 'I14 Four membered lactones', 'C1(=O)OCC1'),
    ('Glaxo', 'I15 Di and Triphosphates', 'P(=O)([OH])OP(=O)[OH]'),
    ('Glaxo', 'I16 Betalactams', 'N1CCC1=O'),
    ('Glaxo', 'N1 Quinones', 'O=C1[#6]~[#6]C(=O)[#6]~[#6]1'),
    ('Glaxo', 'N2 Polyenes', 'C=CC=CC=CC=C'),
    ('Glaxo', 'N3 Saponin derivatives', 'O1CCCCC1OC2CCC3CCCCC3C2'),
    ('Glaxo', 'N4 Cytochalasin derivatives', 'O=C1NCC2CCCCC21'),
    ('Glaxo', 'N5 Cycloheximide derivatives', 'O=C1CCCC(N1)=O'),
    ('Glaxo', 'N6 Monensin derivatives', 'O1CCCCC1C2CCCO2'),
    ('Glaxo', 'N7 Cyanidin derivatives', '[OH]c1cc([OH])cc2=[O+]C(=C([OH])Cc21)c3cc([OH])c([OH])cc3'),
    ('Glaxo', 'N8 Squalestatin derivatives', 'C12OCCC(O1)CC2'),
    ('Dundee', '> 2 ester groups', 'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]'),
    ('Dundee', '2-halo pyridine', 'n1c([F,Cl,Br,I])cccc1'),
    ('Dundee', 'acid halide', 'C(=O)[Cl,Br,I,F]'),
    ('Dundee', 'acyclic C=C-O', 'C=[C!r]O'),
    ('Dundee', 'acyl cyanide', 'N#CC(=O)'),
    ('Dundee', 'acyl hydrazine', 'C(=O)N[NH2]'),
    ('Dundee', 'aldehyde', '[CH1](=O)'),
    ('Dundee', 'Aliphatic long chain', '[R0;D2][R0;D2][R0;D2][R0;D2]'),
    ('Dundee', 'alkyl halide', '[CX4][Cl,Br,I]'),
    ('Dundee', 'amidotetrazole', 'c1nnnn1C=O'),
    ('Dundee', 'aniline', 'c1cc([NH2])ccc1'),
    ('Dundee', 'azepane', '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1'),
    ('Dundee', 'Azido group', 'N=[N+]=[N-]'),
    ('Dundee', 'Azo group', 'N#N'),
    ('Dundee', 'azocane', '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1'),
    ('Dundee', 'benzidine', '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2'),
    ('Dundee', 'beta-keto/anhydride', '[C,c](=O)[CX4,CR0X3,O][C,c](=O)'),
    ('Dundee', 'biotin analogue', 'C12C(NC(N1)=O)CSC2'),
    ('Dundee', 'Carbocation/anion', '[C+,c+,C-,c-]'),
    ('Dundee', 'catechol', 'c1c([OH])c([OH,NH2,NH])ccc1'),
    ('Dundee', 'charged oxygen or sulfur atoms', '[O+,o+,S+,s+]'),
    ('Dundee', 'chinone', 'C1(=[O,N])C=CC(=[O,N])C=C1'),
    ('Dundee', 'chinone', 'C1(=[O,N])C(=[O,N])C=CC=C1'),
    ('Dundee', 'conjugated nitrile group', 'C=[C!r]C#N'),
    ('Dundee', 'crown ether', '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]'),
    ('Dundee', 'cumarine', 'c1ccc2c(c1)ccc(=O)o2'),
    ('Dundee', 'cyanamide', 'N[CH2]C#N'),
    ('Dundee', 'cyanate/aminonitrile/thiocyanate', '[N,O,S]C#N'),
    ('Dundee', 'cyanohydrins', 'N#CC[OH]'),
    ('Dundee', 'cycloheptane', '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2]1'),
    ('Dundee', 'cycloheptane', '[CR2]1[CR2][CR2]cc[CR2][CR2]1'),
    ('Dundee', 'cyclooctane', '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1'),
    ('Dundee', 'cyclooctane', '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1'),
    ('Dundee', 'diaminobenzene', '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1'),
    ('Dundee', 'diaminobenzene', '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1'),
    ('Dundee', 'diaminobenzene', '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])'),
    ('Dundee', 'diazo group', '[N!R]=[N!R]'),
    ('Dundee', 'diketo group', '[C,c](=O)[C,c](=O)'),
    ('Dundee', 'disulphide', 'SS'),
    ('Dundee', 'enamine', '[CX2R0][NX3R0]'),
    ('Dundee', 'ester of HOBT', 'C(=O)Onnn'),
    ('Dundee', 'four member lactones', 'C1(=O)OCC1'),
    ('Dundee', 'halogenated ring', 'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]'),
    ('Dundee', 'halogenated ring', 'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]'),
    ('Dundee', 'heavy metal', '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si]'),
    ('Dundee', 'het-C-het not in ring', '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]'),
    ('Dundee', 'hydantoin', 'C1NC(=O)NC(=O)1'),
    ('Dundee', 'hydrazine', 'N[NH2]'),
    ('Dundee', 'hydroquinone', '[OH]c1ccc([OH,NH2,NH])cc1'),
    ('Dundee', 'hydroxamic acid', 'C(=O)N[OH]'),
    ('Dundee', 'imine', 'C=[N!R]'),
    ('Dundee', 'imine', 'N=[CR0][N,n,O,S]'),
    ('Dundee', 'iodine', 'I'),
    ('Dundee', 'isocyanate', 'N=C=O'),
    ('Dundee', 'isolated alkene', '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]'),
    ('Dundee', 'ketene', 'C=C=O'),
    ('Dundee', 'methylidene-1,3-dithiole', 'S1C=CSC1=S'),
    ('Dundee', 'Michael acceptor', 'C=!@CC=[O,S]'),
    ('Dundee', 'Michael acceptor', '[$([CH]),$(CC)]#CC(=O)[C,c]'),
    ('Dundee', 'Michael acceptor', '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]'),
    ('Dundee', 'Michael acceptor', 'C=C(C=O)C=O'),
    ('Dundee', 'Michael acceptor', '[$([CH]),$(CC)]#CC(=O)O[C,c]'),
    ('Dundee', 'N oxide', '[NX2,nX3][OX1]'),
    ('Dundee', 'N-acyl-2-amino-5-mercapto-1,3,4-thiadiazole', 's1c(S)nnc1NC=O'),
    ('Dundee', 'N-C-halo', 'NC[F,Cl,Br,I]'),
    ('Dundee', 'N-halo', '[NX3,NX4][F,Cl,Br,I]'),
    ('Dundee', 'N-hydroxyl pyridine', 'n[OH]'),
    ('Dundee', 'nitro group', '[N+](=O)[O-]'),
    ('Dundee', 'N-nitroso', '[#7]-N=O'),
    ('Dundee', 'oxime', '[C,c]=N[OH]'),
    ('Dundee', 'oxime', '[C,c]=NOC=O'),
    ('Dundee', 'Oxygen-nitrogen single bond', '[OR0,NR0][OR0,NR0]'),
    ('Dundee', 'perfluorinated chain', '[CX4](F)(F)[CX4](F)F'),
    ('Dundee', 'peroxide', 'OO'),
    ('Dundee', 'phenol ester', 'c1ccccc1OC(=O)[#6]'),
    ('Dundee', 'phenyl carbonate', 'c1ccccc1OC(=O)O'),
    ('Dundee', 'phosphor', 'P'),
    ('Dundee', 'phthalimide', '[cR,CR]~C(=O)NC(=O)~[cR,CR]'),
    ('Dundee', 'Polycyclic aromatic hydrocarbon', 'a1aa2a3a(a1)A=AA=A3=AA=A2'),
    ('Dundee', 'Polycyclic aromatic hydrocarbon', 'a21aa3a(aa1aaaa2)aaaa3'),
    ('Dundee', 'Polycyclic aromatic hydrocarbon', 'a31a(a2a(aa1)aaaa2)aaaa3'),
    ('Dundee', 'polyene', '[CR0]=[CR0][CR0]=[CR0]'),
    ('Dundee', 'quaternary nitrogen', '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]'),
    ('Dundee', 'quaternary nitrogen', '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]'),
    ('Dundee', 'quaternary nitrogen', '[*]=[N+]=[*]'),
    ('Dundee', 'saponine derivative', 'O1CCCCC1OC2CCC3CCCCC3C2'),
    ('Dundee', 'silicon halogen', '[Si][F,Cl,Br,I]'),
    ('Dundee', 'stilbene', 'c1ccccc1C=Cc2ccccc2'),
    ('Dundee', 'sulfinic acid', '[SX3](=O)[O-,OH]'),
    ('Dundee', 'Sulfonic acid', '[C,c]S(=O)(=O)O[C,c]'),
    ('Dundee', 'Sulfonic acid', 'S(=O)(=O)[O-,OH]'),
    ('Dundee', 'sulfonyl cyanide', 'S(=O)(=O)C#N'),
    ('Dundee', 'sulfur oxygen single bond', '[SX2]O'),
    ('Dundee', 'sulphate', 'OS(=O)(=O)[O-]'),
    ('Dundee', 'Sulphur-nitrogen single bond', '[SX2H0][N]'),
    ('Dundee', 'Thiobenzothiazole', 'c12ccccc1(SC(S)=N2)'),
    ('Dundee', 'thiobenzothiazole', 'c12ccccc1(SC(=S)N2)'),
    ('Dundee', 'Thiocarbonyl group', '[C,c]=S'),
    ('Dundee', 'thioester', 'SC=O'),
    ('Dundee', 'thiol', '[S-]'),
    ('Dundee', 'thiol', '[SH]'),
    ('Dundee', 'Three-membered heterocycle', '*1[O,S,N]*1'),
    ('Dundee', 'triflate', 'OS(=O)(=O)C(F)(F)F'),
    ('Dundee', 'triphenyl methylsilyl', '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)'),
    ('Dundee', 'triple bond', 'C#C'),
    ('BMS', '2halo_pyrazine_3EWG', '[#7;R1]1[#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#7][#6][#6]1'),
    ('BMS', '2halo_pyrazine_5EWG', '[#7;R1]1[#6]([F,Cl,Br,I])[#6;!$(c-N)][#7][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6;!$(c-N)]1'),
    ('BMS', '2halo_pyridazine_3EWG', '[#7;R1]1[#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6][#6][#7]1'),
    ('BMS', '2halo_pyridazine_5EWG', '[#7;R1]1[#6]([F,Cl,Br,I])[#6][#6][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#7]1'),
    ('BMS', '2halo_pyridine_3EWG', '[#7;R1]1[#6;!$(c=O)]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6;!$(c-N)][#6][#6;!$(c-N)]1'),
    ('BMS', '2halo_pyridine_5EWG', '[#7;R1]1[#6;!$(c=O)]([F,Cl,Br,I])[#6][#6;!$(c-N)][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6;!$(c=O);!$(c-N)]1'),
    ('BMS', '2halo_pyrimidine_5EWG', '[#7;R1]1[#6]([F,Cl,Br,I])[#7][#6][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]1'),
    ('BMS', '3halo_pyridazine_2EWG', '[#7;R1]1[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]([F,Cl,Br,I])[#6][#6][#7]1'),
    ('BMS', '3halo_pyridazine_4EWG', '[#7;R1]1[#6][#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6][#7]1'),
    ('BMS', '4_pyridone_3_5_EWG', '[#7,#8,#16]1~[#6;H]~[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])~[#6](=O)~[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])~[#6;H]1'),
    ('BMS', '4halo_pyridine_3EWG', '[#7;R1]1[#6;!$(c=O);!$(c-N)][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]([F,Cl,Br,I])[#6][#6;!$(c=O);!$(c-N)]1'),
    ('BMS', '4halo_pyrimidine_2_6EWG', '[#7]1[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#7;R1][#6]([F,Cl,Br,I])[#6][#6]1([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])'),
    ('BMS', '4halo_pyrimidine_5EWG', '[#7]1[#6][#7;R1][#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]1'),
    ('BMS', 'CH2_S#O_3_ring', '[CH2]1[O,S]C1'),
    ('BMS', 'HOBT_ester', 'O=C(-[!N])O[$(nnn),$([#7]-[#7]=[#7])]'),
    ('BMS', 'NO_phosphonate', 'P(=O)ON'),
    ('BMS', 'acrylate', '[CH2]=[C;!$(C-N);!$(C-O)]C(=O)'),
    ('BMS', 'activated_4mem_ring', '[#6]1~[$(C(=O)),$(S(=O))]~[O,S,N]~[$(C(=O)),$(S(=O))]1'),
    ('BMS', 'activated_S#O_3_ring', 'C1~[O,S]~[C,N,O,S]1[a,N,O,S]'),
    ('BMS', 'activated_acetylene', '[$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))]C#[C;!$(C-N);!$(C-n)]'),
    ('BMS', 'activated_diazo', '[N;!R]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])=[N;!R]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])'),
    ('BMS', 'activated_vinyl_ester', 'O=COC=[$(C(S(=O)(=O))),$(C(C(F)(F)(F))),$(C(C#N)),$(C(N(=O)(=O))),$(C([N+](=O)[O-])),$(C(C(=O)));!$(C(N))]'),
    ('BMS', 'activated_vinyl_sulfonate', 'O(-S(=O)(=O))C=[$(C(S(=O)(=O))),$(C(C(F)(F)(F))),$(C(C#N)),$(C(N(=O)(=O))),$(C([N+](=O)[O-])),$(C(C(=O)));!$(C(N))]'),
    ('BMS', 'acyclic_imide', '[C,c][C;!R](=O)[N;!R][C;!R](=O)[C,c]'),
    ('BMS', 'acyl_123_triazole', '[#7;R1]1~[#7;R1]~[#7;R1](-C(=O))~[#6]~[#6]1'),
    ('BMS', 'acyl_134_triazole', '[#7]1~[#7]~[#6]~[#7](-C(=O)[!N])~[#6]1'),
    ('BMS', 'acyl_activated_NO', 'O=C(-[!N])O[$([#7;+]),$(N(C=[O,S,N])(C=[O,S,N]))]'),
    ('BMS', 'acyl_cyanide', 'C(=O)-C#N'),
    ('BMS', 'acyl_imidazole', '[C;!$(C-N)](=O)[#7]1[#6;H1,$([#6]([*;!R]))][#7][#6;H1,$([#6]([*;!R]))][#6;H1,$([#6]([*;!R]))]1'),
    ('BMS', 'acyl_pyrazole', '[C;!$(C-N)](=O)[#7]1[#7][#6;H1,$([#6]([*;!R]))][#6;H1,$([#6]([*;!R]))][#6;H1,$([#6]([*;!R]))]1'),
    ('BMS', 'aldehyde', '[C,c][C;H1](=O)'),
    ('BMS', 'alpha_dicarbonyl', 'C(=O)!@C(=O)'),
    ('BMS', 'alpha_halo_EWG', '[$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-])]-[CH,CH2]-[Cl,Br,I,$(O(S(=O)(=O)))]'),
    ('BMS', 'alpha_halo_amine', '[F,Cl,Br,I,$(O(S(=O)(=O)))]-[CH,CH2;!$(C(F)F)]-[N,n]'),
    ('BMS', 'alpha_halo_carbonyl', 'C(=O)([CH,CH2][Cl,Br,I,$(O(S(=O)(=O)))])'),
    ('BMS', 'alpha_halo_heteroatom', '[N,n,O,S;!$(S(=O)(=O))]-[CH,CH2;!$(C(F)(F))][F,Cl,Br,I,$(O(S(=O)(=O)))]'),
    ('BMS', 'alpha_halo_heteroatom_tert', '[N,n,O,S;!$(S(=O)(=O))]-C([Cl,Br,I,$(O(S(=O)(=O)))])(C)(C)'),
    ('BMS', 'anhydride', '[$(C(=O)),$(C(=S))]-[O,S]-[$(C(=O)),$(C(=S)),$(C(=[N;!R])),$(C(=N(-[C;X4])))]'),
    ('BMS', 'aryl_phosphonate', 'P(=O)-[O;!R]-a'),
    ('BMS', 'aryl_thiocarbonyl', 'a-[S;X2;!R]-[C;!R](=O)'),
    ('BMS', 'azide', '[$(N#[N+]-[N-]),$([N-]=[N+]=N)]'),
    ('BMS', 'aziridine_diazirine', '[C,N]1~[C,N]~N~1'),
    ('BMS', 'azo_amino', '[N]=[N;!R]-[N]'),
    ('BMS', 'azo_aryl', 'c[N;!R;!+]=[N;!R;!+]-c'),
    ('BMS', 'azo_filter1', '[N;!R]=[N;!R]-[N]=[*]'),
    ('BMS', 'azo_filter2', '[N;!$(N-S(=O)(=O));!$(N-C=O)]-[N;!r3;!$(N-S(=O)(=O));!$(N-C=O)]-[N;!$(N-S(=O)(=O));!$(N-C=O)]'),
    ('BMS', 'azo_filter3', '[N;!R]-[N;!R]-[N;!R]'),
    ('BMS', 'azo_filter4', 'a-N=N-[N;H2]'),
    ('BMS', 'bad_boron', '[B-,BH2,BH3,$(B(F)(F))]'),
    ('BMS', 'bad_cations', '[C+,F+,Cl+,Br+,I+,Se+]'),
    ('BMS', 'benzidine_like', 'c([N;!+])1ccc(c2ccc([N;!+])cc2)cc1'),
    ('BMS', 'beta_lactone', '[#6,#15,#16]1(=O)~[#6]~[#6]~[#8,#16]1'),
    ('BMS', 'betalactam', 'C1(=O)~[#6]~[#6]N1'),
    ('BMS', 'betalactam_EWG', 'C1(=O)~[#6]~[#6]N1([$(S(=O)(=O)[C,c,O&D2]),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)[C,c,O&D2])])'),
    ('BMS', 'bis_activated_aryl_ester', 'O=[C,S]Oc1aaa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])1'),
    ('BMS', 'bis_keto_olefin', 'CC(=O)[$([C&H1]),$(C-F),$(C-Cl),$(C-Br),$(C-I)]=[$([C&H1]),$(C-F),$(C-Cl),$(C-Br),$(C-I)]C(=O)C'),
    ('BMS', 'boron_warhead', '[C,c]~[#5]'),
    ('BMS', 'branched_polycyclic_aromatic', 'a1(a2aa(a3aaaaa3)aa(a4aaaaa4)a2)aaaaa1'),
    ('BMS', 'carbodiimide_iso#thio#cyanate', 'N=C=[N,O,S]'),
    ('BMS', 'carbonyl_halide', 'O=C[F,Cl,Br,I]'),
    ('BMS', 'contains_metal', '[$([Ru]),$([Rh]),$([#34]),$([Pd]),$([Sc]),$([Bi]),$([Sb]),$([Ag]),$([Ti]),$([Al]),$([Cd]),$([V]),$([In]),$([Cr]),$([Sn]),$([Mn]),$([La]),$([Fe]),$([Er]),$([Tm]),$([Yb]),$([Lu]),$([Hf]),$([Ta]),$([W]),$([Re]),$([Co]),$([Os]),$([Ni]),$([Ir]),$([Cu]),$([Zn]),$([Ga]),$([Ge]),$([#33]),$([Y]),$([Zr]),$([Nb]),$([Ce]),$([Pr]),$([Nd]),$([Sm]),$([Eu]),$([Gd]),$([Tb]),$([Dy]),$([Ho]),$([Pt]),$([Au]),$([Hg]),$([Tl]),$([Pb]),$([Ac]),$([Th]),$([Pa]),$([Mo]),$([U]),$([Tc]),$([Te]),$([Po]),$([At])]'),
    ('BMS', 'crown_ether', '[$([O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18]),$([O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18]),$([O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][CH,CH2;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18][O,S,#7;R1;r9,r10,r11,r12,r13,r14,r15,r16,r17,r18])]'),
    ('BMS', 'cyano_phosphonate', 'P(O[A,a])(O[A,a])(=O)C#N'),
    ('BMS', 'cyanohydrin', '[C;X4](-[OH,NH1,NH2,SH])(-C#N)'),
    ('BMS', 'diamino_sulfide', '[N,n]~[S;!R;D2]~[N,n]'),
    ('BMS', 'diazo_carbonyl', '[$(N=N=C~C=O),$(N#N-C~C=O)]'),
    ('BMS', 'diazonium', 'a[N+]#N'),
    ('BMS', 'dicarbonyl_sulfonamide', '[$(N(-C(=O))(-C(=O))(-S(=O))),$(n([#6](=O))([#6](=O))([#16](=O)))]'),
    ('BMS', 'disulfide_acyclic', '[S;!R;X2]-[S;!R;X2]'),
    ('BMS', 'disulfonyliminoquinone', 'S(=O)(=O)N=C1C=CC(=NS(=O)(=O))C=C1'),
    ('BMS', 'double_trouble_warhead', 'NC(C[S;D1])C([N;H1]([O;D1]))=O'),
    ('BMS', 'flavanoid', 'O=C2CC(a3aaaaa3)Oa1aaaaa12'),
    ('BMS', 'four_nitriles', 'C#N.C#N.C#N.C#N'),
    ('BMS', 'gte_10_carbon_sb_chain', '[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]-[C;!R]'),
    ('BMS', 'gte_2_N_quats', '[N,n;H0;+;!$(N~O);!$(n~O)].[N,n;H0;+;!$(N~O);!$(n~O)]'),
    ('BMS', 'gte_2_free_phos', 'P([O;D1])=O.P([O;D1])=O'),
    ('BMS', 'gte_2_sulfonic_acid', '[C,c]S(=O)(=O)[O;D1].[C,c]S(=O)(=O)[O;D1]'),
    ('BMS', 'gte_3_COOH', 'C(=O)[O;D1].C(=O)[O;D1].C(=O)[O;D1]'),
    ('BMS', 'gte_3_iodine', '[#53].[#53].[#53]'),
    ('BMS', 'gte_4_basic_N', '[N;!$(N(=[N,O,S,C]));!$(N(S(=O)(=O)));!$(N(C(F)(F)(F)));!$(N(C#N));!$(N(C(=O)));!$(N(C(=S)));!$(N(C(=N)));!$(N(#C));!$(Nc)].[N;!$(N(=[N,O,S,C]));!$(N(S(=O)(=O)));!$(N(C(F)(F)(F)));!$(N(C#N));!$(N(C(=O)));!$(N(C(=S)));!$(N(C(=N)));!$(N(#C));!$(Nc)].[N;!$(N(=[N,O,S,C]));!$(N(S(=O)(=O)));!$(N(C(F)(F)(F)));!$(N(C#N));!$(N(C(=O)));!$(N(C(=S)));!$(N(C(=N)));!$(N(#C));!$(Nc)].[N;!$(N(=[N,O,S,C]));!$(N(S(=O)(=O)));!$(N(C(F)(F)(F)));!$(N(C#N));!$(N(C(=O)));!$(N(C(=S)));!$(N(C(=N)));!$(N(#C));!$(N-c)]'),
    ('BMS', 'gte_4_nitro', '[$([N+](=O)[O-]),$(N(=O)=O)].[$([N+](=O)[O-]),$(N(=O)=O)].[$([N+](=O)[O-]),$(N(=O)=O)].[$([N+](=O)[O-]),$(N(=O)=O)]'),
    ('BMS', 'gte_5_phenolic_OH', 'a[O;D1].a[O;D1].a[O;D1].a[O;D1].a[O;D1]'),
    ('BMS', 'gte_7_aliphatic_OH', 'C[O;D1].C[O;D1].C[O;D1].C[O;D1].C[O;D1].C[O;D1].C[O;D1]'),
    ('BMS', 'gte_7_total_hal', '[Cl,Br,I].[Cl,Br,I].[Cl,Br,I].[Cl,Br,I].[Cl,Br,I].[Cl,Br,I].[Cl,Br,I]'),
    ('BMS', 'gte_8_CF2_or_CH2', '[CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0][CH2,$(C(F)(F));R0]'),
    ('BMS', 'halo_5heterocycle_bis_EWG', '[#7,#8,#16]1[#6]([$(S(=O)(=O)),$([F,Cl]),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])[#6]([$(S(=O)(=O)),$([F,Cl]),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])[#7][#6]1([Cl,Br,I])'),
    ('BMS', 'halo_acrylate', '[$([C;H2]),$([C&H1;$(C-F)]),$([C&H1;$(C-Cl)]),$([C&H1;$(C-Br)]),$([C&H1;$(CI)]),$(C(F)F),$(C(Cl)Cl),$(C(Br)Br),$(C(I)I),$(C(F)Cl),$(C(F)Br),$(C(F)I),$(C(Cl)Br),$(C(Br)I)](=[$([C&H1;$(C(-C(=O)))]),$(C(F)(C(=O))),$(C(Cl)(C(=O))),$(C(Br)(C(=O))),$(C(I)(C(=O))),$(C(C)(C(=O))),$(C(c)(C(=O)))])'),
    ('BMS', 'halo_imino', 'C(=[#7])([Cl,Br,I,$(O(S(=O)(=O)))])'),
    ('BMS', 'halo_olefin_bis_EWG', 'C([Cl,Br,I,$(O(S(=O)(=O)))])=C([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])'),
    ('BMS', 'halo_phenolic_carbonyl', 'C(=O)Oc1c([Cl,F])[cH1,$(c[F,Cl])]c([F,Cl])[cH1,$(c[F,Cl])]c1([F,Cl])'),
    ('BMS', 'halo_phenolic_sulfonyl', 'S(=O)Oc1c([Cl,F])[cH1,$(c[F,Cl])]c([F,Cl])[cH1,$(c[F,Cl])]c1([F,Cl])'),
    ('BMS', 'halogen_heteroatom', '[!C;!c;!H][F,Cl,Br,I]'),
    ('BMS', 'hetero_silyl', '[Si]~[!#6]'),
    ('BMS', 'hydrazine', '[N;X3;!$(N-S(=O)(=O));!$(N-C(F)(F)(F));!$(N-C#N);!$(N-C(=O));!$(N-C(=S));!$(N-C(=N))]-[N;X3;!$(N-S(=O)(=O));!$(N-C(F)(F)(F));!$(N-C#N);!$(N-C(=O));!$(N-C(=S));!$(N-C(=N))]'),
    ('BMS', 'hydrazothiourea', '[N;!R]=NC(=S)N'),
    ('BMS', 'hydroxamate_warhead', 'C([N;H1]([O;D1]))=O'),
    ('BMS', 'hyperval_sulfur', '[$([#16&D3]),$([#16&D4])]=,:[#6]'),
    ('BMS', 'isonitrile', '[N+]#[C-]'),
    ('BMS', 'keto_def_heterocycle', '[$(c([C;!R;!$(C-[N,O,S]);!$(C-[H])](=O))1naaaa1),$(c([C;!R;!$(C-[N,O,S]);!$(C-[H])](=O))1naa[n,s,o]1)]'),
    ('BMS', 'linear_polycyclic_aromatic_I', '[$(a12aaaaa1aa3a(aa(aaaa4)a4a3)a2),$(a12aaaaa1aa3a(aaa4a3aaaa4)a2),$(a12aaaaa1a(aa5)a3a(aaa4a3a5aaa4)a2)]'),
    ('BMS', 'linear_polycyclic_aromatic_II', '[$(a12aaaa4a1a3a(aaaa3aa4)aa2),$(a12aaaaa1a3a(aaa4a3aaaa4)aa2),$(a1(a(aaaa4)a4a3a2aaaa3)a2aaaa1)]'),
    ('BMS', 'maleimide_etc', '[$([C;H1]),$(C(-[F,Cl,Br,I]))]1=[$([C;H1]),$(C(-[F,Cl,Br,I]))]C(=O)[N,O,S]C(=O)1'),
    ('BMS', 'meldrums_acid_deriv', 'O=C1OC(C)(C)OC(C1)=O'),
    ('BMS', 'monofluoroacetate', '[C;H2](F)C(=O)[O,N,S]'),
    ('BMS', 'nitrone', '[C;!R]=[N+][O;D1]'),
    ('BMS', 'nitrosamine', 'N-[N;X2](=O)'),
    ('BMS', 'non_ring_CH2O_acetal', '[O,N,S;!$(S~O)]!@[CH2]!@[O,S,N;!$(S~O)]'),
    ('BMS', 'non_ring_acetal', '[O,N,S;!$(S~O)]!@[C;H1;X4]!@[O,N,S;!$(S~O)]'),
    ('BMS', 'non_ring_ketal', '[O,N,S;!$(S~O)]!@[C;H0;X4](!@[O,N,S;!$(S~O)])(C)'),
    ('BMS', 'ortho_hydroiminoquinone', 'c1c([N;D1])c([N;D1])c[cH1][cH1]1'),
    ('BMS', 'ortho_hydroquinone', 'a1c([O,S;D1])c([O,S;D1])a[cH1][cH1]1'),
    ('BMS', 'ortho_nitrophenyl_carbonyl', '[#6]1(-O-[C;!R](=[O,N;!R]))[#6]([$(N(=O)(=O)),$([N+](=O)[O-])])[#6][#6][#6][#6]1'),
    ('BMS', 'ortho_quinone', '[CH1,$(C(-[Cl,Br,I]))]1=CC(=[O,N,S;!R])C(=[O,N,S])C=[CH1,$(C(-[Cl,Br,I]))]1'),
    ('BMS', 'oxaziridine', 'C1~[O,S]~N1'),
    ('BMS', 'oxime', '[$(C=N[O;D1]);!$(C=[N+])][#6][#6]'),
    ('BMS', 'oxonium', '[o+,O+]'),
    ('BMS', 'para_hydroiminoquinone', 'a1[cH1]c([N;D1])[cH1]ac([N;D1])1'),
    ('BMS', 'para_hydroquinone', 'a1[cH1]c([O,S;D1])[cH1]ac([O,S;D1])1'),
    ('BMS', 'para_nitrophenyl_ester', '[#6]1(-O(-[C;!R](-[!N])(=[O,N;!R])))[#6][#6][#6]([$(N(=O)(=O)),$([N+](=O)[O-])])[#6][#6]1'),
    ('BMS', 'para_quinone', '[CH1,$(C(-[Cl,Br,I]))]1=[CH1,$(C(-[Cl,Br,I]))]C(=[O,N,S])[CH1,$(C(-[Cl,Br,I]))]=[CH1,$(C(-[Cl,Br,I]))]C1(=[O,N,S])'),
    ('BMS', 'paraquat_like', '[#6]1[#6][#6]([#6]2[#6][#6][#7;+][#6][#6]2)[#6][#6][#7;+]1'),
    ('BMS', 'pentafluorophenylester', 'C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)'),
    ('BMS', 'perchloro_cp', 'C1(Cl)(Cl)C(Cl)C(Cl)=C(Cl)C1(Cl)'),
    ('BMS', 'perhalo_dicarbonyl_phenyl', 'c1(C=O)c([Br,Cl,I])c([Br,Cl,I])c([Br,Cl,I])c([Br,Cl,I])c1(C=O)'),
    ('BMS', 'perhalo_phenyl', 'c1c([Br,Cl,I])c([Br,Cl,I])c([Br,Cl,I])c([Br,Cl,I])c1([Br,Cl,I])'),
    ('BMS', 'peroxide', '[#8]~[#8]'),
    ('BMS', 'phenolate_bis_EWG', 'O=[C,S]Oc1aaa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])1'),
    ('BMS', 'phos_serine_warhead', 'NC(COP(O)(O)=O)C(O)=O'),
    ('BMS', 'phos_threonine_warhead', 'NC(C(C)OP(O)(O)=O)C(O)=O'),
    ('BMS', 'phos_tyrosine_warhead', 'NC(Cc1ccc(OP(O)(O)=O)cc1)C(O)=O'),
    ('BMS', 'phosphite', '[c,C]-[P;v3]'),
    ('BMS', 'phosphonium', '[#15;+]~[!O]'),
    ('BMS', 'phosphorane', 'C=P'),
    ('BMS', 'phosphorous_nitrogen_bond', '[#15]~[N,n]'),
    ('BMS', 'phosphorus_phosphorus_bond', 'P~P'),
    ('BMS', 'phosphorus_sulfur_bond', 'P~S'),
    ('BMS', 'polyene', 'C=[C;!R][C;!R]=[C;!R][C;!R]=[C;!R]'),
    ('BMS', 'polyhalo_phenol_a', 'c1c([O;D1])c(-[Cl,Br,I])c(-[Cl,Br,I])cc1.c1c([O;D1])c(-[Cl,Br,I])c(-[Cl,Br,I])cc1'),
    ('BMS', 'polyhalo_phenol_b', 'c1c([O;D1])c(-[Cl,Br,I])cc(-[Cl,Br,I])c1.c1c([O;D1])c(-[Cl,Br,I])cc(-[Cl,Br,I])c1'),
    ('BMS', 'polyhalo_phenol_c', 'c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1.c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1'),
    ('BMS', 'polyhalo_phenol_d', 'c(-[Cl,Br,I])1c([O;D1])c(-[Cl,Br,I])ccc1.c(-[Cl,Br,I])1c([O;D1])c(-[Cl,Br,I])ccc1'),
    ('BMS', 'polyhalo_phenol_e', 'c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1.c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1'),
    ('BMS', 'polysulfide', '[S;D2]-[S;D2]-[S;D2]'),
    ('BMS', 'porphyrin', '[#6;r16,r17,r18]~[#6]1~[#6]~[#6]~[#6](~[#6])~[#7]1'),
    ('BMS', 'primary_halide_sulfate', '[CH2][Cl,Br,I,$(O(S(=O)(=O)[!$(N);!$([O&D1])]))]'),
    ('BMS', 'quat_N_N', '[N,n;R;+]!@[N,n]'),
    ('BMS', 'quat_N_acyl', '[N,n;+]!@C(=O)'),
    ('BMS', 'quinone_methide', '[#6;!$([#6](-[N,O,S]))]1=[#6;!$([#6](-[N,O,S]))][#6](=[#6])[#6;!$([#6](-[N,O,S]))]=[#6;!$([#6](-[N,O,S]))][#6]1(=[O,N,S])'),
    ('BMS', 'rhodanine', 'C(=C)1SC(=S)NC(=O)1'),
    ('BMS', 'secondary_halide_sulfate', '[CH;!$(C=C)][Cl,Br,I,$(O(S(=O)(=O)[!$(N);!$([O&D1])]))]'),
    ('BMS', 'sulf_D2_nitrogen', '[S;D2](-[N;!$(N(=C));!$(N(-S(=O)(=O)));!$(N(-C(=O)))])'),
    ('BMS', 'sulf_D2_oxygen_D2', '[S;D2][O;D2]'),
    ('BMS', 'sulf_D3_nitrogen', '[S;D3](-N)(-[c,C])(-[c,C])'),
    ('BMS', 'sulfite_sulfate_ester', '[C,c]OS(=O)O[C,c]'),
    ('BMS', 'sulfonium', '[S+;X3;$(S-C);!$(S-[O;D1])]'),
    ('BMS', 'sulfonyl_anhydride', '[$(C(=O)),$(S(=O)(=O))][O,S](S(=O)(=O))'),
    ('BMS', 'sulfonyl_halide', 'S(=O)(=O)[F,Cl,Br,I]'),
    ('BMS', 'sulfonyl_heteroatom', '[!#6;!#1;!#11;!#19]O(S(=O)(=O)(-[C,c]))'),
    ('BMS', 'sulphonyl_cyanide', 'S(=O)(=O)C#N'),
    ('BMS', 'tertiary_halide_sulfate', '[C;X4](-[Cl,Br,I,$(O(S(=O)(=O)[!$(N);!$([O&D1])]))])(-[c,C])(-[c,C])(-[c,C])'),
    ('BMS', 'thio_hydroxamate', '[S;D2]([$(N(=C)),$(N(-S(=O)(=O))),$(N(-C(=O)))])'),
    ('BMS', 'thio_xanthate', '[S;!R]-[C;!R](=[S;!R])(-[S;!R])'),
    ('BMS', 'thiocarbonate', 'SC(=O)[O,S]'),
    ('BMS', 'thioester', '[S;!R;H0]C(=[S,O;!R])([!O;!S;!N])'),
    ('BMS', 'thiol_warhead', 'NC(C[S;D1])C(O)=O'),
    ('BMS', 'thiopyrylium', 'c1[S,s;+]cccc1'),
    ('BMS', 'thiosulfoxide', '[C,c][S;X3](~O)-S'),
    ('BMS', 'triamide', '[$(N(-C(=O))(-C(=O))(-C(=O))),$(n([#6](=O))([#6](=O))([#6](=O)))]'),
    ('BMS', 'triaryl_phosphine_oxide', 'P(=O)(a)(a)(a)'),
    ('BMS', 'trichloromethyl_ketone', '[$(C(=O));!$(C-N);!$(C-O);!$(C-S)]C(Cl)(Cl)(Cl)'),
    ('BMS', 'triflate', 'OS(=O)(=O)(C(F)(F)(F))'),
    ('BMS', 'trifluoroacetate_ester', 'C(F)(F)(F)C(=O)O'),
    ('BMS', 'trifluoroacetate_thioester', 'C(F)(F)(F)C(=O)S'),
    ('BMS', 'trifluoromethyl_ketone', '[$(C(=O));!$(C-N);!$(C-O);!$(C-S)]C(F)(F)(F)'),
    ('BMS', 'trihalovinyl_heteroatom', 'C(-[Cl,Br,I])(-[Cl,Br,I])=C(-[Cl,Br,I])(-[N,O,S])'),
    ('BMS', 'trinitro_aromatic', '[$(a1aaa([$(N(=O)(=O)),$([N+](=O)[O-])])a([$(N(=O)(=O)),$([N+](=O)[O-])])a1([$(N(=O)(=O)),$([N+](=O)[O-])])),$(a1aa([$(N(=O)(=O)),$([N+](=O)[O-])])a([$(N(=O)(=O)),$([N+](=O)[O-])])aa1([$(N(=O)(=O)),$([N+](=O)[O-])])),$(a1a([$(N(=O)(=O)),$([N+](=O)[O-])])aa([$(N(=O)(=O)),$([N+](=O)[O-])])aa1([$(N(=O)(=O)),$([N+](=O)[O-])]))]'),
    ('BMS', 'trinitromethane_derivative', 'C([$([N+](=O)[O-]),$(N(=O)=O)])([$([N+](=O)[O-]),$(N(=O)=O)])([$([N+](=O)[O-]),$(N(=O)=O)])'),
    ('BMS', 'tris_activated_aryl_ester', '[$(O=[C,S]Oc1a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa1),$(O=[C,S]Oc1a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aaa([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])1),$(O=[C,S]Oc1a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])a1),$(O=[C,S]Oc1a([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),F,$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])1)]'),
    ('BMS', 'trisub_bis_act_olefin', '[CH;!R;!$(C-N)]=C([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O))])'),
    ('BMS', 'vinyl_carbonyl_EWG', '[C;!R]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])=[C;!R]([C;!R](=O))([!$([#8]);!$([#7])])'),
    ('PAINS', 'ene_six_het_A(483)', '[#6]-1(-[#6](~[!#6&!#1]~[#6]-[!#6&!#1]-[#6]-1=[!#6&!#1])~[!#6&!#1])=[#6;!R]'),
    ('PAINS', 'hzone_phenol_A(479)', 'c:1:c:c(:c(:c:c:1)-[#6]=[#7]-[#7])-[O;H1]'),
    ('PAINS', 'anil_di_alk_A(478)', '[C;H2]N([C;H2])c1cc([$([H]),$([C;H2]),$([O][C;H2][C;H2])])c(N)c([H])c1'),
    ('PAINS', 'indol_3yl_alk(461)', 'n:1(c(c(c:2:c:1:c:c:c:c:2-[H])-[C;D4]-[H])-[$([C;H2]),$([C]=,:[!C]),$([C;H1][N]),$([C;H1]([C;H2])[N;H1][C;H2]),$([C;H1]([C;H2])[C;H2][N;H1][C;H2])])-[$([H]),$([C;H2])]'),
    ('PAINS', 'quinone_A(370)', '[!#6&!#1]=[#6]-1-[#6]=,:[#6]-[#6](=[!#6&!#1])-[#6]=,:[#6]-1'),
    ('PAINS', 'azo_A(324)', '[#7;!R]=[#7]'),
    ('PAINS', 'imine_one_A(321)', '[#6]-[#6](=[!#6&!#1;!R])-[#6](=[!#6&!#1;!R])-[$([#6]),$([#16](=[#8])=[#8])]'),
    ('PAINS', 'mannich_A(296)', '[#7]-[C;X4]-c1ccccc1-[O;H1]'),
    ('PAINS', 'anil_di_alk_B(251)', 'c:1:c:c(:c:c:c:1-[#7](-[#6;X4])-[#6;X4])-[#6]=[#6]'),
    ('PAINS', 'anil_di_alk_C(246)', 'c:1:c:c(:c:c:c:1-[#8]-[#6;X4])-[#7](-[#6;X4])-[$([#1]),$([#6;X4])]'),
    ('PAINS', 'ene_rhod_A(235)', '[#7]-1-[#6](=[#16])-[#16]-[#6](=[#6])-[#6]-1=[#8]'),
    ('PAINS', 'hzone_phenol_B(215)', 'c:1(:c:c:c(:c:c:1)-[#6]=[#7]-[#7])-[#8]-[#1]'),
    ('PAINS', 'ene_five_hetA1(201A)', '[#6]-1(=[#6])-[#6]=[#7]-[#7,#8,#16]-[#6]-1=[#8]'),
    ('PAINS', 'ene_five_het_A(201)', '[#6]-1(=[#6])-[#6]=[#7]-[!#6&!#1]-[#6]-1=[#8]'),
    ('PAINS', 'anil_di_alk_D(198)', 'c:1:c:c(:c:c:c:1-[#7](-[#6;X4])-[#6;X4])-[#6;X4]-[$([#8]-[#1]),$([#6]=[#6]-[#1]),$([#7]-[#6;X4])]'),
    ('PAINS', 'imine_one_isatin(189)', '[#8]=[#6]-2-[#6](=!@[#7]-[#7])-c:1:c:c:c:c:c:1-[#7]-2'),
    ('PAINS', 'anil_di_alk_E(186)', '[#6](-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[$([#1]),$([#6](-[#1])-[#1])])-[#6](-[#1])-[$([#1]),$([#6]-[#1])])-[#1])-[#1]'),
    ('PAINS', 'thiaz_ene_A(128)', '[#6]-1(=[#6](-[$([#1]),$([#6](-[#1])-[#1]),$([#6]=[#8])])-[#16]-[#6](-[#7]-1-[$([#1]),$([#6]-[#1]),$([#6]:[#6])])=[#7;!R])-[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('PAINS', 'pyrrole_A(118)', 'n2(-[#6]:1:[!#1]:[#6]:[#6]:[#6]:[#6]:1)c(cc(c2-[#6;X4])-[#1])-[#6;X4]'),
    ('PAINS', 'catechol_A(92)', 'c:1:c:c(:c(:c:c:1)-[#8;H1])-[#8;H1]'),
    ('PAINS', 'ene_five_het_B(90)', '[#6]-1(=[#6])-[#6](-[#7]=[#6]-[#16]-1)=[#8]'),
    ('PAINS', 'imine_one_fives(89)', '[#6]-1=[!#1]-[!#6&!#1]-[#6](-[#6]-1=[!#6&!#1;!R])=[#8]'),
    ('PAINS', 'ene_five_het_C(85)', '[#6]-1(-[#6](-[#6]=[#6]-[!#6&!#1]-1)=[#6])=[!#6&!#1]'),
    ('PAINS', 'hzone_pipzn(79)', 'CN1[C;H2][C;H2]N(N=[C;H1][#6]=,:[#6])[C;H2][C;H2]1'),
    ('PAINS', 'keto_keto_beta_A(68)', 'c:1-2:c(:c:c:c:c:1)-[#6](=[#8])-[#6;X4]-[#6]-2=[#8]'),
    ('PAINS', 'hzone_pyrrol(64)', 'Cn1cccc1C=NN'),
    ('PAINS', 'ene_one_ene_A(57)', '[#6]=!@[#6](-[!#1])-@[#6](=!@[!#6&!#1])-@[#6](=!@[#6])-[!#1]'),
    ('PAINS', 'cyano_ene_amine_A(56)', 'N#CC=C(N)C(C#N)C#N'),
    ('PAINS', 'ene_five_one_A(55)', 'c:1-2:c(:c:c:c:c:1)-[#6](=[#8])-[#6](=[#6])-[#6]-2=[#8]'),
    ('PAINS', 'cyano_pyridone_A(54)', 'N#Cc1ccc[#7;H1]c1=S'),
    ('PAINS', 'anil_alk_ene(51)', 'c:1:c:c-2:c(:c:c:1)-[#6]-3-[#6](-[#6]-[#7]-2)-[#6]-[#6]=[#6]-3'),
    ('PAINS', 'amino_acridine_A(46)', 'c:1:c:2:c(:c:c:c:1):n:c:3:c(:c:2-[#7]):c:c:c:c:3'),
    ('PAINS', 'ene_five_het_D(46)', '[#6]-1(=[#6])-[#6](=[#8])-[#7]-[#7]-[#6]-1=[#8]'),
    ('PAINS', 'thiophene_amino_Aa(45)', '[H]N([H])c1sc([!#1])c([!#1])c1C=O'),
    ('PAINS', 'ene_five_het_E(44)', '[#7]-[#6]=!@[#6]-2-[#6](=[#8])-c:1:c:c:c:c:c:1-[!#6&!#1]-2'),
    ('PAINS', 'sulfonamide_A(43)', 'NS(=O)(=O)c1cc([F,Cl,Br,I])cc([F,Cl,Br,I])c1O'),
    ('PAINS', 'thio_ketone(43)', '[#6]-[#6](=[#16])-[#6]'),
    ('PAINS', 'sulfonamide_B(41)', '[H]N(c1ccc([O;H1])cc1)S(=O)=O'),
    ('PAINS', 'anil_no_alk(40)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[$([#8]),$([#7]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('PAINS', 'thiophene_amino_Ab(40)', '[$([#1]),$([#6](-[#1])-[#1]),$([#6]:[#6])]-c:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6])-[#6](=[#8])-[#8])-[$([#6]:1:[#6]:[#6]:[#6]:[#6]:[#6]:1),$([#6]:1:[#16]:[#6]:[#6]:[#6]:1)]'),
    ('PAINS', 'het_pyridiniums_A(39)', '[H]c1c([$([N]),$([H])])ccc2ccc[n+]([$([O;X1]),$([C;H3]),$([#6][#6]:[#6]),$([#6][#6][#8]),$([#6][#6](C)=[#8]),$([#6][#6](N)=[#8]),$([#6][#6][#6])])c12'),
    ('PAINS', 'anthranil_one_A(38)', 'CC(=O)c1ccccc1[#7;H1][!$([#6]=[#8])]'),
    ('PAINS', 'cyano_imine_A(37)', '[#7;H1][#7]=[#6](-[#6]#[#7])-[#6]=[!#6&!#1;!R]'),
    ('PAINS', 'diazox_sulfon_A(36)', '[#7](-c:1:c:c:c:c:c:1)-[#16](=[#8])(=[#8])-[#6]:2:[#6]:[#6]:[#6]:[#6]:3:[#7]:[$([#8]),$([#16])]:[#7]:[#6]:2:3'),
    ('PAINS', 'hzone_anil_di_alk(35)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])=[#7]-[#7]-[$([#6](=[#8])-[#6](-[#1])(-[#1])-[#16]-[#6]:[#7]),$([#6](=[#8])-[#6](-[#1])(-[#1])-[!#1]:[!#1]:[#7]),$([#6](=[#8])-[#6]:[#6]-[#8]-[#1]),$([#6]:[#7]),$([#6](-[#1])(-[#1])-[#6](-[#1])-[#8]-[#1])])-[#1])-[#1]'),
    ('PAINS', 'rhod_sat_A(33)', '[#7]-1-[#6](=[#16])-[#16]-[#6;X4]-[#6]-1=[#8]'),
    ('PAINS', 'hzone_enamin(30)', '[#7][#7]=[#6][#6](-[$([#1]),$([#6])])=[#6]([#6])-!@[$([#7]),$([#8])]'),
    ('PAINS', 'pyrrole_B(29)', '[#6;X4]c1ccc([#6]:[#6])n1c2ccccc2'),
    ('PAINS', 'thiophene_hydroxy(28)', 's1ccc(c1)-[#8;H1]'),
    ('PAINS', 'cyano_pyridone_B(27)', '[!#6][#6]1=,:[#7][#6]([#6])=,:[#6](C#N)[#6](=O)[#7]1'),
    ('PAINS', 'imine_one_sixes(27)', '[#6]-1(-[#6](=[#8])-[#7]-[#6](=[#8])-[#7]-[#6]-1=[#8])=[#7]'),
    ('PAINS', 'dyes5A(27)', '[#6]=,:[#6]:[#7]([#6])~[#6]:[#6]=,:[#6][#6]~[#6]:[#7]'),
    ('PAINS', 'naphth_amino_A(25)', 'c1cc2cccc3[#7][#6]=,:[#7]c(c1)c23'),
    ('PAINS', 'naphth_amino_B(25)', '[C;X4]1[N;H1]c3cccc2cccc([N;H1]1)c23'),
    ('PAINS', 'ene_one_ester(24)', '[#6]-[#8]-[#6](=[#8])-[#6](-[#7][#6])=[#6]-[#6](-[#6])=[#8]'),
    ('PAINS', 'thio_dibenzo(23)', 'S=[#6]1[#6]=,:[#6][!#6,!#6][#6]=,:[#6]1'),
    ('PAINS', 'cyano_cyano_A(23)', '[#6](-[#6]#[#7])(-[#6]#[#7])-[#6](-[$([#6]#[#7]),$([#6]=[#7])])-[#6]#[#7]'),
    ('PAINS', 'hzone_acyl_naphthol(22)', '[H]c2c([H])c([H])c1c([H])c(C(=O)NN=C)c(O)c([H])c1c2[H]'),
    ('PAINS', 'het_65_A(21)', 'O=Cc1cnn2c([#8;H1])ccnc12'),
    ('PAINS', 'imidazole_A(19)', 'n:1:c(:n(:c(:c:1-c:2:c:c:c:c:c:2)-c:3:c:c:c:c:c:3)-[#1])-[#6]:[!#1]'),
    ('PAINS', 'ene_cyano_A(19)', '[#6](-[#6]#[#7])(-[#6]#[#7])=[#6]-c:1:c:c:c:c:c:1'),
    ('PAINS', 'anthranil_acid_A(19)', 'C=NNc1ccccc1C(=O)[#8;H1]'),
    ('PAINS', 'dyes3A(19)', '[#6]-,:[#6]:[#7+]=,:[#6][#6]=[#6][#7][#6;X4]'),
    ('PAINS', 'dhp_bis_amino_CN(19)', '[#6]=,:[#6]C1C(C#N)=C(N)SC(N)=C1C#N'),
    ('PAINS', 'het_6_tetrazine(18)', '[#7]~[#6]:1:[#7]:[#7]:[#6](:[$([#7]),$([#6]-[#1]),$([#6]-[#7]-[#1])]:[$([#7]),$([#6]-[#7])]:1)-[$([#7]-[#1]),$([#8]-[#6](-[#1])-[#1])]'),
    ('PAINS', 'ene_one_hal(17)', '[#6]-[#6]=[#6](-[F,Cl,Br,I])-[#6](=[#8])-[#6]'),
    ('PAINS', 'cyano_imine_B(17)', 'N#CC(C#N)=NNc1ccccc1'),
    ('PAINS', 'thiaz_ene_B(17)', '[#6]NC(=O)-!@[#6]1=,:[#6]([$([N]),$(NC(=O)[#6]:[#6])])[#7]([$([#6;H2]-[#6;H1]=[#6;H2]),$([#6]=,:[#6])])[#6](=S)[#16]1'),
    ('PAINS', 'ene_rhod_B(16)', '[H]C([$([#6]-[#35]),$([#6]:[#6](-[#1]):[#6](-[F,Cl,Br,I]):[#6]:[#6]-[F,Cl,Br,I]),$([#6]:[#6](-[#1]):[#6](-[#1]):[#6]-[#16]-[#6](-[#1])-[#1]),$([#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]:[#6]-[#8]-[#6;H2]),$([#6]:1:[#6](-[#6;H2]):[#7](-[#6;H2]):[#6](-[#6;H2]):[#6]:1)])=C1SC(=O)[N]C1=O'),
    ('PAINS', 'thio_carbonate_A(15)', '[#7,#8]c2ccc1oc(=[#8,#16])sc1c2'),
    ('PAINS', 'anil_di_alk_furan_A(15)', '[#7](-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-c:1:c(:c(:c(:o:1)-[#6]=[#7]-[#7](-[#1])-[#6]=[!#6&!#1])-[#1])-[#1]'),
    ('PAINS', 'ene_five_het_F(15)', 'O=[#6]2[#6](=!@[#6]c1ccccc1)Sc3ccccc23'),
    ('PAINS', 'anil_di_alk_F(14)', 'c:1:c:c(:c:c:c:1-[#6;X4]-c:2:c:c:c(:c:c:2)-[#7](-[$([#1]),$([#6;X4])])-[$([#1]),$([#6;X4])])-[#7](-[$([#1]),$([#6;X4])])-[$([#1]),$([#6;X4])]'),
    ('PAINS', 'hzone_anil(14)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#6]=[#7]-[#7]-[#1]'),
    ('PAINS', 'het_5_pyrazole_OH(14)', 'c1(nn(c(c1-[$([#1]),$([#6]-[#1])])-[#8]-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#1])-[#6;X4]'),
    ('PAINS', 'het_thio_666_A(13)', 'c:2(:c:1-[#16]-c:3:c(-[#7](-c:1:c(:c(:c:2-[#1])-[#1])-[#1])-[$([#1]),$([#6](-[#1])(-[#1])-[#1]),$([#6](-[#1])(-[#1])-[#6]-[#1])]):c(:c(~[$([#1]),$([#6]:[#6])]):c(:c:3-[#1])-[$([#1]),$([#7](-[#1])-[#1]),$([#8]-[#6;X4])])~[$([#1]),$([#7](-[#1])-[#6;X4]),$([#6]:[#6])])-[#1]'),
    ('PAINS', 'styrene_A(13)', '[#6]-2-[#6]-c:1:c(:c:c:c:c:1)-[#6](-c:3:c:c:c:c:c-2:3)=[#6]-[#6]'),
    ('PAINS', 'ene_rhod_C(13)', '[#16]-1-[#6](=[#7]-[#6]:[#6])-[#7](-[$([#1]),$([#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#8]),$([#6]:[#6])])-[#6](=[#8])-[#6]-1=[#6](-[#1])-[$([#6]:[#6]:[#6]-[#17]),$([#6]:[!#6&!#1])]'),
    ('PAINS', 'dhp_amino_CN_A(13)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#6]=[#6])-[#8]-1)-[#6](-[#1])-[#1]'),
    ('PAINS', 'cyano_imine_C(12)', '[#8]=[#16](=[#8])-[#6](-[#6]#[#7])=[#7]-[#7]-[#1]'),
    ('PAINS', 'thio_urea_A(12)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2'),
    ('PAINS', 'thiophene_amino_B(12)', 'c:1:c(:c:c:c:c:1)-[#7](-[#1])-c:2:c(:c(:c(:s:2)-[$([#6]=[#8]),$([#6]#[#7]),$([#6](-[#8]-[#1])=[#6])])-[#7])-[$([#6]#[#7]),$([#6](:[#7]):[#7])]'),
    ('PAINS', 'keto_keto_beta_B(12)', '[#6;X4]-1-[#6](=[#8])-[#7]-[#7]-[#6]-1=[#8]'),
    ('PAINS', 'keto_phenone_A(11)', 'c:1:c-3:c(:c:c:c:1)-[#6]:2:[#7]:[!#1]:[#6]:[#6]:[#6]:2-[#6]-3=[#8]'),
    ('PAINS', 'cyano_pyridone_C(11)', '[#6]-1(-[#6](=[#6](-[#6]#[#7])-[#6](~[#8])~[#7]~[#6]-1~[#8])-[#6](-[#1])-[#1])=[#6](-[#1])-[#6]:[#6]'),
    ('PAINS', 'thiaz_ene_C(11)', '[#6]-1(=[#6](-!@[#6]=[#7])-[#16]-[#6](-[#7]-1)=[#8])-[$([F,Cl,Br,I]),$([#7+](:[#6]):[#6])]'),
    ('PAINS', 'hzone_thiophene_A(11)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1]):[!#6&!#1]:[#6](:[#6]:2-[#6](-[#1])=[#7]-[#7](-[#1])-[$([#6]:1:[#7]:[#6]:[#6](-[#1]):[#16]:1),$([#6]:[#6](-[#1]):[#6]-[#1]),$([#6]:[#7]:[#6]:[#7]:[#6]:[#7]),$([#6]:[#7]:[#7]:[#7]:[#7])])-[$([#1]),$([#8]-[#1]),$([#6](-[#1])-[#1])]'),
    ('PAINS', 'ene_quin_methide(10)', '[!#1]:[!#1]-[#6](-[$([#1]),$([#6]#[#7])])=[#6]-1-[#6]=:[#6]-[#6](=[$([#8]),$([#7;!R])])-[#6]=:[#6]-1'),
    ('PAINS', 'het_thio_676_A(10)', 'c:1:c:c-2:c(:c:c:1)-[#6]-[#6](-c:3:c(-[#16]-2):c(:c(-[#1]):c(:c:3-[#1])-[$([#1]),$([#8]),$([#16;X2]),$([#6;X4]),$([#7](-[$([#1]),$([#6;X4])])-[$([#1]),$([#6;X4])])])-[#1])-[#7](-[$([#1]),$([#6;X4])])-[$([#1]),$([#6;X4])]'),
    ('PAINS', 'ene_five_het_G(10)', '[#6]-1(=[#6])-[#6](-[#7,#16,#8][#6](-[!#1])=[#7]-1)=[#8]'),
    ('PAINS', 'acyl_het_A(9)', '[#7+](:[!#1]:[!#1]:[!#1])-[!#1]=[#8]'),
    ('PAINS', 'anil_di_alk_G(9)', '[#6;X4]-[#7](-[#6;X4])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6]2=:[#7][#6]:[#6]:[!#1]2)-[#1])-[#1]'),
    ('PAINS', 'dhp_keto_A(9)', '[#7]-1(-[$([#6;X4]),$([#1])])-[#6]=:[#6](-[#6](=[#8])-[#6]:[#6]:[#6])-[#6](-[#6])-[#6](=[#6]-1-[#6](-[#1])(-[#1])-[#1])-[$([#6]=[#8]),$([#6]#[#7])]'),
    ('PAINS', 'thio_urea_B(9)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2'),
    ('PAINS', 'anil_alk_bim(9)', 'c:1:3:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2)-[#1]):n:c(-[#1]):n:3-[#6]'),
    ('PAINS', 'imine_imine_A(9)', 'c:1:c:c-2:c(:c:c:1)-[#7]=[#6]-[#6]-2=[#7;!R]'),
    ('PAINS', 'thio_urea_C(9)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6](=[#8])-[#6]-2:[!#1]:[!#6&!#1]:[#6]:[#6]-2'),
    ('PAINS', 'imine_one_fives_B(9)', '[#7;!R]=[#6]-2-[#6](=[#8])-c:1:c:c:c:c:c:1-[#16]-2'),
    ('PAINS', 'dhp_amino_CN_B(9)', '[$([#7](-[#1])-[#1]),$([#8]-[#1])]-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-c:1:c(:n(-[#6]):n:c:1)-[#8]-2'),
    ('PAINS', 'anil_OC_no_alk_A(8)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:n:c:1-[#1])-[#8]-c:2:c:c:c:c:c:2)-[#1])-[#1]'),
    ('PAINS', 'het_thio_66_one(8)', '[#6](=[#8])-[#6]-1=[#6]-[#7]-c:2:c(-[#16]-1):c:c:c:c:2'),
    ('PAINS', 'styrene_B(8)', 'c:1:c:c-2:c(:c:c:1)-[#6](-c:3:c(-[$([#16;X2]),$([#6;X4])]-2):c:c:c(:c:3)-[$([#1]),$([#17]),$([#6;X4])])=[#6]-[#6]'),
    ('PAINS', 'het_thio_5_A(8)', '[#6](-[#1])(-[#1])-[#16;X2]-c:1:n:c(:c(:n:1-!@[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2)-[#1]'),
    ('PAINS', 'anil_di_alk_ene_A(8)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2=[#6](-[#1])-c:1:c(:c:c:c:c:1)-[#16;X2]-c:3:c-2:c:c:c:c:3'),
    ('PAINS', 'ene_rhod_D(8)', '[#16]-1-[#6](=!@[#7]-[$([#1]),$([#7](-[#1])-[#6]:[#6])])-[#7](-[$([#1]),$([#6]:[#7]:[#6]:[#6]:[#16])])-[#6](=[#8])-[#6]-1=[#6](-[#1])-[#6]:[#6]-[$([#17]),$([#8]-[#6]-[#1])]'),
    ('PAINS', 'ene_rhod_E(8)', '[#16]-1-[#6](=[#8])-[#7]-[#6](=[#16])-[#6]-1=[#6](-[#1])-[#6]:[#6]'),
    ('PAINS', 'anil_OH_alk_A(8)', 'c:1:c(:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#7](-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#1])-[#1])-[#1]'),
    ('PAINS', 'pyrrole_C(8)', 'n1(-[#6;X4])c(c(-[#1])c(c1-[#6]:[#6])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'thio_urea_D(8)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-c:2:c:c:c:c:c:2'),
    ('PAINS', 'thiaz_ene_D(8)', '[#7](-c:1:c:c:c:c:c:1)-c2[n+]c(cs2)-c:3:c:c:c:c:c:3'),
    ('PAINS', 'ene_rhod_F(8)', 'n:1:c:c:c(:c:1-[#6](-[#1])-[#1])-[#6](-[#1])=[#6]-2-[#6](=[#8])-[#7]-[#6](=[!#6&!#1])-[#7]-2'),
    ('PAINS', 'thiaz_ene_E(8)', '[#6]-1(=[#6](-[#6](-[#1])(-[#6])-[#6])-[#16]-[#6](-[#7]-1-[$([#1]),$([#6](-[#1])-[#1])])=[#8])-[#16]-[#6;R]'),
    ('PAINS', 'het_65_B(7)', '[!#1]:1:[!#1]-2:[!#1](:[!#1]:[!#1]:[!#1]:1)-[#7](-[#1])-[#7](-[#6]-2=[#8])-[#6]'),
    ('PAINS', 'keto_keto_beta_C(7)', 'c:1:c:c-2:c(:c:c:1)-[#6](=[#6](-[#6]-2=[#8])-[#6])-[#8]-[#1]'),
    ('PAINS', 'het_66_A(7)', 'c:2:c:c:1:n:n:c(:n:c:1:c:c:2)-[#6](-[#1])(-[#1])-[#6]=[#8]'),
    ('PAINS', 'thio_urea_E(7)', 'c:1:c:c:c:c:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:n:c:c:c:c:2'),
    ('PAINS', 'thiophene_amino_C(7)', '[#6](-[#1])-[#6](-[#1])(-[#1])-c:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6]-[#6]-[#6]=[#8])-[$([#6](=[#8])-[#8]),$([#6]#[#7])])-[#6](-[#1])-[#1]'),
    ('PAINS', 'hzone_phenone(7)', '[#6](-c:1:c(:c(:c(:c:c:1-[#1])-[$([#6;X4]),$([#1])])-[#1])-[#1])(-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[$([#1]),$([#17])])-[#1])-[#1])=[$([#7]-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]),$([#7]-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]),$([#7]-[#7](-[#1])-[#6](=[#7]-[#1])-[#7](-[#1])-[#1]),$([#6](-[#1])-[#7])]'),
    ('PAINS', 'ene_rhod_G(7)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c(:c:c:c:1)-[#6]:[!#1]:[#6]-[#6](-[#1])=[#6]-2-[#6](=[!#6&!#1])-[#7]-[#6](=[!#6&!#1])-[!#6&!#1]-2'),
    ('PAINS', 'ene_cyano_B(7)', '[#6]-1(=[#6]-[#6](-c:2:c:c(:c(:n:c-1:2)-[#7](-[#1])-[#1])-[#6]#[#7])=[#6])-[#6]#[#7]'),
    ('PAINS', 'dhp_amino_CN_C(7)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#6]:[#6])-[#8]-1)-[#6]#[#7]'),
    ('PAINS', 'het_5_A(7)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6]=[#8])-[#6;X4]-[#6]-2=[#8]'),
    ('PAINS', 'ene_five_het_H(6)', '[#7]-1=[#6]-[#6](-[#6](-[#7]-1)=[#16])=[#6]'),
    ('PAINS', 'thio_amide_A(6)', 'c1(coc(c1-[#1])-[#6](=[#16])-[#7]-2-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[!#1]-[#6](-[#1])(-[#1])-[#6]-2(-[#1])-[#1])-[#1]'),
    ('PAINS', 'ene_cyano_C(6)', '[#6]=[#6](-[#6]#[#7])-[#6](=[#7]-[#1])-[#7]-[#7]'),
    ('PAINS', 'hzone_furan_A(6)', 'c:1(:c(:c(:c(:o:1)-[$([#1]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#6](-[$([#1]),$([#6](-[#1])-[#1])])=[#7]-[#7](-[#1])-c:2:n:c:c:s:2'),
    ('PAINS', 'anil_di_alk_H(6)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-[#6]:2:[#6]:[!#1]:[#6]:[#6]:[#6]:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_65_C(6)', 'n2c1ccccn1c(c2-[$([#6](-[!#1])=[#6](-[#1])-[#6]:[#6]),$([#6]:[#8]:[#6])])-[#7]-[#6]:[#6]'),
    ('PAINS', 'thio_urea_F(6)', '[#6]-1-[#7](-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#7]-1-[#1]'),
    ('PAINS', 'ene_five_het_I(6)', 'c:1(:c:c:c:o:1)-[#6](-[#1])=!@[#6]-3-[#6](=[#8])-c:2:c:c:c:c:c:2-[!#6&!#1]-3'),
    ('PAINS', 'keto_keto_gamma(5)', '[#8]=[#6]-1-[#6;X4]-[#6]-[#6](=[#8])-c:2:c:c:c:c:c-1:2'),
    ('PAINS', 'quinone_B(5)', 'c:1:c:c-2:c(:c:c:1)-[#6](-c3cccc4noc-2c34)=[#8]'),
    ('PAINS', 'het_6_pyridone_OH(5)', '[#8](-[#1])-c:1:n:c(:c:c:c:1)-[#8]-[#1]'),
    ('PAINS', 'hzone_naphth_A(5)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(:c(:c:2-[#1])-[#1])-[#6]=[#7]-[#7](-[#1])-[$([#6]:[#6]),$([#6]=[#16])])-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_ester_A(5)', '[#6]-1=[#6](-[#16]-[#6](-[#6]=[#6]-1)=[#16])-[#7]'),
    ('PAINS', 'ene_misc_A(5)', '[#6]-1=[#6]-[#6](-[#8]-[#6]-1-[#8])(-[#8])-[#6]'),
    ('PAINS', 'cyano_pyridone_D(5)', '[#8]=[#6]-1-[#6](=[#6]-[#6](=[#7]-[#7]-1)-[#6]=[#8])-[#6]#[#7]'),
    ('PAINS', 'het_65_Db(5)', 'C3=CN1C(=NC(=C1-[#7]-[#6])-c:2:c:c:c:c:n:2)C=C3'),
    ('PAINS', 'het_666_A(5)', '[#7]N-2-c:1:c:c:c:c:c:1-[#6](=[#7])-c:3:c-2:c:c:c:c:3'),
    ('PAINS', 'diazox_sulfon_B(5)', 'c:1:c(:c:c:c:c:1)-[#7]-2-[#6](-[#1])-[#6](-[#1])-[#7](-[#6](-[#1])-[#6]-2-[#1])-[#16](=[#8])(=[#8])-c:3:c:c:c:c:4:n:s:n:c:3:4'),
    ('PAINS', 'anil_NH_alk_A(5)', 'c:1(:c(:c-2:c(:c(:c:1-[#1])-[#1])-[#7](-[#6](-[#7]-2-[#1])=[#8])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'sulfonamide_C(5)', 'c:1(:c(:c-3:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-c:2:c:c:c(:c:c:2)-[!#6&!#1])-[#1])-[#8]-[#6](-[#8]-3)(-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_thio_N_55(5)', '[#6](-[#1])-[#6]:2:[#7]:[#7](-c:1:c:c:c:c:c:1):[#16]:3:[!#6&!#1]:[!#1]:[#6]:[#6]:2:3'),
    ('PAINS', 'keto_keto_beta_D(5)', '[#8]=[#6]-[#6]=[#6](-[#1])-[#8]-[#1]'),
    ('PAINS', 'ene_rhod_H(5)', '[#7]-1-2-[#6](=[#7]-[#6](=[#8])-[#6](=[#7]-1)-[#6](-[#1])-[#1])-[#16]-[#6](=[#6](-[#1])-[#6]:[#6])-[#6]-2=[#8]'),
    ('PAINS', 'imine_ene_A(5)', '[#6]:[#6]-[#6](-[#1])=[#6](-[#1])-[#6](-[#1])=[#7]-[#7](-[#6;X4])-[#6;X4]'),
    ('PAINS', 'het_thio_656a(5)', 'c:1:3:c(:c:c:c:c:1):c:2:n:n:c(-[#16]-[#6](-[#1])(-[#1])-[#6]=[#8]):n:c:2:n:3-[#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1]'),
    ('PAINS', 'pyrrole_D(5)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#1])-[#1])-[#1]'),
    ('PAINS', 'pyrrole_E(5)', 'n2(-[#6]:1:[!#1]:[!#6&!#1]:[!#1]:[#6]:1-[#1])c(c(-[#1])c(c2-[#6;X4])-[#1])-[#6;X4]'),
    ('PAINS', 'thio_urea_G(5)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6]([#7;R])[#7;R]'),
    ('PAINS', 'anisol_A(5)', 'c:1(:c(:c(:c(:c(:c:1-[$([#1]),$([#6](-[#1])-[#1])])-[#1])-[#8]-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[$([#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]),$([#6](-[#1])(-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](=[#16])-[#7]-[#1])])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'pyrrole_F(5)', 'n2(-[#6]:1:[#6](-[#6]#[#7]):[#6]:[#6]:[!#6&!#1]:1)c(c(-[#1])c(c2)-[#1])-[#1]'),
    ('PAINS', 'dhp_amino_CN_D(5)', '[#7](-[#1])(-[#1])-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-c:1:c(:c:c:s:1)-[#8]-2'),
    ('PAINS', 'thiazole_amine_A(4)', '[#7](-[#1])-c:1:n:c(:c:s:1)-c:2:c:n:c(-[#7](-[#1])-[#1]):s:2'),
    ('PAINS', 'het_6_imidate_A(4)', '[#7]=[#6]-1-[#7](-[#1])-[#6](=[#6](-[#7]-[#1])-[#7]=[#7]-1)-[#7]-[#1]'),
    ('PAINS', 'anil_OC_no_alk_B(4)', 'c:1:c(:c:2:c(:c:c:1):c:c:c:c:2)-[#8]-c:3:c(:c(:c(:c(:c:3-[#1])-[#1])-[#7]-[#1])-[#1])-[#1]'),
    ('PAINS', 'styrene_C(4)', 'c:1:c:c-2:c(:c:c:1)-[#6]-[#16]-c3c(-[#6]-2=[#6])ccs3'),
    ('PAINS', 'azulene(4)', 'c:2:c:c:c:1:c(:c:c:c:1):c:c:2'),
    ('PAINS', 'furan_acid_A(4)', 'c:1(:c(:c(:c(:o:1)-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6]:[#6])-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'cyano_pyridone_E(4)', '[!#1]:[#6]-[#6]-1=[#6](-[#1])-[#6](=[#6](-[#6]#[#7])-[#6](=[#8])-[#7]-1-[#1])-[#6]:[#8]'),
    ('PAINS', 'anil_alk_thio(4)', '[#6]-1-3=[#6](-[#6](-[#7]-c:2:c:c:c:c:c-1:2)(-[#6])-[#6])-[#16]-[#16]-[#6]-3=[!#1]'),
    ('PAINS', 'anil_di_alk_I(4)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#6](=[#8])-c:2:c:c:c:c:c:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_thio_6_furan(4)', '[#6](-[#1])(-[#1])-[#16;X2]-c:1:n:n:c(:c(:n:1)-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-c:3:c(:c(:c(:o:3)-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_di_alk_ene_B(4)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2=[#6]-c:1:c(:c:c:c:c:1)-[#6]-2(-[#1])-[#1]'),
    ('PAINS', 'imine_one_B(4)', '[#7](-[#1])(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6](=[#8])-[#6](-[#1])-[#1])-[#7](-[#1])-[$([#7]-[#1]),$([#6]:[#6])]'),
    ('PAINS', 'anil_OC_alk_A(4)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1]):o:c:3:c(-[#1]):c(:c(-[#8]-[#6](-[#1])-[#1]):c(:c:2:3)-[#1])-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'ene_five_het_J(4)', '[#16]=[#6]-1-[#7](-[#1])-[#6]=[#6]-[#6]-2=[#6]-1-[#6](=[#8])-[#8]-[#6]-2=[#6]-[#1]'),
    ('PAINS', 'pyrrole_G(4)', 'n2(-c:1:c(:c:c(:c(:c:1)-[#1])-[$([#7](-[#1])-[#1]),$([#6]:[#7])])-[#1])c(c(-[#1])c(c2-[#1])-[#1])-[#1]'),
    ('PAINS', 'ene_five_het_K(4)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])=[#6]-2-[#6](=[#8])-[!#6&!#1]-[#6]=:[!#1]-2)-[#1])-[#1]'),
    ('PAINS', 'cyano_ene_amine_B(4)', '[#6]=[#6]-[#6](-[#6]#[#7])(-[#6]#[#7])-[#6](-[#6]#[#7])=[#6]-[#7](-[#1])-[#1]'),
    ('PAINS', 'thio_ester_B(4)', '[#6]:[#6]-[#6](=[#16;X1])-[#16;X2]-[#6](-[#1])-[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('PAINS', 'ene_five_het_L(4)', '[#8]=[#6]-3-[#6](=!@[#6](-[#1])-c:1:c:n:c:c:1)-c:2:c:c:c:c:c:2-[#7]-3'),
    ('PAINS', 'hzone_thiophene_B(4)', 'c:1(:c(:c(:c(:s:1)-[#1])-[#1])-[$([#1]),$([#6](-[#1])-[#1])])-[#6](-[#1])=[#7]-[#7](-[#1])-c:2:c:c:c:c:c:2'),
    ('PAINS', 'dhp_amino_CN_E(4)', '[#6](-[#1])(-[#1])-[#16;X2]-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](-[#6]#[#7])-[#6](=[#8])-[#7]-1'),
    ('PAINS', 'het_5_B(4)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#7](-[#1])-[#6]=[#8])-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('PAINS', 'imine_imine_B(3)', '[#6]:[#6]-[#6](-[#1])=[#6](-[#1])-[#6](-[#1])=[#7]-[#7]=[#6]'),
    ('PAINS', 'thiazole_amine_B(3)', 'c:1(:c:c:c(:c:c:1)-[#6](-[#1])-[#1])-c:2:c(:s:c(:n:2)-[#7](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]'),
    ('PAINS', 'imine_ene_one_A(3)', '[#6]-2(-[#6]=[#7]-c:1:c:c:c:c:c:1-[#7]-2)=[#6](-[#1])-[#6]=[#8]'),
    ('PAINS', 'diazox_A(3)', '[#8](-c:1:c:c:c:c:c:1)-c:3:c:c:2:n:o:n:c:2:c:c:3'),
    ('PAINS', 'ene_one_A(3)', '[!#1]:1:[!#1]:[!#1]:[!#1](:[!#1]:[!#1]:1)-[#6](-[#1])=[#6](-[#1])-[#6](-[#7]-c:2:c:c:c:3:c(:c:2):c:c:c(:n:3)-[#7](-[#6])-[#6])=[#8]'),
    ('PAINS', 'anil_OC_no_alk_C(3)', '[#7](-[#1])(-[#1])-c:1:c(:c:c:c:n:1)-[#8]-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('PAINS', 'thiazol_SC_A(3)', '[#6]-[#16;X2]-c:1:n:c(:c:s:1)-[#1]'),
    ('PAINS', 'het_666_B(3)', 'c:1:c-3:c(:c:c:c:1)-[#7](-c:2:c:c:c:c:c:2-[#8]-3)-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'furan_A(3)', 'c:1(:c(:c(:c(:o:1)-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#8]-[#1])-[#6]#[#6]-[#6;X4]'),
    ('PAINS', 'colchicine_A(3)', '[#6]-1(-[#6](=[#6]-[#6]=[#6]-[#6]=[#6]-1)-[#7]-[#1])=[#7]-[#6]'),
    ('PAINS', 'thiophene_C(3)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])=[#6]-[#6](=[#8])-c:1:c(-[#16;X2]):s:c(:c:1)-[$([#6]#[#7]),$([#6]=[#8])]'),
    ('PAINS', 'anil_OC_alk_B(3)', 'c:1:3:c(:c:c:c:c:1)-[#7]-2-[#6](=[#8])-[#6](=[#6](-[F,Cl,Br,I])-[#6]-2=[#8])-[#7](-[#1])-[#6]:[#6]:[#6]:[#6](-[#8]-[#6](-[#1])-[#1]):[#6]:[#6]:3'),
    ('PAINS', 'het_thio_66_A(3)', 'c:1-2:c(:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]=[#6]-2-[#16;X2]-[#6](-[#1])(-[#1])-[#6](=[#8])-c:3:c:c:c:c:c:3'),
    ('PAINS', 'rhod_sat_B(3)', '[#7]-2(-c:1:c:c:c:c:c:1-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#6](-[#1])(-[#1])-[!#1]:[!#1]:[!#1]:[!#1]:[!#1])-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('PAINS', 'ene_rhod_I(3)', '[#7]-2(-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](=[#6](-[#1])-c:1:c:c:c:c(:c:1)-[Br])-[#6]-2=[#8]'),
    ('PAINS', 'keto_thiophene(3)', 'c:1(:c(:c:2:c(:s:1):c:c:c:c:2)-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'imine_imine_C(3)', '[#7](-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])=[#7]-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#6](-[#1])-[#1])-[#6]:[#6]'),
    ('PAINS', 'het_65_pyridone_A(3)', '[#6]:2(:[#6](-[#6](-[#1])-[#1]):[#6]-1:[#6](-[#7]=[#6](-[#7](-[#6]-1=[!#6&!#1;X1])-[#6](-[#1])-[$([#6](=[#8])-[#8]),$([#6]:[#6])])-[$([#1]),$([#16]-[#6](-[#1])-[#1])]):[!#6&!#1;X2]:2)-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'thiazole_amine_C(3)', 'c:1(:n:c(:c(-[#1]):s:1)-[!#1]:[!#1]:[!#1](-[$([#8]-[#6](-[#1])-[#1]),$([#6](-[#1])-[#1])]):[!#1]:[!#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c(-[#1]):c(:c(-[#1]):o:2)-[#1]'),
    ('PAINS', 'het_thio_pyr_A(3)', 'n:1:c(:c(:c(:c(:c:1-[#16]-[#6]-[#1])-[#6]#[#7])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])-[#1])-[#1])-[#6]:[#6]'),
    ('PAINS', 'melamine_A(3)', 'c:1:4:c(:n:c(:n:c:1-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-[#7](-[#1])-c:3:c:c(:c(:c:c:3-[$([#1]),$([#6](-[#1])-[#1]),$([#16;X2]),$([#8]-[#6]-[#1]),$([#7;X3])])-[$([#1]),$([#6](-[#1])-[#1]),$([#16;X2]),$([#8]-[#6]-[#1]),$([#7;X3])])-[$([#1]),$([#6](-[#1])-[#1]),$([#16;X2]),$([#8]-[#6]-[#1]),$([#7;X3])]):c:c:c:c:4'),
    ('PAINS', 'anil_NH_alk_B(3)', '[#7](-[#1])(-[#6]:1:[#6]:[#6]:[!#1]:[#6]:[#6]:1)-c:2:c:c:c(:c:c:2)-[#7](-[#1])-[#6]-[#1]'),
    ('PAINS', 'rhod_sat_C(3)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#7]-[#6]=[#8])-[#16]-[#6](-[#1])(-[#1])-[#6]-2=[#8]'),
    ('PAINS', 'thiophene_amino_D(3)', '[#6]=[#6]-[#6](=[#8])-[#7]-c:1:c(:c(:c(:s:1)-[#6](=[#8])-[#8])-[#6]-[#1])-[#6]#[#7]'),
    ('PAINS', 'anil_OC_alk_C(3)', '[$([#1]),$([#6](-[#1])-[#1])]-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:n:c:c:n:2'),
    ('PAINS', 'het_thio_65_A(3)', '[#6](-[#1])(-[#1])-[#16;X2]-c3nc1c(n(nc1-[#6](-[#1])-[#1])-c:2:c:c:c:c:c:2)nn3'),
    ('PAINS', 'het_thio_656b(3)', '[#6]-[#6](=[#8])-[#6](-[#1])(-[#1])-[#16;X2]-c:3:n:n:c:2:c:1:c(:c(:c(:c(:c:1:n(:c:2:n:3)-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thiazole_amine_D(3)', 's:1:c(:[n+](-[#6](-[#1])-[#1]):c(:c:1-[#1])-[#6])-[#7](-[#1])-c:2:c:c:c:c:c:2[$([#6](-[#1])-[#1]),$([#6]:[#6])]'),
    ('PAINS', 'thio_urea_H(3)', '[#6]-2(=[#16])-[#7](-[#6](-[#1])(-[#1])-c:1:c:c:c:o:1)-[#6](=[#7]-[#7]-2-[#1])-[#6]:[#6]'),
    ('PAINS', 'cyano_pyridone_F(3)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#6](=[#6]-[#6](=[#7]-2)-[#6]#[#7])-[#6]#[#7]'),
    ('PAINS', 'rhod_sat_D(3)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6])-[#6]-2=[#8]'),
    ('PAINS', 'ene_rhod_J(3)', '[#6](-[#1])(-[#1])-[#7]-2-[#6](=[$([#16]),$([#7])])-[!#6&!#1]-[#6](=[#6]-1-[#6](=[#6](-[#1])-[#6]:[#6]-[#7]-1-[#6](-[#1])-[#1])-[#1])-[#6]-2=[#8]'),
    ('PAINS', 'imine_phenol_A(3)', '[#6]=[#7;!R]-c:1:c:c:c:c:c:1-[#8]-[#1]'),
    ('PAINS', 'thio_carbonate_B(3)', '[#8]=[#6]-2-[#16]-c:1:c(:c(:c:c:c:1)-[#8]-[#6](-[#1])-[#1])-[#8]-2'),
    ('PAINS', 'het_thio_N_5A(3)', '[#7]=[#6]-1-[#7]=[#6]-[#7]-[#16]-1'),
    ('PAINS', 'het_thio_N_65A(3)', '[#7]-2-[#16]-[#6]-1=[#6](-[#6]:[#6]-[#7]-[#6]-1)-[#6]-2=[#16]'),
    ('PAINS', 'anil_di_alk_J(3)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])=[#7]-[#7]=[#6](-[#6])-[#6]:[#6])-[#1])-[#1]'),
    ('PAINS', 'pyrrole_H(3)', 'n1-2cccc1-[#6]=[#7](-[#6])-[#6]-[#6]-2'),
    ('PAINS', 'ene_cyano_D(3)', '[#6](-[#6]#[#7])(-[#6]#[#7])=[#6](-[#16])-[#16]'),
    ('PAINS', 'cyano_cyano_B(3)', '[#6]-1(-[#6]#[#7])(-[#6]#[#7])-[#6](-[#1])(-[#6](=[#8])-[#6])-[#6]-1-[#1]'),
    ('PAINS', 'ene_five_het_M(3)', '[#6]-1=:[#6]-[#6](-[#6](-[$([#8]),$([#16])]-1)=[#6]-[#6]=[#8])=[#8]'),
    ('PAINS', 'cyano_ene_amine_C(3)', '[#6]:[#6]-[#6](=[#8])-[#7](-[#1])-[#6](=[#8])-[#6](-[#6]#[#7])=[#6](-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'thio_urea_I(3)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#7]=[#6]-c:2:c:n:c:c:2'),
    ('PAINS', 'dhp_amino_CN_F(3)', '[#7](-[#1])(-[#1])-[#6]-2=[#6](-[#6]#[#7])-[#6](-[#1])(-c:1:c:c:c:s:1)-[#6](=[#6](-[#6](-[#1])-[#1])-[#8]-2)-[#6](=[#8])-[#8]-[#6]'),
    ('PAINS', 'anthranil_acid_B(3)', 'c:1:c-3:c(:c:c(:c:1)-[#6](=[#8])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#6](=[#8])-[#8]-[#1])-[#6](-[#7](-[#6]-3=[#8])-[#6](-[#1])-[#1])=[#8]'),
    ('PAINS', 'diazox_B(3)', '[Cl]-c:2:c:c:1:n:o:n:c:1:c:c:2'),
    ('PAINS', 'thio_aldehyd_A(3)', '[#6]-[#6](=[#16])-[#1]'),
    ('PAINS', 'thio_amide_B(2)', '[#6;X4]-[#7](-[#1])-[#6](-[#6]:[#6])=[#6](-[#1])-[#6](=[#16])-[#7](-[#1])-c:1:c:c:c:c:c:1'),
    ('PAINS', 'imidazole_B(2)', '[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#16]-[#6](-[#1])(-[#1])-c1cn(cn1)-[#1]'),
    ('PAINS', 'thiazole_amine_E(2)', '[#8]=[#6]-[#7](-[#1])-c:1:c(-[#6]:[#6]):n:c(-[#6](-[#1])(-[#1])-[#6]#[#7]):s:1'),
    ('PAINS', 'thiazole_amine_F(2)', '[#6](-[#1])-[#7](-[#1])-c:1:n:c(:c:s:1)-c2cnc3n2ccs3'),
    ('PAINS', 'thio_ester_C(2)', '[#7]-1-[#6](=[#8])-[#6](=[#6](-[#6])-[#16]-[#6]-1=[#16])-[#1]'),
    ('PAINS', 'ene_one_B(2)', '[#6](-[#16])(-[#7])=[#6](-[#1])-[#6]=[#6](-[#1])-[#6]=[#8]'),
    ('PAINS', 'quinone_C(2)', '[#8]=[#6]-3-c:1:c(:c:c:c:c:1)-[#6]-2=[#6](-[#8]-[#1])-[#6](=[#8])-[#7]-c:4:c-2:c-3:c:c:c:4'),
    ('PAINS', 'keto_naphthol_A(2)', 'c:1:2:c:c:c:c(:c:1:c(:c:c:c:2)-[$([#8]-[#1]),$([#7](-[#1])-[#1])])-[#6](-[#6])=[#8]'),
    ('PAINS', 'thio_amide_C(2)', '[#6](-[#1])(-c:1:c:c:c:c:c:1)(-c:2:c:c:c:c:c:2)-[#6](=[#16])-[#7]-[#1]'),
    ('PAINS', 'phthalimide_misc(2)', '[#7]-2(-[#6](=[#8])-c:1:c(:c(:c(:c(:c:1-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1])-[#6]-2=[#8])-c:3:c(:c:c(:c(:c:3)-[#1])-[#8])-[#1]'),
    ('PAINS', 'sulfonamide_D(2)', 'c:1:c:c(:c:c:c:1-[#7](-[#1])-[#16](=[#8])=[#8])-[#7](-[#1])-[#16](=[#8])=[#8]'),
    ('PAINS', 'anil_NH_alk_C(2)', '[#6](-[#1])-[#7](-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6]-[#1]'),
    ('PAINS', 'het_65_E(2)', 's1c(c(c-2c1-[#7](-[#1])-[#6](-[#6](=[#6]-2-[#1])-[#6](=[#8])-[#8]-[#1])=[#8])-[#7](-[#1])-[#1])-[#6](=[#8])-[#7]-[#1]'),
    ('PAINS', 'hzide_naphth(2)', 'c:2(:c:1:c(:c(:c(:c(:c:1:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#7](-[#1])-[#6]=[#8])-[#1])-[#1])-[#1]'),
    ('PAINS', 'anisol_B(2)', '[#6](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6;X4])-[#1]'),
    ('PAINS', 'thio_carbam_ene(2)', '[#6]-1=[#6]-[#7]-[#6](-[#16]-[#6;X4]-1)=[#16]'),
    ('PAINS', 'thio_amide_D(2)', '[#6](-[#7](-[#6]-[#1])-[#6]-[#1]):[#6]-[#7](-[#1])-[#6](=[#16])-[#6]-[#1]'),
    ('PAINS', 'het_65_Da(2)', 'n2nc(c1cccc1c2-[#6])-[#6]'),
    ('PAINS', 'thiophene_D(2)', 's:1:c(:c(-[#1]):c(:c:1-[#6](=[#8])-[#7](-[#1])-[#7]-[#1])-[#8]-[#6](-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_thio_6_ene(2)', '[#6]-1:[#6]-[#7]=[#6]-[#6](=[#6]-[#7]-[#6])-[#16]-1'),
    ('PAINS', 'cyano_keto_A(2)', '[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6]#[#7])-[#6](=[#8])-[#6]'),
    ('PAINS', 'anthranil_acid_C(2)', 'c2(c(-[#7](-[#1])-[#1])n(-c:1:c:c:c:c:c:1-[#6](=[#8])-[#8]-[#1])nc2-[#6]=[#8])-[$([#6]#[#7]),$([#6]=[#16])]'),
    ('PAINS', 'naphth_amino_C(2)', 'c:2:c:1:c:c:c:c-3:c:1:c(:c:c:2)-[#7](-[#7]=[#6]-3)-[#1]'),
    ('PAINS', 'naphth_amino_D(2)', 'c:2:c:1:c:c:c:c-3:c:1:c(:c:c:2)-[#7]-[#7]=[#7]-3'),
    ('PAINS', 'thiazole_amine_G(2)', 'c1csc(n1)-[#7]-[#7]-[#16](=[#8])=[#8]'),
    ('PAINS', 'het_66_B(2)', 'c:1:c:c:c:2:c(:c:1):n:c(:n:c:2)-[#7](-[#1])-[#6]-3=[#7]-[#6](-[#6]=[#6]-[#7]-3-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'coumarin_A(2)', 'c:1-3:c(:c(:c(:c(:c:1)-[#8]-[#6]-[#1])-[#1])-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#8])-[#8]-3'),
    ('PAINS', 'anthranil_acid_D(2)', 'c:12:c(:c:c:c:n:1)c(c(-[#6](=[#8])~[#8;X1])s2)-[#7](-[#1])-[#1]'),
    ('PAINS', 'het_66_C(2)', 'c:1:2:n:c(:c(:n:c:1:[#6]:[#6]:[#6]:[!#1]:2)-[#6](-[#1])=[#6](-[#8]-[#1])-[#6])-[#6](-[#1])=[#6](-[#8]-[#1])-[#6]'),
    ('PAINS', 'thiophene_amino_E(2)', 'c1csc(c1-[#7](-[#1])-[#1])-[#6](-[#1])=[#6](-[#1])-c2cccs2'),
    ('PAINS', 'het_6666_A(2)', 'c:2:c:c:1:n:c:3:c(:n:c:1:c:c:2):c:c:c:4:c:3:c:c:c:c:4'),
    ('PAINS', 'sulfonamide_E(2)', '[#6]:[#6]-[#7](-[#1])-[#16](=[#8])(=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'anil_di_alk_K(2)', 'c:1:c:c(:c:c:c:1-[#7](-[#1])-[#1])-[#7](-[#6;X3])-[#6;X3]'),
    ('PAINS', 'het_5_C(2)', '[#7]-2=[#6](-c:1:c:c:c:c:c:1)-[#6](-[#1])(-[#1])-[#6](-[#8]-[#1])(-[#6](-[#9])(-[#9])-[#9])-[#7]-2-[$([#6]:[#6]:[#6]:[#6]:[#6]:[#6]),$([#6](=[#16])-[#6]:[#6]:[#6]:[#6]:[#6]:[#6])]'),
    ('PAINS', 'ene_six_het_B(2)', 'c:1:c(:c:c:c:c:1)-[#6](=[#8])-[#6](-[#1])=[#6]-3-[#6](=[#8])-[#7](-[#1])-[#6](=[#8])-[#6](=[#6](-[#1])-c:2:c:c:c:c:c:2)-[#7]-3-[#1]'),
    ('PAINS', 'steroid_A(2)', '[#8]=[#6]-4-[#6]-[#6]-[#6]-3-[#6]-2-[#6](=[#8])-[#6]-[#6]-1-[#6]-[#6]-[#6]-[#6]-1-[#6]-2-[#6]-[#6]-[#6]-3=[#6]-4'),
    ('PAINS', 'het_565_A(2)', 'c:1:2:c:3:c(:c(-[#8]-[#1]):c(:c:1:c(:c:n:2-[#6])-[#6]=[#8])-[#1]):n:c:n:3'),
    ('PAINS', 'thio_imine_ium(2)', '[#6;X4]-[#7+](-[#6;X4]-[#8]-[#1])=[#6]-[#16]-[#6]-[#1]'),
    ('PAINS', 'anthranil_acid_E(2)', '[#6]-3(=[#8])-[#6](=[#6](-[#1])-[#7](-[#1])-c:1:c:c:c:c:c:1-[#6](=[#8])-[#8]-[#1])-[#7]=[#6](-c:2:c:c:c:c:c:2)-[#8]-3'),
    ('PAINS', 'hzone_furan_B(2)', 'c:1(:c(:c(:c(:o:1)-[$([#1]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#6](-[$([#1]),$([#6](-[#1])-[#1])])=[#7]-[#7](-[#1])-c:2:c:c:n:c:c:2'),
    ('PAINS', 'thiophene_E(2)', 'c:1(:c(:c(:c(:s:1)-[$([#1]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#6](-[$([#1]),$([#6](-[#1])-[#1])])-[#6](=[#8])-[#7](-[#1])-c:2:n:c:c:s:2'),
    ('PAINS', 'ene_misc_B(2)', '[#6]:[#6]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6]=[#8])-[#7]-2-[#6](=[#8])-[#6]-1(-[#1])-[#6](-[#1])(-[#1])-[#6]=[#6]-[#6](-[#1])(-[#1])-[#6]-1(-[#1])-[#6]-2=[#8]'),
    ('PAINS', 'het_thio_5_B(2)', '[#6]-1(-[#6]=[#8])(-[#6]:[#6])-[#16;X2]-[#6]=[#7]-[#7]-1-[#1]'),
    ('PAINS', 'thiophene_amino_F(2)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-c:2:c:c:c:c:c:2)-[#6]#[#7])-[#6]:3:[!#1]:[!#1]:[!#1]:[!#1]:[!#1]:3'),
    ('PAINS', 'anil_OC_alk_D(2)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2-[$([#6](-[#1])-[#1]),$([#8]-[#6](-[#1])-[#1])]'),
    ('PAINS', 'tert_butyl_A(2)', '[#6](-[#1])(-[#1])(-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-c:1:c(:c:c(:c(:c:1-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6](-[#1])-[#7])-[#1]'),
    ('PAINS', 'thio_urea_J(2)', 'c:1(:c(:o:c:c:1)-[#6]-[#1])-[#6]=[#7]-[#7](-[#1])-[#6](=[#16])-[#7]-[#1]'),
    ('PAINS', 'het_thio_65_B(2)', '[#7](-[#1])-c1nc(nc2nnc(n12)-[#16]-[#6])-[#7](-[#1])-[#6]'),
    ('PAINS', 'coumarin_B(2)', 'c:1-2:c(:c:c:c:c:1-[#6](-[#1])(-[#1])-[#6](-[#1])=[#6](-[#1])-[#1])-[#6](=[#6](-[#6](=[#8])-[#7](-[#1])-[#6]:[#6])-[#6](=[#8])-[#8]-2)-[#1]'),
    ('PAINS', 'thio_urea_K(2)', '[#6]-2(=[#16])-[#7]-1-[#6]:[#6]-[#7]=[#7]-[#6]-1=[#7]-[#7]-2-[#1]'),
    ('PAINS', 'thiophene_amino_G(2)', '[#6]:[#6]:[#6]:[#6]:[#6]:[#6]-c:1:c:c(:c(:s:1)-[#7](-[#1])-[#6](=[#8])-[#6])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'anil_NH_alk_D(2)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c:c:1-[#7](-[#1])-[#6](-[#1])(-[#6])-[#6](-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_thio_5_C(2)', '[#16]=[#6]-2-[#7](-[#1])-[#7]=[#6](-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#8]-2'),
    ('PAINS', 'thio_keto_het(2)', '[#16]=[#6]-c:1:c:c:c:2:c:c:c:c:n:1:2'),
    ('PAINS', 'het_thio_N_5B(2)', '[#6]~1~[#6](~[#7]~[#7]~[#6](~[#6](-[#1])-[#1])~[#6](-[#1])-[#1])~[#7]~[#16]~[#6]~1'),
    ('PAINS', 'quinone_D(2)', '[#6]-1(-[#6]=:[#6]-[#6]=:[#6]-[#6]-1=[!#6&!#1])=[!#6&!#1]'),
    ('PAINS', 'anil_di_alk_furan_B(2)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:1:c(-[#1]):c(:c(:o:1)-[#6](-[#1])=[#6]-[#6]#[#7])-[#1]'),
    ('PAINS', 'ene_six_het_C(2)', '[#8]=[#6]-1-[#6]:[#6]-[#6](-[#1])(-[#1])-[#7]-[#6]-1=[#6]-[#1]'),
    ('PAINS', 'het_55_A(2)', '[#6]:[#6]-[#7]:2:[#7]:[#6]:1-[#6](-[#1])(-[#1])-[#16;X2]-[#6](-[#1])(-[#1])-[#6]:1-[#6]:2-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])=[#6]-[#1]'),
    ('PAINS', 'het_thio_65_C(2)', 'n:1:c(:n(:c:2:c:1:c:c:c:c:2)-[#6](-[#1])-[#1])-[#16]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6](-[#1])-[#6](-[#1])=[#6]-[#1]'),
    ('PAINS', 'hydroquin_A(2)', 'c:1(:c:c(:c(:c:c:1)-[#8]-[#1])-[#6](=!@[#6]-[#7])-[#6]=[#8])-[#8]-[#1]'),
    ('PAINS', 'anthranil_acid_F(2)', 'c:1(:c:c(:c(:c:c:1)-[#7](-[#1])-[#6](=[#8])-[#6]:[#6])-[#6](=[#8])-[#8]-[#1])-[#8]-[#1]'),
    ('PAINS', 'pyrrole_I(2)', 'n2(-[#6](-[#1])-[#1])c-1c(-[#6]:[#6]-[#6]-1=[#8])cc2-[#6](-[#1])-[#1]'),
    ('PAINS', 'thiophene_amino_H(2)', '[#6](-[#1])-[#7](-[#1])-c:1:c(:c(:c(:s:1)-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'imine_one_fives_C(2)', '[#6]:[#6]-[#7;!R]=[#6]-2-[#6](=[!#6&!#1])-c:1:c:c:c:c:c:1-[#7]-2'),
    ('PAINS', 'keto_phenone_zone_A(2)', 'c:1:c:c:c:c:c:1-[#6](=[#8])-[#7](-[#1])-[#7]=[#6]-3-c:2:c:c:c:c:c:2-c:4:c:c:c:c:c-3:4'),
    ('PAINS', 'dyes7A(2)', 'c:1:c(:c:c:c:c:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])=[#6](-[#1])-[#6]=!@[#6](-[#1])-[#6](-[#1])=[#6]-[#6]=@[#7]-c:2:c:c:c:c:c:2'),
    ('PAINS', 'het_pyridiniums_B(2)', '[#6]:1:2:[!#1]:[#7+](:[!#1]:[#6](:[!#1]:1:[#6]:[#6]:[#6]:[#6]:2)-[*])~[#6]:[#6]'),
    ('PAINS', 'het_5_D(2)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#16]-[#6])-[#6]-2=[#8]'),
    ('PAINS', 'thiazole_amine_H(1)', 'c:1:c:c:c(:c:c:1-[#7](-[#1])-c2nc(c(-[#1])s2)-c:3:c:c:c(:c:c:3)-[#6](-[#1])(-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'thiazole_amine_I(1)', '[#6](-[#1])(-[#1])-[#7](-[#1])-[#6]=[#7]-[#7](-[#1])-c1nc(c(-[#1])s1)-[#6]:[#6]'),
    ('PAINS', 'het_thio_N_5C(1)', '[#6]:[#6]-[#7](-[#1])-[#6](=[#8])-c1c(snn1)-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'sulfonamide_F(1)', '[#8]=[#16](=[#8])(-[#6]:[#6])-[#7](-[#1])-c1nc(cs1)-[#6]:[#6]'),
    ('PAINS', 'thiazole_amine_J(1)', '[#8]=[#16](=[#8])(-[#6]:[#6])-[#7](-[#1])-[#7](-[#1])-c1nc(cs1)-[#6]:[#6]'),
    ('PAINS', 'het_65_F(1)', 's2c:1:n:c:n:c(:c:1c(c2-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7]-[#7]=[#6]-c3ccco3'),
    ('PAINS', 'keto_keto_beta_E(1)', '[#6](=[#8])-[#6](-[#1])=[#6](-[#8]-[#1])-[#6](-[#8]-[#1])=[#6](-[#1])-[#6](=[#8])-[#6]'),
    ('PAINS', 'ene_five_one_B(1)', 'c:2(:c:1-[#6](-[#6](-[#6](-c:1:c(:c(:c:2-[#1])-[#1])-[#1])(-[#1])-[#1])=[#8])=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]'),
    ('PAINS', 'keto_keto_beta_zone(1)', '[#6]:[#6]-[#7](-[#1])-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'thio_urea_L(1)', '[#6;X4]-[#16;X2]-[#6](=[#7]-[!#1]:[!#1]:[!#1]:[!#1])-[#7](-[#1])-[#7]=[#6]'),
    ('PAINS', 'het_thio_urea_ene(1)', '[#6]-1(=[#7]-[#7](-[#6](-[#16]-1)=[#6](-[#1])-[#6]:[#6])-[#6]:[#6])-[#6]=[#8]'),
    ('PAINS', 'cyano_amino_het_A(1)', 'c:1(:c(:c:2:c(:n:c:1-[#7](-[#1])-[#1]):c:c:c(:c:2-[#7](-[#1])-[#1])-[#6]#[#7])-[#6]#[#7])-[#6]#[#7]'),
    ('PAINS', 'tetrazole_hzide(1)', '[!#1]:1:[!#1]:[!#1]:[!#1](:[!#1]:[!#1]:1)-[#6](-[#1])=[#6](-[#1])-[#6](-[#7](-[#1])-[#7](-[#1])-c2nnnn2-[#6])=[#8]'),
    ('PAINS', 'imine_naphthol_A(1)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(:c(:c:2-[#1])-[#1])-[#6](=[#7]-[#6]:[#6])-[#6](-[#1])-[#1])-[#8]-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'misc_anisole_A(1)', 'c:1(:c(:c:2:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1]):c(:c(:c(:c:2-[#7](-[#1])-[#6](-[#1])(-[#1])-[#1])-[#1])-c:3:c(:c(:c(:c(:c:3-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_thio_665(1)', 'c:1:c:c-2:c(:c:c:1)-[#16]-c3c(-[#7]-2)cc(s3)-[#6](-[#1])-[#1]'),
    ('PAINS', 'anil_di_alk_L(1)', 'c:1:c:c:c-2:c(:c:1)-[#6](-[#6](-[#7]-2-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-4-[#6](-c:3:c:c:c:c:c:3-[#6]-4=[#8])=[#8])(-[#1])-[#1])(-[#1])-[#1]'),
    ('PAINS', 'colchicine_B(1)', 'c:1(:c:c:c(:c:c:1)-[#6]-3=[#6]-[#6](-c2cocc2-[#6](=[#6]-3)-[#8]-[#1])=[#8])-[#16]-[#6](-[#1])-[#1]'),
    ('PAINS', 'misc_aminoacid_A(1)', '[#6;X4]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#16]-[#6](-[#1])(-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1]'),
    ('PAINS', 'imidazole_amino_A(1)', 'n:1:c(:n(:c(:c:1-c:2:c:c:c:c:c:2)-c:3:c:c:c:c:c:3)-[#7]=!@[#6])-[#7](-[#1])-[#1]'),
    ('PAINS', 'phenol_sulfite_A(1)', '[#6](-c:1:c:c:c(:c:c:1)-[#8]-[#1])(-c:2:c:c:c(:c:c:2)-[#8]-[#1])-[#8]-[#16](=[#8])=[#8]'),
    ('PAINS', 'het_66_D(1)', 'c:2:c:c:1:n:c(:c(:n:c:1:c:c:2)-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]:[#6])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]:[#6]'),
    ('PAINS', 'misc_anisole_B(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c(-[#6](-[#1])-[#1])c:c:2'),
    ('PAINS', 'tetrazole_A(1)', '[#6](-[#1])(-[#1])-c1nnnn1-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_65_G(1)', '[#6]-2(=[#7]-c1c(c(nn1-[#6](-[#6]-2(-[#1])-[#1])=[#8])-[#7](-[#1])-[#1])-[#7](-[#1])-[#1])-[#6]'),
    ('PAINS', 'misc_trityl_A(1)', '[#6](-[#6]:[#6])(-[#6]:[#6])(-[#6]:[#6])-[#16]-[#6]:[#6]-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'misc_pyridine_OC(1)', '[#8]=[#6](-c:1:c(:c(:n:c(:c:1-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_6_hydropyridone(1)', '[#7]-1=[#6](-[#7](-[#6](-[#6](-[#6]-1(-[#1])-[#6]:[#6])(-[#1])-[#1])=[#8])-[#1])-[#7]-[#1]'),
    ('PAINS', 'misc_stilbene(1)', '[#6]-1(=[#6](-[#6](-[#6](-[#6](-[#6]-1(-[#1])-[#1])(-[#1])-[#6](=[#8])-[#6])(-[#1])-[#6](=[#8])-[#8]-[#1])(-[#1])-[#1])-[#6]:[#6])-[#6]:[#6]'),
    ('PAINS', 'misc_imidazole(1)', '[#6](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[Cl])-[#1])-[#1])(-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[Cl])-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c3nc(c(n3-[#6](-[#1])(-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_NH_no_alk_A(1)', 'n:1:c(:c(:c(:c(:c:1-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'het_6_imidate_B(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#8]-[#1])-[#6]-2=[#6](-[#8]-[#6](-[#7]=[#7]-2)=[#7])-[#7](-[#1])-[#1]'),
    ('PAINS', 'anil_alk_B(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'styrene_anil_A(1)', 'c:1:c:c-3:c(:c:c:1)-c:2:c:c:c(:c:c:2-[#6]-3=[#6](-[#1])-[#6])-[#7](-[#1])-[#1]'),
    ('PAINS', 'misc_aminal_acid(1)', 'c:1:c:c-2:c(:c:c:1)-[#7](-[#6](-[#8]-[#6]-2)(-[#6](=[#8])-[#8]-[#1])-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](-[#1])-[#1]'),
    ('PAINS', 'anil_no_alk_D(1)', 'n:1:c(:c(:c(:c(:c:1-[#7](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('PAINS', 'anil_alk_C(1)', '[#7](-[#1])(-c:1:c:c:c:c:c:1)-[#6](-[#6])(-[#6])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'misc_anisole_C(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#1])-[#8]-[#6]-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])-[#1])-[#6]:[#6]'),
    ('PAINS', 'het_465_misc(1)', 'c:1-2:c:c-3:c(:c:c:1-[#8]-[#6]-[#8]-2)-[#6]-[#6]-3'),
    ('PAINS', 'anthranil_acid_G(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'anil_di_alk_M(1)', 'c:1(:c:4:c(:n:c(:c:1-[#6](-[#1])(-[#1])-[#7]-3-c:2:c(:c(:c(:c(:c:2-[#6](-[#1])(-[#1])-[#6]-3(-[#1])-[#1])-[#1])-[#1])-[#1])-[#1])-[#1]):c(:c(:c(:c:4-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'anthranil_acid_H(1)', 'c:1:c(:c2:c(:c:c:1)c(c(n2-[#1])-[#6]:[#6])-[#6]:[#6])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'thio_urea_M(1)', '[#6]:[#6]-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-c:1:c(:c(:c(:c(:c:1-[F,Cl,Br,I])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thiazole_amine_K(1)', 'n:1:c3:c(:c:c2:c:1nc(s2)-[#7])sc(n3)-[#7]'),
    ('PAINS', 'het_thio_5_imine_A(1)', '[#7]=[#6]-1-[#16]-[#6](=[#7])-[#7]=[#6]-1'),
    ('PAINS', 'thio_amide_E(1)', 'c:1:c(:n:c:c:c:1)-[#6](=[#16])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_thio_676_B(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#6](-c:3:c(-[#16]-[#6]-2(-[#1])-[#1]):c(:c(-[#1]):c(:c:3-[#1])-[#1])-[#1])-[#8]-[#6]:[#6])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'sulfonamide_G(1)', '[#6](-[#1])(-[#1])(-[#1])-c:1:c(:c(:c(:c(:n:1)-[#7](-[#1])-[#16](-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])(=[#8])=[#8])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_thiomorph_Z(1)', '[#6](=[#8])(-[#7]-1-[#6]-[#6]-[#16]-[#6]-[#6]-1)-c:2:c(:c(:c(:c(:c:2-[#16]-[#6](-[#1])-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'naphth_ene_one_A(1)', 'c:1:c:c:3:c:2:c(:c:1)-[#6](-[#6]=[#6](-c:2:c:c:c:3)-[#8]-[#6](-[#1])-[#1])=[#8]'),
    ('PAINS', 'naphth_ene_one_B(1)', 'c:1-3:c:2:c(:c(:c:c:1)-[#7]):c:c:c:c:2-[#6](-[#6]=[#6]-3-[#6](-[F])(-[F])-[F])=[#8]'),
    ('PAINS', 'amino_acridine_A(1)', 'c:1:c:c:c:c:2:c:1:c:c:3:c(:n:2):n:c:4:c(:c:3-[#7]):c:c:c:c:4'),
    ('PAINS', 'keto_phenone_B(1)', 'c:1:c-3:c(:c:c:c:1)-[#6]-2=[#7]-[!#1]=[#6]-[#6]-[#6]-2-[#6]-3=[#8]'),
    ('PAINS', 'hzone_acid_A(1)', 'c:1-3:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#7]-[#7](-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])-[#1])-[#1])-c:4:c-3:c(:c(:c(:c:4-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'sulfonamide_H(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1])-[#1])-[#16](=[#8])(=[#8])-[#7](-[#1])-c:2:n:n:c(:c(:c:2-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_565_indole(1)', 'c2(c(-[#1])n(-[#6](-[#1])-[#1])c:3:c(:c(:c:1n(c(c(c:1:c2:3)-[#1])-[#1])-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1]'),
    ('PAINS', 'pyrrole_J(1)', 'c1(c-2c(c(n1-[#6](-[#8])=[#8])-[#6](-[#1])-[#1])-[#16]-[#6](-[#1])(-[#1])-[#16]-2)-[#6](-[#1])-[#1]'),
    ('PAINS', 'pyrazole_amino_B(1)', 's1ccnc1-c2c(n(nc2-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('PAINS', 'pyrrole_K(1)', 'c1(c(c(c(n1-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'anthranil_acid_I(1)', 'c:1:2(:c(:c(:c(:o:1)-[#6])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6](-[#1]):[#6](-[#1]):[#6](-[#1]):[#6](-[#1]):[#6]:2-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'thio_amide_F(1)', '[!#1]:[#6]-[#6](=[#16])-[#7](-[#1])-[#7](-[#1])-[#6]:[!#1]'),
    ('PAINS', 'ene_one_C(1)', '[#6]-1(=[#8])-[#6](-[#6](-[#6]#[#7])=[#6](-[#1])-[#7])-[#6](-[#7])-[#6]=[#6]-1'),
    ('PAINS', 'het_65_H(1)', 'c2(c-1n(-[#6](-[#6]=[#6]-[#7]-1)=[#8])nc2-c3cccn3)-[#6]#[#7]'),
    ('PAINS', 'cyano_imine_D(1)', '[#8]=[#6]-1-[#6](=[#7]-[#7]-[#6]-[#6]-1)-[#6]#[#7]'),
    ('PAINS', 'cyano_misc_A(1)', 'c:2(:c:1:c:c:c:c:c:1:n:n:c:2)-[#6](-[#6]:[#6])-[#6]#[#7]'),
    ('PAINS', 'ene_misc_C(1)', 'c:1:c:c-2:c(:c:c:1)-[#6]=[#6]-[#6](-[#7]-2-[#6](=[#8])-[#7](-[#1])-c:3:c:c(:c(:c:c:3)-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_66_E(1)', 'c:2:c:c:1:n:c(:c(:n:c:1:c:c:2)-c:3:c:c:c:c:c:3)-c:4:c:c:c:c:c:4-[#8]-[#1]'),
    ('PAINS', 'keto_keto_beta_F(1)', '[#6](-[#1])(-[#1])-[#6](-[#8]-[#1])=[#6](-[#6](=[#8])-[#6](-[#1])-[#1])-[#6](-[#1])-[#6]#[#6]'),
    ('PAINS', 'misc_naphthimidazole(1)', 'c:1:c:4:c(:c:c2:c:1nc(n2-[#1])-[#6]-[#8]-[#6](=[#8])-c:3:c:c(:c:c(:c:3)-[#7](-[#1])-[#1])-[#7](-[#1])-[#1]):c:c:c:c:4'),
    ('PAINS', 'naphth_ene_one_C(1)', 'c:2(:c:1:c:c:c:c-3:c:1:c(:c:c:2)-[#6]=[#6]-[#6]-3=[#7])-[#7]'),
    ('PAINS', 'keto_phenone_C(1)', 'c:2(:c:1:c:c:c:c:c:1:c-3:c(:c:2)-[#6](-c:4:c:c:c:c:c-3:4)=[#8])-[#8]-[#1]'),
    ('PAINS', 'coumarin_C(1)', '[#6]-2(-[#6]=[#7]-c:1:c:c(:c:c:c:1-[#8]-2)-[Cl])=[#8]'),
    ('PAINS', 'thio_est_cyano_A(1)', '[#6]-1=[#6]-[#7](-[#6](-c:2:c-1:c:c:c:c:2)(-[#6]#[#7])-[#6](=[#16])-[#16])-[#6]=[#8]'),
    ('PAINS', 'het_65_imidazole(1)', 'c2(nc:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])n2-[#6])-[#7](-[#1])-[#6](-[#7](-[#1])-c:3:c(:c:c:c:c:3-[#1])-[#1])=[#8]'),
    ('PAINS', 'anthranil_acid_J(1)', '[#7](-[#1])(-[#6]:[#6])-c:1:c(-[#6](=[#8])-[#8]-[#1]):c:c:c(:n:1)-[#6]:[#6]'),
    ('PAINS', 'colchicine_het(1)', 'c:1-3:c(:c:c:c:c:1)-[#16]-[#6](=[#7]-[#7]=[#6]-2-[#6]=[#6]-[#6]=[#6]-[#6]=[#6]-2)-[#7]-3-[#6](-[#1])-[#1]'),
    ('PAINS', 'ene_misc_D(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](=[#6](-[#6])-[#16]-[#6]-2(-[#1])-[#1])-[#6]'),
    ('PAINS', 'indole_3yl_alk_B(1)', 'c:12:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])c(c(-[#6]:[#6])n2-!@[#6]:[#6])-[#6](-[#1])-[#1]'),
    ('PAINS', 'anil_OH_no_alk_A(1)', '[#7](-[#1])(-[#1])-c:1:c:c:c(:c:c:1-[#8]-[#1])-[#16](=[#8])(=[#8])-[#8]-[#1]'),
    ('PAINS', 'thiazole_amine_L(1)', 's:1:c:c:c(:c:1-[#1])-c:2:c:s:c(:n:2)-[#7](-[#1])-[#1]'),
    ('PAINS', 'pyrazole_amino_A(1)', 'c1c(-[#7](-[#1])-[#1])nnc1-c2c(-[#6](-[#1])-[#1])oc(c2-[#1])-[#1]'),
    ('PAINS', 'het_thio_N_5D(1)', 'n1nscc1-c2nc(no2)-[#6]:[#6]'),
    ('PAINS', 'anil_alk_indane(1)', 'c:1(:c:c-3:c(:c:c:1)-[#7]-[#6]-4-c:2:c:c:c:c:c:2-[#6]-[#6]-3-4)-[#6;X4]'),
    ('PAINS', 'anil_di_alk_N(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#6](-[#1])-[#6]-3-[#6](-[#6]#[#7])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#7]-2-3)-[#1]'),
    ('PAINS', 'het_666_C(1)', 'c:2-3:c(:c:c:1:c:c:c:c:c:1:c:2)-[#7](-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](=[#7]-3)-[#6]:[#6]-[#7](-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'ene_one_D(1)', '[#6](-[#8]-[#1]):[#6]-[#6](=[#8])-[#6](-[#1])=[#6](-[#6])-[#6]'),
    ('PAINS', 'anil_di_alk_indol(1)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]):c(:c(-[#1]):n:2-[#1])-[#16](=[#8])=[#8]'),
    ('PAINS', 'anil_no_alk_indol_A(1)', 'c:1:2:c(:c(:c(:c(:c:1-[#1])-[#1])-[#7](-[#1])-[#1])-[#1]):c(:c(-[#1]):n:2-[#6](-[#1])-[#1])-[#1]'),
    ('PAINS', 'dhp_amino_CN_G(1)', '[#16;X2]-1-[#6]=[#6](-[#6]#[#7])-[#6](-[#6])(-[#6]=[#8])-[#6](=[#6]-1-[#7](-[#1])-[#1])-[$([#6]=[#8]),$([#6]#[#7])]'),
    ('PAINS', 'anil_di_alk_dhp(1)', '[#7]-2-[#6]=[#6](-[#6]=[#8])-[#6](-c:1:c:c:c(:c:c:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#6]~3=[#6]-2~[#7]~[#6](~[#16])~[#7]~[#6]~3~[#7]'),
    ('PAINS', 'anthranil_amide_A(1)', 'c:1:c(:c:c:c:c:1)-[#6](=[#8])-[#7](-[#1])-c:2:c(:c:c:c:c:2)-[#6](=[#8])-[#7](-[#1])-[#7](-[#1])-c:3:n:c:c:s:3'),
    ('PAINS', 'hzone_anthran_Z(1)', 'c:1:c:2:c(:c:c:c:1):c(:c:3:c(:c:2):c:c:c:c:3)-[#6]=[#7]-[#7](-[#1])-c:4:c:c:c:c:c:4'),
    ('PAINS', 'ene_one_amide_A(1)', 'c:1:c(:c:c:c:c:1)-[#6](-[#1])-[#7]-[#6](=[#8])-[#6](-[#7](-[#1])-[#6](-[#1])-[#1])=[#6](-[#1])-[#6](=[#8])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_76_A(1)', 's:1:c(:c(-[#1]):c(:c:1-[#6]-3=[#7]-c:2:c:c:c:c:c:2-[#6](=[#7]-[#7]-3-[#1])-c:4:c:c:n:c:c:4)-[#1])-[#1]'),
    ('PAINS', 'thio_urea_N(1)', 'o:1:c(:c(-[#1]):c(:c:1-[#6](-[#1])(-[#1])-[#7](-[#1])-[#6](=[#16])-[#7](-[#6]-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c:c:c:2)-[#1])-[#1]'),
    ('PAINS', 'anil_di_alk_coum(1)', 'c:1:c(:c:c:c:c:1)-[#7](-[#6]-[#1])-[#6](-[#1])-[#6](-[#1])-[#6](-[#1])-[#7](-[#1])-[#6](=[#8])-[#6]-2=[#6](-[#8]-[#6](-[#6](=[#6]-2-[#6](-[#1])-[#1])-[#1])=[#8])-[#6](-[#1])-[#1]'),
    ('PAINS', 'ene_one_amide_B(1)', 'c2-3:c:c:c:1:c:c:c:c:c:1:c2-[#6](-[#1])-[#6;X4]-[#7]-[#6]-3=[#6](-[#1])-[#6](=[#8])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_thio_656c(1)', 'c:1:c(:c:c:c:c:1)-[#6]-4=[#7]-[#7]:2:[#6](:[#7+]:c:3:c:2:c:c:c:c:3)-[#16]-[#6;X4]-4'),
    ('PAINS', 'het_5_ene(1)', '[#6]-2(=[#8])-[#6](=[#6](-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#7]=[#6](-c:1:c:c:c:c:c:1)-[#8]-2'),
    ('PAINS', 'thio_imide_A(1)', 'c:1:c(:c:c:c:c:1)-[#7]-2-[#6](=[#8])-[#6](=[#6](-[#1])-[#6]-2=[#8])-[#16]-c:3:c:c:c:c:c:3'),
    ('PAINS', 'dhp_amidine_A(1)', '[#7]-1(-[#1])-[#7]=[#6](-[#7]-[#1])-[#16]-[#6](=[#6]-1-[#6]:[#6])-[#6]:[#6]'),
    ('PAINS', 'thio_urea_O(1)', 'c:1(:c(:c-3:c(:c(:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])-c:2:c(:c(:c(:o:2)-[#6]-[#1])-[#1])-[#1])-[#1])-[#8]-[#6](-[#8]-3)(-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_di_alk_O(1)', 'c:1(:c(:c(:c(:c(:c:1-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-c:2:c:c:c:c:c:2)-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_urea_P(1)', '[#8]=[#6]-!@n:1:c:c:c-2:c:1-[#7](-[#1])-[#6](=[#16])-[#7]-2-[#1]'),
    ('PAINS', 'het_pyraz_misc(1)', '[#6](-[F])(-[F])-[#6](=[#8])-[#7](-[#1])-c:1:c(-[#1]):n(-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6](-[#1])(-[#1])-[#6]:[#6]):n:c:1-[#1]'),
    ('PAINS', 'diazox_C(1)', '[#7]-2=[#7]-[#6]:1:[#7]:[!#6&!#1]:[#7]:[#6]:1-[#7]=[#7]-[#6]:[#6]-2'),
    ('PAINS', 'diazox_D(1)', '[#6]-2(-[#1])(-[#8]-[#1])-[#6]:1:[#7]:[!#6&!#1]:[#7]:[#6]:1-[#6](-[#1])(-[#8]-[#1])-[#6]=[#6]-2'),
    ('PAINS', 'misc_cyclopropane(1)', '[#6]-1(-[#6](-[#1])(-[#1])-[#6]-1(-[#1])-[#1])(-[#6](=[#8])-[#7](-[#1])-c:2:c:c:c(:c:c:2)-[#8]-[#6](-[#1])(-[#1])-[#8])-[#16](=[#8])(=[#8])-[#6]:[#6]'),
    ('PAINS', 'imine_ene_one_B(1)', '[#6]-1:[#6]-[#6](=[#8])-[#6]=[#6]-1-[#7]=[#6](-[#1])-[#7](-[#6;X4])-[#6;X4]'),
    ('PAINS', 'coumarin_D(1)', 'c:1:c:c(:c:c-2:c:1-[#6](=[#6](-[#1])-[#6](=[#8])-[#8]-2)-c:3:c:c:c:c:c:3)-[#8]-[#6](-[#1])(-[#1])-[#6]:[#8]:[#6]'),
    ('PAINS', 'misc_furan_A(1)', 'c:1:c(:o:c(:c:1-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#8]-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-c:2:c:c-3:c(:c:c:2)-[#8]-[#6](-[#8]-3)(-[#1])-[#1]'),
    ('PAINS', 'rhod_sat_E(1)', '[#7]-4(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#7](-[#1])-c:2:c:c:c:c:3:c:c:c:c:c:2:3)-[#6]-4=[#8]'),
    ('PAINS', 'rhod_sat_imine_A(1)', '[#7]-3(-[#6](=[#8])-c:1:c:c:c:c:c:1)-[#6](=[#7]-c:2:c:c:c:c:c:2)-[#16]-[#6](-[#1])(-[#1])-[#6]-3=[#8]'),
    ('PAINS', 'rhod_sat_F(1)', '[#7]-2(-c:1:c:c:c:c:c:1)-[#6](=[#8])-[#16]-[#6](-[#1])(-[#1])-[#6]-2=[#16]'),
    ('PAINS', 'het_thio_5_imine_B(1)', '[#7]-1(-[#6](-[#1])-[#1])-[#6](=[#16])-[#7](-[#6]:[#6])-[#6](=[#7]-[#6]:[#6])-[#6]-1=[#7]-[#6]:[#6]'),
    ('PAINS', 'het_thio_5_imine_C(1)', '[#16]-1-[#6](=[#7]-[#7]-[#1])-[#16]-[#6](=[#7]-[#6]:[#6])-[#6]-1=[#7]-[#6]:[#6]'),
    ('PAINS', 'ene_five_het_N(1)', '[#6]-2(=[#8])-[#6](=[#6](-[#1])-c:1:c(:c:c:c(:c:1)-[F,Cl,Br,I])-[#8]-[#6](-[#1])-[#1])-[#7]=[#6](-[#16]-[#6](-[#1])-[#1])-[#16]-2'),
    ('PAINS', 'thio_carbam_A(1)', '[#6](-[#1])(-[#1])-[#16]-[#6](=[#16])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('PAINS', 'misc_anilide_A(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6])-[#1])-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('PAINS', 'misc_anilide_B(1)', 'c:1(:c(:c:c(:c:c:1-[#6])-[Br])-[#6])-[#7](-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]-[#6]-[#6]'),
    ('PAINS', 'mannich_B(1)', 'c:1-2:c(:c:c:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#7](-[#6]:[#6]-[#8]-[#6](-[#1])-[#1])-[#6]-2(-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'mannich_catechol_A(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6]-2(-[#1])-[#1])-[#1])-[#8])-[#8])-[#1]'),
    ('PAINS', 'anil_alk_D(1)', '[#7](-[#1])(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_65_I(1)', 'n:1:2:c:c:c(:c:c:1:c:c(:c:2-[#6](=[#8])-[#6]:[#6])-[#6]:[#6])-[#6](~[#8])~[#8]'),
    ('PAINS', 'misc_urea_A(1)', 'c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#6](=[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1])-[#6](-[#6;X4])(-[#6;X4])-[#7](-[#1])-[#6](=[#8])-[#7](-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]'),
    ('PAINS', 'imidazole_C(1)', '[#6]-3(-[#1])(-n:1:c(:n:c(:c:1-[#1])-[#1])-[#1])-c:2:c(:c(:c(:c(:c:2-[#1])-[Br])-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-c:4:c-3:c(:c(:c(:c:4-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'styrene_imidazole_A(1)', '[#6](=[#6](-[#1])-[#6](-[#1])(-[#1])-n:1:c(:n:c(:c:1-[#1])-[#1])-[#1])(-[#6]:[#6])-[#6]:[#6]'),
    ('PAINS', 'thiazole_amine_M(1)', 'c:1(:n:c(:c(-[#1]):s:1)-c:2:c:c:n:c:c:2)-[#7](-[#1])-[#6]:[#6]-[#6](-[#1])-[#1]'),
    ('PAINS', 'misc_pyrrole_thiaz(1)', 'c:1(:n:c(:c(-[#1]):s:1)-c:2:c:c:c:c:c:2)-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](-[#1])(-[#1])-c:3:c:c:c:n:3-[#1]'),
    ('PAINS', 'pyrrole_L(1)', 'n:1(-[#1]):c(:c(-[#6](-[#1])-[#1]):c(:c:1-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])-[#1])-[#6](=[#8])-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_thio_65_D(1)', 'c:2(:n:c:1:c(:c(:c:c(:c:1-[#1])-[F,Cl,Br,I])-[#1]):n:2-[#1])-[#16]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#7](-[#1])-[#6]:[#6]'),
    ('PAINS', 'ene_misc_E(1)', 'c:1(:c(:c-2:c(:c(:c:1-[#8]-[#6](-[#1])-[#1])-[#1])-[#6]=[#6]-[#6](-[#1])-[#16]-2)-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'thio_cyano_A(1)', '[#7]-1(-[#1])-[#6](=[#16])-[#6](-[#1])(-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6]-1-[#6]:[#6])-[#1]'),
    ('PAINS', 'cyano_amino_het_B(1)', 'n:1:c(:c(:c(:c(:c:1-[#16;X2]-c:2:c:c:c:c:c:2-[#7](-[#1])-[#1])-[#6]#[#7])-c:3:c:c:c:c:c:3)-[#6]#[#7])-[#7](-[#1])-[#1]'),
    ('PAINS', 'cyano_pyridone_G(1)', '[#7]-2(-c:1:c:c:c(:c:c:1)-[#8]-[#6](-[#1])-[#1])-[#6](=[#8])-[#6](=[#6]-[#6](=[#7]-2)-n:3:c:n:c:c:3)-[#6]#[#7]'),
    ('PAINS', 'het_65_J(1)', 'o:1:c(:c:c:2:c:1:c(:c(:c(:c:2-[#1])-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#6](~[#8])~[#8]'),
    ('PAINS', 'ene_one_yne_A(1)', '[#6]#[#6]-[#6](=[#8])-[#6]#[#6]'),
    ('PAINS', 'anil_OH_no_alk_B(1)', 'c:2(:c:1:c(:c(:c(:c(:c:1:c(:c(:c:2-[#8]-[#1])-[#6]=[#8])-[#1])-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#1]'),
    ('PAINS', 'hzone_acyl_misc_A(1)', 'c:1(:c(:c(:c(:o:1)-[$([#1]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6](-[$([#1]),$([#6](-[#1])-[#1])])-c:2:c:c:c:c(:c:2)-[*]-[*]-[*]-c:3:c:c:c:o:3'),
    ('PAINS', 'thiophene_F(1)', '[#16](=[#8])(=[#8])-[#7](-[#1])-c:1:c(:c(:c(:s:1)-[#6]-[#1])-[#6]-[#1])-[#6](=[#8])-[#7]-[#1]'),
    ('PAINS', 'anil_OC_alk_E(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#8]-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'anil_OC_alk_F(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-[#6](-[#1])(-[#6]=[#8])-[#16]'),
    ('PAINS', 'het_65_K(1)', 'n1nnnc2cccc12'),
    ('PAINS', 'het_65_L(1)', 'c:1-2:c(-[#1]):s:c(:c:1-[#6](=[#8])-[#7]-[#7]=[#6]-2-[#7](-[#1])-[#1])-[#6]=[#8]'),
    ('PAINS', 'coumarin_E(1)', 'c:1-3:c(:c:2:c(:c:c:1-[Br]):o:c:c:2)-[#6](=[#6]-[#6](=[#8])-[#8]-3)-[#1]'),
    ('PAINS', 'coumarin_F(1)', 'c:1-3:c(:c:c:c:c:1)-[#6](=[#6](-[#6](=[#8])-[#7](-[#1])-c:2:n:o:c:c:2-[Br])-[#6](=[#8])-[#8]-3)-[#1]'),
    ('PAINS', 'coumarin_G(1)', 'c:1-2:c(:c:c(:c:c:1-[F,Cl,Br,I])-[F,Cl,Br,I])-[#6](=[#6](-[#6](=[#8])-[#7](-[#1])-[#1])-[#6](=[#7]-[#1])-[#8]-2)-[#1]'),
    ('PAINS', 'coumarin_H(1)', 'c:1-3:c(:c:c:c:c:1)-[#6](=[#6](-[#6](=[#8])-[#7](-[#1])-c:2:n:c(:c:s:2)-[#6]:[#16]:[#6]-[#1])-[#6](=[#8])-[#8]-3)-[#1]'),
    ('PAINS', 'het_thio_67_A(1)', '[#6](-[#1])(-[#1])-[#16;X2]-c:2:n:n:c:1-[#6]:[#6]-[#7]=[#6]-[#8]-c:1:n:2'),
    ('PAINS', 'sulfonamide_I(1)', '[#16](=[#8])(=[#8])(-c:1:c:n(-[#6](-[#1])-[#1]):c:n:1)-[#7](-[#1])-c:2:c:n(:n:c:2)-[#6](-[#1])(-[#1])-[#6]:[#6]-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_65_mannich(1)', 'c:1-2:c(:c(:c(:c(:c:1-[#8]-[#6](-[#1])(-[#1])-[#8]-2)-[#6](-[#1])(-[#1])-[#7]-3-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6]:[#6]-3)-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_alk_A(1)', '[#6](-[#1])(-[#1])-[#8]-[#6]:[#6]-[#6](-[#1])(-[#1])-[#7](-[#1])-c:2:c(:c(:c:1:n(:c(:n:c:1:c:2-[#1])-[#1])-[#6]-[#1])-[#1])-[#1]'),
    ('PAINS', 'het_5_inium(1)', '[#7]-4(-c:1:c:c:c:c:c:1)-[#6](=[#7+](-c:2:c:c:c:c:c:2)-[#6](=[#7]-c:3:c:c:c:c:c:3)-[#7]-4)-[#1]'),
    ('PAINS', 'anil_di_alk_P(1)', '[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])-[#1])-c:2:c:c:c:1:s:c(:n:c:1:c:2)-[#16]-[#6](-[#1])-[#1]'),
    ('PAINS', 'thio_urea_Q(1)', 'c:1:2:c(:c(:c(:c(:c:1:c(:c(-[#1]):c(:c:2-[#1])-[#1])-[#6](-[#6](-[#1])-[#1])=[#7]-[#7](-[#1])-[#6](=[#16])-[#7](-[#1])-[#6]:[#6]:[#6])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_pyridine_A(1)', '[#6]:1(:[#7]:[#6](:[#7]:[!#1]:[#7]:1)-c:2:c(:c(:c(:o:2)-[#1])-[#1])-[#1])-[#16]-[#6;X4]'),
    ('PAINS', 'melamine_B(1)', 'n:1:c(:n:c(:n:c:1-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#7](-[#6]-[#1])-[#6]=[#8]'),
    ('PAINS', 'misc_phthal_thio_N(1)', 'c:1(:n:s:c(:n:1)-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#7]-[#6](=[#8])-c:2:c:c:c:c:c:2-[#6](=[#8])-[#8]-[#1])-c:3:c:c:c:c:c:3'),
    ('PAINS', 'hzone_acyl_misc_B(1)', 'n:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#7]=[#6](-[#1])-c:2:c:c:c:c:c:2-[#8]-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]-[#1]'),
    ('PAINS', 'tert_butyl_B(1)', '[#6](-[#1])(-[#1])(-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-c:1:c(:c(:c(:c(:c:1-[#8]-[#1])-[#6](-[#6](-[#1])(-[#1])-[#1])(-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1])-[#1])-[#6](-[#1])(-[#1])-c:2:c:c:c(:c(:c:2-[#1])-[#1])-[#8]-[#1])-[#1]'),
    ('PAINS', 'diazox_E(1)', '[#7](-[#1])(-[#1])-c:1:c(-[#7](-[#1])-[#1]):c(:c(-[#1]):c:2:n:o:n:c:1:2)-[#1]'),
    ('PAINS', 'anil_NH_no_alk_B(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#7](-[#1])-[#16](=[#8])=[#8])-[#1])-[#7](-[#1])-[#6](-[#1])-[#1])-[F,Cl,Br,I])-[#1]'),
    ('PAINS', 'anil_no_alk_A(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#7]=[#6]-2-[#6](=[#6]~[#6]~[#6]=[#6]-2)-[#1])-[#1])-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_no_alk_B(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-n:2:c:c:c:c:2)-[#1])-[#6](-[#1])-[#1])-[#6](-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_ene_amine_A(1)', '[#16]=[#6]-[#6](-[#6](-[#1])-[#1])=[#6](-[#6](-[#1])-[#1])-[#7](-[#6](-[#1])-[#1])-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_55_B(1)', '[#6]-1:[#6]-[#8]-[#6]-2-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]-[#6]-1-2'),
    ('PAINS', 'cyanamide_A(1)', '[#8]-[#6](=[#8])-[#6](-[#1])(-[#1])-[#16;X2]-[#6](=[#7]-[#6]#[#7])-[#7](-[#1])-c:1:c:c:c:c:c:1'),
    ('PAINS', 'ene_one_one_A(1)', '[#8]=[#6]-[#6]-1=[#6](-[#16]-[#6](=[#6](-[#1])-[#6])-[#16]-1)-[#6]=[#8]'),
    ('PAINS', 'ene_six_het_D(1)', '[#8]=[#6]-1-[#7]-[#7]-[#6](=[#7]-[#6]-1=[#6]-[#1])-[!#1]:[!#1]'),
    ('PAINS', 'ene_cyano_E(1)', '[#8]=[#6]-[#6](-[#1])=[#6](-[#6]#[#7])-[#6]'),
    ('PAINS', 'ene_cyano_F(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c(:c(:c(:c(:c:1-[#8]-[#1])-[#1])-c:2:c(-[#1]):c(:c(:o:2)-[#6](-[#1])=[#6](-[#6]#[#7])-c:3:n:c:c:n:3)-[#1])-[#1])-[#1]'),
    ('PAINS', 'hzone_furan_C(1)', 'c:1:c(:c:c:c:c:1)-[#7](-c:2:c:c:c:c:c:2)-[#7]=[#6](-[#1])-[#6]:3:[#6](:[#6](:[#6](:[!#1]:3)-c:4:c:c:c:c(:c:4)-[#6](=[#8])-[#8]-[#1])-[#1])-[#1]'),
    ('PAINS', 'anil_no_alk_C(1)', '[#7](-[#1])(-[#1])-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-c:2:c(-[#1]):c(:c(-[#6](-[#1])-[#1]):o:2)-[#6]=[#8])-[#1])-[#1]'),
    ('PAINS', 'hzone_acid_D(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c:c(:c:c:1)-[#7]-[#7]=[#6](-[#1])-[#6]:2:[#6](:[#6](:[#6](:[!#1]:2)-c:3:c:c:c:c:c:3)-[#1])-[#1]'),
    ('PAINS', 'hzone_furan_E(1)', '[#8](-[#1])-[#6](=[#8])-c:1:c:c:c:c(:c:1)-[#6]:[!#1]:[#6]-[#6]=[#7]-[#7](-[#1])-[#6](=[#8])-[#6](-[#1])(-[#1])-[#8]'),
    ('PAINS', 'het_6_pyridone_NH2(1)', '[#8](-[#1])-[#6]:1:[#6](:[#6]:[!#1]:[#6](:[#7]:1)-[#7](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#8]'),
    ('PAINS', 'imine_one_fives_D(1)', '[#6]-1(=[!#6&!#1])-[#6](-[#7]=[#6]-[#16]-1)=[#8]'),
    ('PAINS', 'pyrrole_M(1)', 'n2(-c:1:c:c:c:c:c:1)c(c(-[#1])c(c2-[#6]=[#7]-[#8]-[#1])-[#1])-[#1]'),
    ('PAINS', 'pyrrole_N(1)', 'n2(-[#6](-[#1])-c:1:c(:c(:c:c(:c:1-[#1])-[#1])-[#1])-[#1])c(c(-[#1])c(c2-[#6]-[#1])-[#1])-[#6]-[#1]'),
    ('PAINS', 'pyrrole_O(1)', 'n1(-[#6](-[#1])-[#1])c(c(-[#6](=[#8])-[#6])c(c1-[#6]:[#6])-[#6])-[#6](-[#1])-[#1]'),
    ('PAINS', 'ene_cyano_G(1)', 'n1(-[#6])c(c(-[#1])c(c1-[#6](-[#1])=[#6](-[#6]#[#7])-c:2:n:c:c:s:2)-[#1])-[#1]'),
    ('PAINS', 'sulfonamide_J(1)', 'n3(-c:1:c:c:c:c:c:1-[#7](-[#1])-[#16](=[#8])(=[#8])-c:2:c:c:c:s:2)c(c(-[#1])c(c3-[#1])-[#1])-[#1]'),
    ('PAINS', 'misc_pyrrole_benz(1)', 'n2(-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#6](=[#8])-[#7](-[#1])-[#6](-[#1])(-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#8]-[#6]:[#6])c(c(-[#1])c(c2-[#1])-[#1])-[#1]'),
    ('PAINS', 'thio_urea_R(1)', 'c:1(:c:c:c:c:c:1)-[#7](-[#1])-[#6](=[#16])-[#7]-[#7](-[#1])-[#6](-[#1])=[#6](-[#1])-[#6]=[#8]'),
    ('PAINS', 'ene_one_one_B(1)', '[#6]-1(-[#6](=[#8])-[#6](-[#1])(-[#1])-[#6]-[#6](-[#1])(-[#1])-[#6]-1=[#8])=[#6](-[#7]-[#1])-[#6]=[#8]'),
    ('PAINS', 'dhp_amino_CN_H(1)', '[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#16]-[#6;X4]-[#16]-1'),
    ('PAINS', 'het_66_anisole(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#1])-[#1])-[#7](-[#1])-c:2:c:c:n:c:3:c(:c:c:c(:c:2:3)-[#8]-[#6](-[#1])-[#1])-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'thiazole_amine_N(1)', '[#6](-[#1])(-[#1])-[#8]-c:1:c(:c(:c(:c(:c:1-[#1])-[#1])-[#8]-[#6](-[#1])-[#1])-[#1])-[#7](-[#1])-c:2:n:c(:c:s:2)-c:3:c:c:c(:c:c:3)-[#8]-[#6](-[#1])-[#1]'),
    ('PAINS', 'het_pyridiniums_C(1)', '[#6]~1~3~[#7](-[#6]:[#6])~[#6]~[#6]~[#6]~[#6]~1~[#6]~2~[#7]~[#6]~[#6]~[#6]~[#7+]~2~[#7]~3'),
    ('PAINS', 'het_5_E(1)', '[#7]-3(-c:2:c:1:c:c:c:c:c:1:c:c:c:2)-[#7]=[#6](-[#6](-[#1])-[#1])-[#6](-[#1])(-[#1])-[#6]-3=[#8]'),
    ('SureChEMBL', '2,2-dimethyl-4,5-dicarboxy-dithiole', 'C1(C)(C)SC(C(=O)O)=C(C(=O)O)S1'),
    ('SureChEMBL', '2,3,4_trihydroxyphenyl', 'c([OH])c([OH])c([OH])'),
    ('SureChEMBL', '2,3,5_trihydroxyphenyl', 'c([OH])c([OH])cc([OH])'),
    ('SureChEMBL', 'acid_anhydrides', 'C(=O)OC(=O)'),
    ('SureChEMBL', 'acid_halides', '[S,C](=[O,S])[F,Br,Cl,I]'),
    ('SureChEMBL', 'Acridine', 'c1c2cc4ccccc4nc2ccc1'),
    ('SureChEMBL', 'Active_Phosphate', 'P(=S)([OH1,O$(O[#6])])([OH1,O$(O[#6])])[S,O]'),
    ('SureChEMBL', 'acyl_cyanide', 'C(=O)-C#N'),
    ('SureChEMBL', 'Adjacent_Ring_Double_Bonds', '[*;R]=[*;R]=[*;R]'),
    ('SureChEMBL', 'Aldehyde', '[#6][C!H0]=O'),
    ('SureChEMBL', 'Aliphatic_Triflate', 'COS(=O)(=O)C(F)(F)F'),
    ('SureChEMBL', 'alkyl_halides', '[Br,Cl,I][CX4;CH,CH2]'),
    ('SureChEMBL', 'AlkylEnamine', '[C;H1$(C([#6;!$(C=O)])),H0$(C([#6;!$(C=O)])[#6;!$(C=O)])]=[CH1]!@N([#6;!$(C(=O))])[#6;!$(C(=O))]'),
    ('SureChEMBL', 'Allene', '*=C=*'),
    ('SureChEMBL', 'Alpha_Halo_Carbonyl', '[C;!$(C[N])](=O)!@[C;h1,h2;H1,H2][F,Cl,Br,I]'),
    ('SureChEMBL', 'amidotetrazole', 'c1nnnn1C=O'),
    ('SureChEMBL', 'Amino_Naphtalimide', 'c1(N)ccc(C(=O)NC3(=O))c(c3ccc2)c21'),
    ('SureChEMBL', 'Aminonitrile', 'NC#N'),
    ('SureChEMBL', 'Anhydride', '[#6]C(=O)OC(=O)[#6]'),
    ('SureChEMBL', 'Any_Carbazide', 'O=*N=[N+]=[N-]'),
    ('SureChEMBL', 'aromatic_azides', 'cN=N=N'),
    ('SureChEMBL', 'Azanitrone', 'N=[N+]([O-])C'),
    ('SureChEMBL', 'azoalkanals', '[N;R0]=[N;R0]CC=O'),
    ('SureChEMBL', 'Azobenzene', 'c1ccccc1[N!r]=[N!r]c2ccccc2'),
    ('SureChEMBL', 'Azocyanamide', '[N;R0]=[N;R0]C#N'),
    ('SureChEMBL', 'b-Carbonyl_Quaternary_Nitrogen', 'C(=O)CC[N+,n+]'),
    ('SureChEMBL', 'benzylic_quaternary_nitrogen', 'cC[N+,NX4]'),
    ('SureChEMBL', 'beta-carbonyl_quaternary_nitrogen', 'C(=O)C[N+,n+,NX4,nX4]'),
    ('SureChEMBL', 'Beta-Fluoro-ethyl-ON', '[C;H2$(CF),H1$(C(F)F)]!@[CH2][N,O]'),
    ('SureChEMBL', 'biotin_analogue', 'C12C(NC(N1)=O)CSC2'),
    ('SureChEMBL', 'carbazides', 'C(=O)N=N=N'),
    ('SureChEMBL', 'carbodiimides', 'N=C=N'),
    ('SureChEMBL', 'CCl3-CHO_releasing', 'C(Cl)(Cl)(Cl)C([O,S])[NX3]'),
    ('SureChEMBL', 'Chloramidine', '[Cl]C([C&R0])=N'),
    ('SureChEMBL', 'Conjugated_Dithioether', 'SC(=[!r])S'),
    ('SureChEMBL', 'Crown_Ether_12CRO4', 'O1CCOCCOCCOCC1'),
    ('SureChEMBL', 'Crown_Ether_15CRO5', 'O1CCOCCOCCOCCOCC1'),
    ('SureChEMBL', 'Crown_Ether_16CRO6', 'O1CCOCCOCCOCCOCCOCC1'),
    ('SureChEMBL', 'crown_ethers', '[O;R1][C;R1][C;R1][O;R1][C;R1][C;R1][O;R1]'),
    ('SureChEMBL', 'cyanamide', 'N[CH2]C#N'),
    ('SureChEMBL', 'Cyanophosphonate', 'P(OCC)(OCC)(=O)C#N'),
    ('SureChEMBL', 'Cyanohydrin', 'N#CC[OH1]'),
    ('SureChEMBL', 'di_and_triphosphates', 'P(=O)([OH])OP(=O)[OH]'),
    ('SureChEMBL', 'Diacetylene', 'C#CC#C'),
    ('SureChEMBL', 'Diazoalkane', 'C=[N+]=[N-]'),
    ('SureChEMBL', 'Diazonium_Salt', '[N+]#N'),
    ('SureChEMBL', 'Diene', 'C!@=[CH1]-C!@=[CH1]-[CX3](=O)'),
    ('SureChEMBL', 'Dinitrobenzene_1', 'c1c([N+](=O)[O-])c([N+](=O)[O-])ccc1'),
    ('SureChEMBL', 'Dinitrobenzene_2', 'c1c([N+](=O)[O-])ccc([N+](=O)[O-])c1'),
    ('SureChEMBL', 'Dinitrobenzene_3', 'c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1'),
    ('SureChEMBL', 'disulfides', '[SX2][SX2]'),
    ('SureChEMBL', 'Dithiocarbamate', 'NC(=S)S'),
    ('SureChEMBL', 'Dithiole-2-thione', 'S1SC=CC1=S'),
    ('SureChEMBL', 'Dithiole-3-thione', 'S1C=CSC1=C'),
    ('SureChEMBL', 'Dithiomethylene_acetal', 'S[C;!$(C=*)]S'),
    ('SureChEMBL', 'Enyne', 'C=!@CC#C'),
    ('SureChEMBL', 'epoxides,_thioepoxides,_aziridines', 'C1[O,S,N]C1'),
    ('SureChEMBL', 'ester_of_HOBT', 'C(=O)Onnn'),
    ('SureChEMBL', 'Flavin', 'c1cccc(NC(=NC(=[N,S,O])NC(=O)3)C3=N2)c12'),
    ('SureChEMBL', 'Fluorescein', 'c1cc(O)cc(OC(=CC(=O)C=C3)C3=C2)c12'),
    ('SureChEMBL', 'Fluorinated_Carbon_1', 'C(C(CF)F)F'),
    ('SureChEMBL', 'Fluorinated_Carbon_2', 'C(C(F)F)(F)F'),
    ('SureChEMBL', 'four_member_lactones', 'C1(=O)OCC1'),
    ('SureChEMBL', 'geminal_amines', '[NH1;!r][CX4][NH1;!r]'),
    ('SureChEMBL', 'geminal_dinitriles', 'N#CCC#N'),
    ('SureChEMBL', 'halo-pyridine,_-diazoles_and_-triazoles', '[Cl,Br,I]c1[c,n][c,n][c,n][c,n]n1'),
    ('SureChEMBL', 'hydrazothiourea', 'N=NC(S)N'),
    ('SureChEMBL', 'Imidazolium', 'c1[n+]([#6])ccn1([#6])'),
    ('SureChEMBL', 'Imine2', '[#6,#8,#16]-[CH1]=[NH1]'),
    ('SureChEMBL', 'imines_(not_ring)', '[#6][C;R0](=[N;R0,O0])[#6]'),
    ('SureChEMBL', 'isocyanates_and_isothiocyanates', 'N=C=[S,O]'),
    ('SureChEMBL', 'isonitrile', '[N+]#[C-]'),
    ('SureChEMBL', 'ketene', 'C=C=O'),
    ('SureChEMBL', 'Lawesson_Reagent_Derivatives', 'P(=S)(S)S'),
    ('SureChEMBL', 'methylidene-1,3-dithiole', 'S1C=CSC1=S'),
    ('SureChEMBL', 'Michael_Phenyl_Ketone', 'c1ccccc1C(=O)C=!@CC(=O)!@*'),
    ('SureChEMBL', 'N-halo', '[NX3,NX4][F,Cl,Br,I]'),
    ('SureChEMBL', 'Nitrobenz-azadiazole_1', 'c1ccc(n[o,s]n2)c2c1[N+](=O)[O-]'),
    ('SureChEMBL', 'Nitrobenz-azadiazole_2', 'c1c([N+](=O)[O-])cc(n[o,s]n2)c2c1'),
    ('SureChEMBL', 'nitrosamine', 'N-[N;X2](=O)'),
    ('SureChEMBL', 'nitroso', '[N&D2](=O)'),
    ('SureChEMBL', 'noname', 'N1=C[S,NH1]C(=[C,N,P][C,N,O,P])C1(=O)'),
    ('SureChEMBL', 'N-Oxide_aliphatic', '[N+!$(N=O)][O-X1]'),
    ('SureChEMBL', 'N-S_(not_sulfonamides)', '[#6][S!$(S(~[OD1])~[OD1])][N;H0]'),
    ('SureChEMBL', 'Orthoester', 'C(O)(O)[OH]'),
    ('SureChEMBL', 'o-tertbutylphenol', 'c1c([OH1])c(C(C)(C)C)ccc1'),
    ('SureChEMBL', 'Oxobenzothiepine', 'C1(=O)C=CCSC=C1'),
    ('SureChEMBL', 'P_or_S_Halides', '[P,S][Cl,Br,F,I]'),
    ('SureChEMBL', 'p-Aminoaryl_diazo', 'Nc1aaa(N!@=N)aa1'),
    ('SureChEMBL', 'PCP', 'PCP'),
    ('SureChEMBL', 'paranitrophenyl_esters', 'C(=O)Oc1ccc([N+](=O)[O-])cc1'),
    ('SureChEMBL', 'pentahalophenyl', 'c1c([F,Cl])c([F,Cl])c([F,Cl])c([F,Cl])c1([F,Cl])'),
    ('SureChEMBL', 'pentafluorophenyl_esters', 'C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)'),
    ('SureChEMBL', 'peroxide', '[#8]~[#8]'),
    ('SureChEMBL', 'Phenanthrene', 'c12cccc3c1c4c(cc3)cccc4cc2'),
    ('SureChEMBL', 'Phenylester', 'C(=O)!@Oc1ccccc1'),
    ('SureChEMBL', 'phosphonate_esters', '[#6]P(=O)(~O)O[#6]'),
    ('SureChEMBL', 'phosphoramides', 'NP(=O)(N)N'),
    ('SureChEMBL', 'phosphorane', 'C=P'),
    ('SureChEMBL', 'Phosphorus_Halide', '[S,P][F,Cl,Br,I]'),
    ('SureChEMBL', 'Polyene', 'C=!@CC=!@C'),
    ('SureChEMBL', 'polyenes', 'C=CC=CC=CC=C'),
    ('SureChEMBL', 'polyene_chain_between_aromatics', 'cC=CC=CC=Cc'),
    ('SureChEMBL', 'polyines', 'CC#CC#CC'),
    ('SureChEMBL', 'Polynuclear_Aromatic_1', 'c1cccc(cc(cccc2)c2c3)c13'),
    ('SureChEMBL', 'Polynuclear_Aromatic_2', 'c1cccc(c(cccc2)c2cc3)c13'),
    ('SureChEMBL', 'Polysulfide', '*[SX2][SX2][SX2]*'),
    ('SureChEMBL', 'Sulphur_Halide', '[#16][F,Cl,Br,I]'),
    ('SureChEMBL', 'pyrene_fragments', 'c1c2cccc3c2c4c(cc3)cccc4c1'),
    ('SureChEMBL', 'Pyrylium', 'c1ccc[o+]c1'),
    ('SureChEMBL', 'reactive_carbonyls', '[C;!r](=[O,S])[S;!r]'),
    ('SureChEMBL', 'reactive_carbonyls', '[C;!r](=[O,S])[CD2;!r][F,Br,Cl]'),
    ('SureChEMBL', 'Ring_Triple_Bond', '[C,c;R]#[C,c;R]'),
    ('SureChEMBL', 'S=N_(not_ring)', '[S;R0]=[N;R0]'),
    ('SureChEMBL', 'Sulfonate_Ester', 'O=[SX4](=O)OC'),
    ('SureChEMBL', 'sulfonyl_cyanide', 'S(=O)(=O)C#N'),
    ('SureChEMBL', 'Sulphate_Ester', 'COS(=O)O[C,c]'),
    ('SureChEMBL', 'sulphonates', 'COS(=O)(=O)[C,c]'),
    ('SureChEMBL', 'Sulphur_Nitrogen_single_bond', '[SX2H0]!@[N]'),
    ('SureChEMBL', 'Tetraazinane', 'C1NNC=NN1'),
    ('SureChEMBL', 'Thiocyanate', 'SC#N'),
    ('SureChEMBL', 'thioesters', 'C[O,S;R0][C;R0](=S)'),
    ('SureChEMBL', 'thioles_(not_aromatic)', '[!a][SX2;H1]'),
    ('SureChEMBL', 'Thiophosphothionate', 'P(=S)(-[S;H1,H0$(S(P)C)])(-[O;H1,H0$(O(P)C)])(-N(C)C)'),
    ('SureChEMBL', 'thiourea', '[N;!r][C;!r](=S)[N;!r]'),
    ('SureChEMBL', 'Three_Membered_Heterocycle', '*1[O,S]*1'),
    ('SureChEMBL', 'Tri_Pentavalent_S', '[#16v3,#16v5]'),
    ('SureChEMBL', 'Triacyloxime', 'C(=O)N(C(=O))OC(=O)'),
    ('SureChEMBL', 'Triazole', 'c1cnnn1!@C!@[NH1][#6]'),
    ('SureChEMBL', 'triflate', 'OS(=O)(=O)(C(F)(F)(F))'),
    ('SureChEMBL', 'Triphenyl_Boranyl', 'B(c1ccccc1)(c2ccccc2)c3ccccc3'),
    ('SureChEMBL', 'triphenylphosphines', 'P(c1aaaaa1)(c1aaaaa1)(c1aaaaa1)'),
    ('SureChEMBL', 'Triphenyl_Silyl', '[Si](c1ccccc1)(c2ccccc2)(c3ccccc3)'),
    ('SureChEMBL', 'Vinyl_Halide', '[Cl,Br,I]C=[!O!R]'),
    ('SureChEMBL', 'Vinyl_Sulphone', '[#6][CH1]!@=[CH1][S;H1,H0$(S(C)C)](=O)(=O)'),
    ('SureChEMBL', 'sulphates', '[#6]S(=O)(=O)O'),
    ('SureChEMBL', 'tropone', 'C1C(=O)C=CC=CC=1'),
    ('SureChEMBL', 'Oxime', '[#6]C(=!@N[$(OC),$([OH])])[#6]'),
    ('SureChEMBL', 'hydrazone', '[#6]C(=!@NNa)[#6]'),
    ('SureChEMBL', 'Nitrosone_not_nitro', '[$(N(~!@[#6])!@O);!$([N+]([O-])=O)]'),
    ('SureChEMBL', 'Thiocarbonyl_group', 'C=S'),
    ('SureChEMBL', 'enamine_like', 'C=[NH]'),
    ('SureChEMBL', 'analine', 'c1ccccc1[NH2,NH3+]'),
    ('SureChEMBL', 'acid_anhydrides_2', '[#6]C(=O)!@OC(=!@[N,O])[#6]'),
    ('SureChEMBL', 'trifluroacetate_amide', 'FC(F)(F)C(=O)N'),
    ('SureChEMBL', 'triple_bond', '[#6]C#[CH]'),
    ('SureChEMBL', 'Allene', 'C=C=C'),
    ('SureChEMBL', 'thiatetrazolidine', '[$(Sc1nnn[nH,n-]1),$(Sc1nn[nH,n-]n1)]'),
    ('SureChEMBL', 'glycol', '[#6][O;R0][$(C([#6])[#6]),$([CH][#6]),$([CH2])][O;R0][#6]'),
    ('SureChEMBL', 'oxy-amide', '[#6]C(=O)!@C!@C(=O)N'),
    ('SureChEMBL', 'formate_formide', 'O=[CH][O,N][#6]'),
    ('SureChEMBL', 'pyranone', 'O=C1C=COC=C1'),
    ('SureChEMBL', 'Coumarin', 'c1cc2C=CC(=O)Oc2cc1'),
    ('SureChEMBL', 'aminothiazole', 's1ccnc1[N!H0]'),
    ('SureChEMBL', 'Thiazolidinone', 'O=C1CSCN1'),
    ('SureChEMBL', 'Thiomorpholinedione', 'N1C(=O)CSCC1=O'),
    ('SureChEMBL', 'oxepine', 'O1C=CC=CC=C1'),
    ('SureChEMBL', 'phenylethene', 'c1ccccc1!@[CH]=!@[C!H0]'),
    ('SureChEMBL', 'Ethene', 'C=[CH2]'),
    ('SureChEMBL', 'cyclobutene', 'C1CC=C1'),
    ('SureChEMBL', 'poly_sub_atomatic', '*!@c1c(!@*)c(!@*)c(!@*)c(!@*)c1'),
    ('SureChEMBL', 'Isotopes', '[2#1,3#1,11C,11c,14C,14c,125I,32P,33P,35S]'),
    ('SureChEMBL', 'Undesirable_Elements_Salts', '[Ac,Ag,Am,Ar,As,At,Au,Ba,Be,Bi,Bk,Cd,Ce,Cf,Cm,Cr,Cs,Dy,Er,Eu,Fr,Ga,Gd,Ge,He,Hf,Ho,In,Ir,Kr,La,Lu,Mo,Nb,Nd,Ne,Ni,Np,Os,Pa,Pb,Pd,Pm,Po,Pr,Pt,Pu,Ra,Rb,Re,Rh,Rn,Ru,Sb,Sc,Se,Sm,Sr,Ta,Tb,Tc,Te,Th,Ti,Tl,Tm,U,V,W,Xe,Y,Yb,Zr]'),
    ('SureChEMBL', 'Metal_Carbon_bond', '[#6;$([#6]~[#3,#11,#12,#13,#19,#20,#26,#27,#28,#29,#30])]'),
    ('SureChEMBL', 'Aromatic_N-Oxide_more_than_one', '[n+][O-X1].[n+][O-X1]'),
    ('SureChEMBL', 'Nitro_more_than_one', '[N+](=O)[O-].[N+](=O)[O-]'),
    ('MLSMR', 'anhydride', 'C(=O)OC(=O)'),
    ('MLSMR', 'pentafluorophenyl ester', 'C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)'),
    ('MLSMR', 'p-nitrophenyl ester', 'C(=O)Oc1ccc([N+]([O-])=O)cc1'),
    ('MLSMR', 'any carbazide', 'O=*N=[N+]=[N-]'),
    ('MLSMR', 'HOBT ester', 'C(=O)Onnn'),
    ('MLSMR', 'aromatic azide', 'cN=[N+]=[N-]'),
    ('MLSMR', 'imine2', '[#6,#8,#16]-[CH1]=[NH1]'),
    ('MLSMR', 'sulfonyl cyanide', 'S(=O)(=O)C#N'),
    ('MLSMR', 'azocyanamide', '[N;R0]=[N;R0]C#N'),
    ('MLSMR', 'cyanohydrin', 'N#CC[OH]'),
    ('MLSMR', 'acyl cyanide', 'N#CC(=O)'),
    ('MLSMR', 'acid halide', '[S,C](=[O,S])[F,Br,Cl,I]'),
    ('MLSMR', 'chloramidine', '[Cl]C([C&R0])=N'),
    ('MLSMR', 'P/S halide', '[P,S][F,Cl,Br,I]'),
    ('MLSMR', 'quaternary', '[C+,Cl+,I+,P+,S+]'),
    ('MLSMR', 'unacceptable atoms', '[!#6;!#7;!#8;!#16;!#1;!#3;!#9;!#11;!#12;!#15;!#17;!#19;!#20;!#30;!#35]'),
    ('MLSMR', 'triacyloxime', 'C(=O)N(C(=O))OC(=O)'),
    ('MLSMR', 'b-carbonyl quaternary nitrogen', 'C(=O)CC[N+,n+]'),
    ('MLSMR', 'benzylic quaternary nitrogen', 'cC[N+]'),
    ('MLSMR', 'phosphorane', 'C=P'),
    ('MLSMR', 'Lawesson reagent derivatives', 'P(=S)(S)S'),
    ('MLSMR', 'cyanophosphonate', 'P(OCC)(OCC)(=O)C#N'),
    ('MLSMR', 'sulfonate', 'COS(=O)(=O)[C,c]'),
    ('MLSMR', 'Heteroaryl sulfonate', 'a-S(=O)(=O)-O[$([a&!#6]),$(c[a&!#6]),$(cc[a&!#6]),$(ccc[a&!#6]),$(cccc[a&!#6]),$(ccccc[a&!#6])]'),
    ('MLSMR', 'sulfate ester', 'COS(=O)O[C,c]'),
    ('MLSMR', 'triflate', 'OS(=O)(=O)C(F)(F)F'),
    ('MLSMR', 'polyacidic', '[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH]'),
    ('MLSMR', 'Sulfonic acid', '[OH]-S(=O)(=O)-*'),
    ('MLSMR', 'thiol', '[SH]'),
    ('MLSMR', 'benzhydrol', '[OH1]-C(-c1ccccc1)c2ccccc2'),
    ('MLSMR', 'dihydroxybenzene', '[OH1]c1ccc([OH1])cc1'),
    ('MLSMR', '2,3,4trihydroxyphenyl', 'c([OH])c([OH])c([OH])'),
    ('MLSMR', '2,4,5trihydroxyphenyl', 'c([OH])c([OH])cc([OH])'),
    ('MLSMR', 'allene', '*=C=*'),
    ('MLSMR', 'Azide', 'N=N=N'),
    ('MLSMR', 'azoalkanal', '[N;R0]=[N;R0]CC=O'),
    ('MLSMR', 'hydrazothiourea', 'N=NC(=S)N'),
    ('MLSMR', 'Azo', 'N=N'),
    ('MLSMR', 'aldehyde', '[#6]-[CH1]=O'),
    ('MLSMR', 'hemiacetal', '[#6]-O[CH1](-[#6])[OH1]'),
    ('MLSMR', 'acetal', '[#6]-O[CH1](-[#6])O-[#6]'),
    ('MLSMR', 'Ketone', '[#6]-C(=O)-[#6]'),
    ('MLSMR', 'Ester', '[#6]-C(=O)O-[#6]'),
    ('MLSMR', 'imine 1', '[#6,#8,#16]-C(=[NH1])[#6,#8,#16]'),
    ('MLSMR', 'Imine 3', 'C=[NH]'),
    ('MLSMR', 'thioketone', 'CC(=S)C'),
    ('MLSMR', 'thioester', 'C[O,S;R0][C;R0](=S)'),
    ('MLSMR', 'thionoester', 'COC(=S)C'),
    ('MLSMR', 'thioamide', 'CC(=S)N'),
    ('MLSMR', 'thiourea', 'NC(=S)N'),
    ('MLSMR', 'nitroso', '[N&D2](=O)'),
    ('MLSMR', 'long chain hydrocarbon', '[CD2;R0][CD2;R0][CD2;R0][CD2;R0][CD2;R0][CD2;R0]'),
    ('MLSMR', 'Long aliphatic chain', '[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]'),
    ('MLSMR', 'Unbranched chain', '[$([A&D2]),$([A&D1])]!@[A&D2]!@[A&D2]!@[A&D2]!@[A&D2]!@[$([A&D2]),$([A&D1])]'),
    ('MLSMR', 'polyene', 'C=CC=CC=CC=C'),
    ('MLSMR', 'Dye 25', 'acC=&!@Cca'),
    ('MLSMR', 'isonitrile', '[N+]#[C-]'),
    ('MLSMR', 'thiocyanate', 'SC#N'),
    ('MLSMR', 'cyanamide', 'N[CH2]C#N'),
    ('MLSMR', 'Dye 16 (1)', 'c([N+](=O)[O-])'),
    ('MLSMR', 'nitro aromatic 2+', 'a-[N+](=O)[O-].a-[N+](=O)[O-]'),
    ('MLSMR', 'Dye 29', 'O=[N+](-[O-])-caac-[$(N(C)C),$([NH]C),$([NH2])]'),
    ('MLSMR', 'Dye 1 (1)', 'c1cccc(C(=O)[C,c]([#7])=,:[C,c]([#7])C2(=O))c12'),
    ('MLSMR', 'Dye 7', 'N=C1[#6]:,=[#6]C(=[C,N])[#6]:,=[#6]1'),
    ('MLSMR', 'Dye 11', '*=,:[#6]C([#6]=,:*)[#6]=,:*'),
    ('MLSMR', 'Dye 9', 'ac-*=&!@*-&!@C(=O)-&!@ca'),
    ('MLSMR', 'Dye 32', 'c1cccc2C(=O)C(C:,=*)C(=O)c12'),
    ('MLSMR', 'Dye 6', 'c12cccc(C(=O)C(ca)C(=O)3)c2c3ccc1'),
    ('MLSMR', 'Dye 22', 'NS(=O)(=O)c1cccc([#7])c1'),
    ('MLSMR', 'Dye 2', 'OCccCO'),
    ('MLSMR', 'Dye 26', 'c1ccccc1-n2nnnc2[CH2]*'),
    ('MLSMR', 'alkyl halide', '[Br,Cl,I][CX4,CH,CH2,CH3]'),
    ('MLSMR', 'Perhalo_ketone', 'O=CC(-[F,Cl,Br,I])([F,Cl,Br,I])-[F,Cl,Br,I]'),
    ('MLSMR', 'Beta halo carbonyl', 'O=CCC[F,Cl,Br,I]'),
    ('MLSMR', '4-halopyridine', '[F,Cl,Br][c]1:[c,n]:[c,n]:[n]:[c,n]:[c,n]1'),
    ('MLSMR', '2-halopyridine', '[F,Cl,Br][c]1:[c,n]:[c,n]:[c,n]:[c,n]:[n]1'),
    ('MLSMR', 'Hetero_hetero', '*[N,S,O]-&!@[N,S,O][#6]'),
    ('MLSMR', 'peroxide', 'OO'),
    ('MLSMR', 'disulfide', 'SS'),
    ('MLSMR', 'hydrazine', '[#6]-[NH]-[NH]-[#6]'),
    ('MLSMR', 'acyl hydrazine', '[N;R0][N;R0]C(=O)'),
    ('MLSMR', 'vinyl michael acceptor1', '[#6]-[CH1]=C-C(=O)[#6,#7,#8]'),
    ('MLSMR', 'vinyl michael acceptor2', '[CH2]=C-C(=O)[#6,#7,#8]'),
    ('MLSMR', 'michael acceptor 5', 'N#CC(=C)C#N'),
    ('MLSMR', 'Michael acceptor 6', '[#6,#7]-&!@[#6](=&!@[CH])-&!@C(=O)-&!@[C,N,O,S]'),
    ('MLSMR', 'alkynyl michael acceptor1', '[#6]-C#CC(=O)[#6,#7,#8]'),
    ('MLSMR', 'alkynyl michael acceptor2', '[CH1]#CC(=O)-[#6,#7,#8]'),
    ('MLSMR', 'nitroalkane', 'C[N+](=O)[O-]'),
    ('MLSMR', 'crown ether', '[O;R1][C;R1][C;R1][O;R1][C;R1][C;R1][O;R1]'),
    ('MLSMR', 'nitrate', '[#6]-O-[N+](=O)[O-]'),
    ('MLSMR', 'Oxalyl', 'O=C-&!@C=O'),
    ('MLSMR', 'Dipeptide', '*-C(=O)-&!@[NH]-C-&!@C(=O)-&!@[NH]-*'),
    ('MLSMR', 'quaternary nitroxy', 'C[N+](-[O-])(C)C'),
    ('MLSMR', 'Triphenylphosphine', 'a-P(-a)-a'),
    ('MLSMR', 'Phosphoric acid', '[OH]-P(=O)(-O)-*'),
    ('MLSMR', 'Phosphoric ester', 'COP(=O)(-*)O'),
    ('MLSMR', 'di/triphosphate', 'P(=O)([OH])OP(=O)[OH]'),
    ('MLSMR', 'tri phosphoric esters', '[#6]OP(=O)(*)O[#6].[#6]OP(=O)(*)O[#6].[#6]OP(=O)(*)O[#6]'),
    ('MLSMR', 'phosphoramide', 'NP(=O)(N)N'),
    ('MLSMR', 'Phenalene', 'c1(c)c2c(c)cccc2ccc1'),
    ('MLSMR', '(poly(azo(anthracene))', 'c12:[c,n]:[c,n]:[c,n]:[c,n]:c1[c,n]c3:[c,n]:[c,n]:[c,n]:[c,n]:c3[c,n]2'),
    ('MLSMR', '(poly(azo(phenanthrene))', 'c12:[c,n]:[c,n]:[c,n]:[c,n]:c1:[c,n]:[c,n]:c3:[c,n]:[c,n]:[c,n]:[c,n]:c23'),
    ('MLSMR', 'Dye 31', 'a1aaac2ac3acac4aaac(c34)c12'),
    ('MLSMR', 'Dye 4', 'c12ccccc1C(=O)c3ccccc3C2=O'),
    ('MLSMR', 'Dye 8', 'c12cccc(C(=O)N(-&!@C)C(=O)3)c2c3ccc1'),
    ('MLSMR', 'epoxide, aziridine, thioepoxide', 'C1[O,S,N]C1'),
    ('MLSMR', 'propiolactone', 'C1(=O)OCC1'),
    ('MLSMR', 'b-lactam', 'N1CCC1=O'),
    ('MLSMR', 'cycloheximide', 'O=C1CCCC(N1)=O'),
    ('MLSMR', 'aromatic Sulfonic ester', '[#6,#7]-S(=O)(=O)Oc'),
    ('MLSMR', 'quinone', '[$([o,n]=c1ccc(=[o,n])cc1),$([O,N]=C1C=CC(=[O,N])C=C1),$([O,N]=C1[#6]:,=[#6]C(=[O,N])[#6]:,=[#6]1)]'),
    ('MLSMR', 'saponin', 'O1CCCCC1OC2CCC3CCCCC3C2'),
    ('MLSMR', 'monensin', 'O1CCCCC1C2CCCO2'),
    ('MLSMR', 'squalestatin', 'C12OCCC(O1)CC2'),
    ('MLSMR', 'cyanidin', '[OH]c1cc([OH])cc2=[O+]C(=C([OH])Cc21)c3cc([OH])c([OH])cc3'),
    ('MLSMR', 'cytochalasin', 'O=C1NCC2CCCCC21'),
    ('Inpharmatica', 'Filter1_2_halo_ether', '[Cl,Br,I][CX4][CX4][$([O,S,N]*),Cl,Br,I]'),
    ('Inpharmatica', 'Filter2_acyl_phosphyl_sulfonyl_halide', '[C,S,P](=O)[F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter3_allyl_halide', '[F,Cl,Br,I][CX4]C=C'),
    ('Inpharmatica', 'Filter4_alpha_halo_carbonyl', '[Br,Cl,I][C!H0]C=[O,S]'),
    ('Inpharmatica', 'Filter5_azo', '[$([NX2R0]-[!#7]),$([NX2R0H1])]=[$([NX2R0]-[!#7]),$([NX2R0H1])]'),
    ('Inpharmatica', 'Filter6_benzyl_halide', '[Br,Cl,I][CX4]c'),
    ('Inpharmatica', 'Filter7_diazo', '[!#7]~[NX2]~[NX1]'),
    ('Inpharmatica', 'Filter8_thio_isocyanat_diimin', 'N=C=[N,O,P,S]'),
    ('Inpharmatica', 'Filter9_metal', '[$([!#1!#6!#7!#8!#9!#15!#16!#17!#35!#53]~[*]),$([!#1!#6!#7!#8!#9!#15!#16!#17!#35!#53;h])]'),
    ('Inpharmatica', 'Filter10_Terminal_vinyl', '[CH2]=[CH][N,O,P,S;R0]'),
    ('Inpharmatica', 'Filter11_nitrosamin', '[NR0]~[NX2]~[OX1]'),
    ('Inpharmatica', 'Filter12_nitroso', '[$([NX2]~*),$([NX2H])]~[OX1]'),
    ('Inpharmatica', 'Filter13_PS_double_bond', 'S=[$([PX2]~*),$([PX2H])]'),
    ('Inpharmatica', 'Filter14_thio_oxopyrylium_salt', 'c1ccc[s,o;X2]c1'),
    ('Inpharmatica', 'Filter15_thiosulfate', '[SX4](~S)(~O)(~O)~*'),
    ('Inpharmatica', 'Filter16_trialkyl_phosphin', '[PX3]([#6])([#6])[#6]'),
    ('Inpharmatica', 'Filter17_trialkyl_phosphin2', '[PX4](~[!O!S])([#6])([#6])[#6]'),
    ('Inpharmatica', 'Filter18_oxime_ester', '[$([CX3H1]-[!#7]),$([CX3H2])]=NO[C,S,P](=O)*'),
    ('Inpharmatica', 'Filter19_hydroxyimide_ester', 'O=C[NX3](C=O)OC(=O)*'),
    ('Inpharmatica', 'Filter20_hydrazine', '[Nv3X3][Nv3X3!H0]'),
    ('Inpharmatica', 'Filter21_cyanhydrin', '[NX1]#C[CX4][OH]'),
    ('Inpharmatica', 'Filter22_sulfonium_salt', '*[SX3](*)*'),
    ('Inpharmatica', 'Filter23_ortho_quinone', 'C1(C=CC=CC1=O)=O'),
    ('Inpharmatica', 'Filter24_react_imide', '*C([F,Cl,Br,I,$(OS(=O)(=O)*)])=[NX2]*'),
    ('Inpharmatica', 'Filter25_sulfonyl_halide', '*S(=O)(=O)[F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter26_alkyl_halide', 'AA[CH2][F,Cl,Br,I,$(OS(=O)(=O)*)]'),
    ('Inpharmatica', 'Filter27_anhydride', '*[C,S](=O)O[C,S](=O)*'),
    ('Inpharmatica', 'Filter28_halo_pyrimidine', 'c1nc(ncc1)[Br,I]'),
    ('Inpharmatica', 'Filter29_thioester', 'CC(=[OX1,SX1,NH])[$([SX2]-*),$([SX2H])]'),
    ('Inpharmatica', 'Filter30_beta_halo_carbonyl', '[$(C(-C)(-C)(=O)),$([CH](=O)-C)](=O)CC[Br,I]'),
    ('Inpharmatica', 'Filter31_so_bond', '[SX2R0][OX2]'),
    ('Inpharmatica', 'Filter32_oo_bond', '[OX2R0][OX2]'),
    ('Inpharmatica', 'Filter33_c10_alkyl', '[CH3][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]'),
    ('Inpharmatica', 'Filter34_isotope', '[2#1,3#1,13C,14C,15N,125I,23F,22Na,32P,33P,35S,45Ca,57Co,103Ru,141Ce]'),
    ('Inpharmatica', 'Filter35_pp_bond', 'P-P'),
    ('Inpharmatica', 'Filter36_ss_double_bond', 'S=S'),
    ('Inpharmatica', 'Filter37_silicate', '[Si]~O'),
    ('Inpharmatica', 'Filter38_aldehyde', 'O=[CH]*'),
    ('Inpharmatica', 'Filter39_imine', 'C[CR0]=[NR0][*!O]'),
    ('Inpharmatica', 'Filter40_epoxide_aziridine', 'C1C[N,S,O]1'),
    ('Inpharmatica', 'Filter41_12_dicarbonyl', '*C(=O)C(=O)*'),
    ('Inpharmatica', 'Filter42_12_dicarbonyl_tautomer', '[*!$(C[OH])]=C([OH])C(=O)*'),
    ('Inpharmatica', 'Filter43_michael_acceptor_sp1', 'C#CC=O'),
    ('Inpharmatica', 'Filter44_michael_acceptor2', '[$([C!$(C1(=O)C=CC(=O)C=C1)]C),$([Ch]),$(C[OH0])](=O)C=[C!$(C[Nv3X3,OH])]'),
    ('Inpharmatica', 'Filter45_allyl_halide2', '[Br,Cl,I][CX4]C=[C,N,P]'),
    ('Inpharmatica', 'Filter46_nhalide', 'N[F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter47_so2f', 'O=S(=O)(*)F'),
    ('Inpharmatica', 'Filter48_foso', 'O=S(*)OF'),
    ('Inpharmatica', 'Filter49_halogen', '[FX2,ClX2,BrX2,IX2,IX3,IX4,IX5]'),
    ('Inpharmatica', 'Filter50_grignard', 'C[Mg][F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter51_pn3', 'N[PX3](N)N'),
    ('Inpharmatica', 'Filter52_NC_haloamine', 'NC[F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter53_para_quinones', 'O=C1C=CC(=O)C=C1'),
    ('Inpharmatica', 'Filter56_SS_bond', 'S-S'),
    ('Inpharmatica', 'Filter57_polyphenol1', 'Oc1cc(O)cc(O)c1'),
    ('Inpharmatica', 'Filter58_polyphenol2', 'Oc1c(O)cc(O)cc1'),
    ('Inpharmatica', 'Filter59_phoshorous_ylide', 'C=P'),
    ('Inpharmatica', 'Filter60_Acyclic_N-S', 'N!@[SX2]'),
    ('Inpharmatica', 'Filter61_phosphor_halide_and_P_S_bond', '[#15]~[F,Cl,Br,I,#16]'),
    ('Inpharmatica', 'Filter62_oxo_thio_halide', '[O,S]~[Cl,Br,I]'),
    ('Inpharmatica', 'Filter63_polyaromatic', 'a1aaaa2ccc3ccccc3c12'),
    ('Inpharmatica', 'Filter64_halo_ketone_sulfone', '[#6][C,S](=[O,S])C[F,Cl,Br,I]'),
    ('Inpharmatica', 'Filter65_alkyl_sulfonate', '[#6][P,S](~[OX1])(~[OX1])O[C!H0]'),
    ('Inpharmatica', 'Filter66_c4_perfluoralkyl', 'C(F)(F)C(F)(F)C(F)(F)C(F)(F)'),
    ('Inpharmatica', 'Filter67_S_or_O_C_triplebond_N', '[O,S]C#N'),
    ('Inpharmatica', 'Filter68_anthracene_acridine', 'a1aaac2cc3ccccc3cc12'),
    ('Inpharmatica', 'Filter69_thio_carbonate', '[O,S]C(=[O,S])[O,S]'),
    ('Inpharmatica', 'Filter70_AlkylCN2', '*(C#N)C#N'),
    ('Inpharmatica', 'Filter71_thio_anhydride', '*C(=[O,S])[O,S]C(=[O,S])*'),
    ('Inpharmatica', 'Filter72_hydrated_di_ketone', 'C(=O)C([OH])[OH]'),
    ('Inpharmatica', 'Filter73_thio_ketone', 'CC(=S)C'),
    ('Inpharmatica', 'Filter74_thiol', '*[SX2H]'),
    ('Inpharmatica', 'Filter75_alkyl_Br_I', '[C!H0][Br,I]'),
    ('Inpharmatica', 'Filter76_S_ester', 'CC(=S)O'),
    ('Inpharmatica', 'Filter77_alkyl_NO2', '[*!#7]~[#6]-,=CN(~O)(~O)'),
    ('Inpharmatica', 'Filter78_bicyclic_Imide', '*~@C1C(=O)NC(=O)C(~@*)1'),
    ('Inpharmatica', 'Filter79_maleimide', '[CH]1C(=O)NC(=O)[CH]=1'),
    ('Inpharmatica', 'Filter80_Thioepoxide_aziridone', '[N,S]1[C,N,S][C,N,S]1'),
    ('Inpharmatica', 'Filter81_Thiocarbamate', 'S!@C(=!@[O,S])!@[#7]'),
    ('Inpharmatica', 'Filter82_pyridinium', '[c,n]1[c,n][c,n][c,n][c,n]n(C)1'),
    ('Inpharmatica', 'Filter83_per_halo_chain', '[F,Cl,Br,I]C-,=;!@C([F,Cl,Br,I])-,=;!@C([F,Cl,Br,I])'),
    ('Inpharmatica', 'Filter84_nitrogen_mustard', '[N,P,Se,S][C!H0]C[Br,Cl,I]'),
    ('Inpharmatica', 'Filter85_keto_acrylonitrile', '*C(=O)C(=!@[C!H0])C#N'),
    ('Inpharmatica', 'Filter86_cyanamide', 'N#C[#7!$(N(C#N)=C(N)NC)]'),
    ('Inpharmatica', 'Filter87_crowns', '[N,O,S]-@[#6!$(*(~@*)(~@*)~@*)]@[#6!$(*(~@*)(~@*)~@*)]-@[N,O,S]-@[#6!$(*(~@*)(~@*)~@*)]@[#6!$(*(~@*)(~@*)~@*)]-@[O,N]-@[#6!$(*(~@*)(~@*)~@*)]@[#6!$(*(~@*)(~@*)~@*)]-@[N,O,S]-@C'),
    ('Inpharmatica', 'Filter88_ene_sulfone', '[C!H0]=CS(=O)(=O)*'),
    ('Inpharmatica', 'Filter89_hydroxylamine', '[*!$(C=O)]!@N!@[$([OX2])]'),
    ('Inpharmatica', 'Filter90_N_double_bond_S', 'N=!@S'),
    ('Inpharmatica', 'Filter92_trityl', '*(c1ccccc1)(c1ccccc1)(c1ccccc1)'),
    ('Inpharmatica', 'Filter93_acetyl_urea', 'C(=O)!@N!@C(=O)[#7]'),
    ('Inpharmatica', 'Filter94_2_halo_pyridine', 'c1nc(ccc1)[Br,I,Cl,F]'),
    ('LINT', 'aromatic NO2', 'O~N(=O)-c(:*):*'),
    ('LINT', 'deuterium', '[2H]'),
    ('LINT', 'C13', '[13#6]'),
    ('LINT', '2-chloropyridine', 'n1c(cccc1)Cl'),
    ('LINT', 'aniline', '[NH2D1]-c(:*):*'),
    ('LINT', 'Si,B,Se atoms', '[Si,B,Se]'),
    ('LINT', 'hetero imides', '[!#6]-[CH2]-N1C(=O)CCC(=O)1'),
    ('LINT', 'poly ethers', 'O[CH2][CH2]O-!@[CH2][CH2]O'),
    ('LINT', 'acyclic imines', '[$([CX3R0]([#6])[#6]),$([CX3HR0][#6])]=[$([NX2R0][#6]),$([NX2HR0])]'),
    ('LINT', 'alkyl esters of S or P', '[S,P](=O)OC'),
    ('LINT', 'ugly P compounds', 'P(=[O,S])[C,N]([C,N])[C,N]'),
    ('LINT', 'acyclic N-,=N and not N bound to carbonyl or sulfone', '[N;!$(N-[C,S]=*)]-,=;!@[N;!$(N-[C,S]=*)]'),
    ('LINT', 'acyclic N-C-N', 'N-!@[CX4]-!@N'),
    ('LINT', 'acyclic N-S', 'N-!@[SX2]-*'),
    ('LINT', 'mustards', '[N,S,O][CH2][CH2]-[F,Cl,Br,I]'),
    ('LINT', 'aldehyde', 'O=[C!H0]'),
    ('LINT', '1,2-dicarbonyl not in ring', 'O=[CX3]-!@[CX3]=O'),
    ('LINT', 'carbamate, T-boc Protected', 'NC(OC([CH3])([CH3])[CH3])=O'),
    ('LINT', 'carbamate, CBZ Protected', 'NC(O[CH2]c1ccccc1)=O'),
    ('LINT', 'carbamate include di-substitued N', 'OC(=O)-!@[NX3]'),
    ('LINT', 'acyl halide', 'O=C-[F,Cl,Br,I]'),
    ('LINT', 'alkyl halide', '[CH2]-[Cl,Br,I]'),
    ('LINT', 'alpha halo carbonyl', 'O=C-C-[F,Cl,Br,I]'),
    ('LINT', 'sufonyl halide', 'S(=O)(=O)-[F,Cl,Br,I]'),
    ('LINT', 'N:C-SCH2 groups', '[ND1]=C-!@[SX2]-[CH2D2]'),
    ('LINT', '26', 'N-C(=S)-N'),
    ('LINT', 'terminal vinyl', '[CH2D1]=[CD2]-!@*'),
    ('LINT', '28', 'S=P~*'),
    ('LINT', 'thio cyanates', 'S-C#N'),
    ('LINT', 'thiols', '*-[S!H0]'),
    ('LINT', 'thionyl', '[Sv4](=O)(-!@[!#1])-!@[!#1]'),
    ('LINT', 'n-haloamines', 'N-[F,Cl,Br,I]'),
    ('LINT', 'N-C-Hal or cyano methyl', 'N-C-[F,Cl,Br,I,$(C#N)]'),
    ('LINT', 'alpha beta-unsaturated ketones; center of Michael reactivity', '[$(C#N),$(N(~O)~O),$(C=O),$(S(=O)=O),$(C(F)(F)F),Cl][C!H0]=[C!H0]'),
    ('LINT', 'aliphatic ketone not ring and not di-carbonyl', '[C;!$(C=*)][C!R](=O)[CH2D2]'),
    ('LINT', 'aliphatic ester, not lactones', 'C[C!R](=O)[O!R][CH2D2]'),
    ('LINT', 'long aliphatic chain, 6+', '[CH2][CH2][CH2]-!@[CH2][CH2][CH2]'),
    ('LINT', 'quinones', 'O=[#6]1[#6]:,=[#6][#6](=O)[#6]:,=[#6]1'),
    ('LINT', 'acyclic C=C-O', 'C=C-!@O-*'),
    ('LINT', 'acyclic C=N-H', '[NH1X2]=[C!R;!$(C(-N)(=[NH1])-N)]'),
    ('LINT', 'acyclic NO not nitro', 'O-!@[N;!$(N(=O)=O);!$([N+](=O)[O-])]'),
    ('LINT', '42', 'S-S'),
    ('LINT', '43', 'O~O'),
    ('LINT', 'thioester', 'C-C(=O)[SD2]'),
    ('LINT', 'aziridine-like N in 3-membered ring', 'N~1~*~*1'),
    ('LINT', 'epoxides', 'O~1C~*1'),
    ('LINT', 'aryl iodide', 'I-c:*'),
    ('LINT', 'aryl bromide', 'Br-c:*'),
    ('LINT', 'multiple aromatic rings', 'a1aaa2a(a1)aaa(a2):a'),
    ('LINT', 'multiple aromatic rings', '[!#1]-,:1-:a2a(-:[!#1]:3:a1aaaa3)aaaa2'),
    ('LINT', 'S/PO3 groups', '[P,S](~O)(~O)~O'),
    ('LINT', 'adamantyl', 'C12CC3CC(C1)CC(C2)C3'),
    ('LINT', 'too many cyano Groups (>1)', 'C#N.C#N'),
    ('LINT', 'too many COOH groups (>1)', '[CX3](=O)[OH1].[CX3](=O)[OH1]'),
    ('LINT', 'amino acid', '[NH2][CX4]C(=O)O'),
    ('LINT', 'chlorates', 'Cl~O'),
    ('LINT', 'high halogen content (>3)', '[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]'),
]  # fmt: skip
_ALERT_Q = {}
_RULE_SETS = ("Glaxo", "Dundee", "BMS", "PAINS", "SureChEMBL", "MLSMR", "Inpharmatica", "LINT")


def reos_filter(
    smiles,
    rule_sets=("Inpharmatica",),
    mw=(0.0, 500.0),
    logp=(-5.0, 5.0),
    hbd=(0, 5),
    hba=(0, 10),
    tpsa=(0.0, 200.0),
    rot=(0, 10),
) -> RichResult:
    r"""REOS-style nuisance-compound filter (Walters and Murcko 2002) as implemented in rd_filters.

    A molecule passes when none of the substructure alerts of the chosen
    rule sets matches and every property of :func:`molecular_properties`
    lies in its closed range. Defaults are rd_filters' ``rules.json``: MW
    0-500, logP -5 to 5, HBD 0-5, HBA 0-10, TPSA 0-200, rotatable bonds
    0-10 and the Inpharmatica alerts; ``rule_sets`` can be any of Glaxo,
    Dundee (Brenk et al. 2008), BMS, PAINS, SureChEMBL, MLSMR, Inpharmatica
    and LINT (1251 alerts in all).

    Returns ``passed``, ``filter`` (the first matching alert as rd_filters
    reports it, or ``"OK"``), all matching ``alerts`` (rule set,
    description), the ``properties`` and the ``violations`` (property names
    out of range).

    References
    ----------
    Walters, W. P. and Murcko, M. A. (2002). Prediction of "drug-likeness".
    *Advanced Drug Delivery Reviews*, 54(3), 255-271.
    Walters, P. rd_filters (github.com/PatWalters/rd_filters), alert_collection.csv
    and rules.json.
    Brenk, R. et al. (2008). Lessons learnt from assembling screening
    libraries for drug discovery for neglected diseases. *ChemMedChem*, 3,
    435-444.

    Examples
    --------
    >>> r = reos_filter("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.passed, r.filter
    (True, 'OK')
    >>> reos_filter("CCCCCCCCCCCCBr", rule_sets=("Glaxo",)).filter
    'R1 Reactive alkyl halides > 0'
    """
    bad = [s for s in rule_sets if s not in _RULE_SETS]
    if bad:
        raise ValueError("unknown rule set: " + ", ".join(bad))
    m = _molecule(smiles)
    cache = {}
    alerts = []
    for rs, desc, sm in _ALERTS:
        if rs not in rule_sets:
            continue
        if sm not in _ALERT_Q:
            _ALERT_Q[sm] = _parse(sm)
        if _has(m, _ALERT_Q[sm], cache):
            alerts.append((rs, desc))
    props = molecular_properties(smiles)
    ranges = {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa, "Rot": rot}
    viol = [k for k, (lo, hi) in ranges.items() if not (lo <= props[k] <= hi)]
    return RichResult(
        payload={
            "passed": not alerts and not viol,
            "filter": alerts[0][1] + " > 0" if alerts else "OK",
            "alerts": alerts,
            "properties": dict(props),
            "violations": viol,
        }
    )


# ---------------------------------------------------------------- Morgan environments (RDKit-compatible ids)
_U32 = 0xFFFFFFFF
# exact masses of the isotopes most often written in SMILES (other isotopes use their mass number)
_ISO_MASS = {(1, 2): 2.014101778, (1, 3): 3.016049268, (6, 11): 11.011433, (6, 13): 13.00335484,
             (6, 14): 14.003241989, (7, 15): 15.0001089, (8, 18): 17.9991610, (9, 18): 18.0009380,
             (15, 32): 31.97390727, (16, 35): 34.96903216, (53, 123): 122.905589, (53, 125): 124.9046302,
             (53, 131): 130.9061246}  # fmt: skip


def _combine(seed, v):
    return (seed ^ ((v + 0x9E3779B9 + ((seed << 6) & _U32) + (seed >> 2)) & _U32)) & _U32


def _hash_pair(a, b):
    return _combine(_combine(0, a & _U32), b)


def _atom_rings(m):
    inring = [False] * m["n"]
    for r in m["ring_bonds"]:
        for k in r:
            a, b = m["bonds"][k]
            inring[a] = inring[b] = True
    return inring


def morgan_environments(smiles, radius: int = 2) -> RichResult:
    r"""Unfolded Morgan (ECFP-like) environment identifiers as RDKit's Morgan fingerprint generator computes them.

    Atom invariants hash (atomic number, total degree, total H, formal
    charge, mass shift, ring flag) with RDKit's 32-bit ``hash_combine``;
    each round hashes the layer, the atom's code and its neighbours' sorted
    (bond type, code) pairs; an environment whose bond set was already seen
    is dropped and its atom retired, as in ``MorganEnvGenerator``. Returns
    ``counts`` (``{id: count}``, the sparse count fingerprint) and
    ``environments`` (``(id, atom, radius)``).

    References
    ----------
    Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints.
    *J. Chem. Inf. Model.*, 50(5), 742-754.
    Landrum, G. et al. RDKit ``MorganGenerator.cpp`` and ``hash.hpp``.

    Examples
    --------
    >>> sorted(morgan_environments("CC", radius=1).counts.items())
    [(2246728737, 2), (3545175291, 1)]
    """
    m = _molecule(smiles)
    n = m["heavy"]
    inring = _atom_rings(m)
    cur = []
    for i in range(n):
        z = m["z"][i]
        hs = m["h"][i] + sum(1 for j, _ in m["nbr"][i] if m["z"][j] == 1)
        deg = len(m["nbr"][i]) + m["h"][i]
        mass = _ISO_MASS.get((z, m["iso"][i]), float(m["iso"][i])) if m["iso"][i] else _WEIGHT[z - 1]
        comp = [z, deg, hs, m["chg"][i] & _U32, int(mass - _WEIGHT[z - 1]) & _U32]
        if inring[i]:
            comp.append(1)
        seed = 0
        for c in comp:
            seed = _combine(seed, c)
        cur.append(seed)
    nbrs = [[(j, k) for j, k in m["nbr"][i] if j < n] for i in range(n)]
    bt = [12 if o == 4 else o for o in m["order"]]
    envs = [(cur[i], i, 0) for i in range(n)]
    nbhd = [frozenset() for _ in range(n)]
    seen = set()
    dead = [False] * n
    for layer in range(radius):
        nxt = [0] * n
        rnd = []
        new_nbhd = list(nbhd)
        for i in range(n):
            if dead[i]:
                continue
            if not nbrs[i]:
                dead[i] = True
                continue
            s = set(nbhd[i])
            pairs = []
            for j, k in nbrs[i]:
                s.add(k)
                s |= nbhd[j]
                pairs.append((bt[k], cur[j]))
            pairs.sort()
            invar = _combine(layer, cur[i])
            for a, b in pairs:
                invar = _combine(invar, _hash_pair(a, b))
            nxt[i] = invar
            new_nbhd[i] = frozenset(s)
            rnd.append((tuple(sorted(s)), invar, i))
        rnd.sort()
        for key, invar, i in rnd:
            if key not in seen:
                envs.append((invar, i, layer + 1))
                seen.add(key)
            else:
                dead[i] = True
        cur = nxt
        nbhd = new_nbhd
    counts = {}
    for code, _i, _r in envs:
        counts[code] = counts.get(code, 0) + 1
    return RichResult(payload={"counts": counts, "environments": envs})


# ---------------------------------------------------------------- synthetic accessibility
def _ranks(m):
    """Symmetry classes by iterated refinement of (element, degree, H, charge, isotope) with neighbour classes."""
    n = m["heavy"]
    lab = [(m["z"][i], len(m["nbr"][i]), m["h"][i], m["chg"][i], m["iso"][i]) for i in range(n)]
    cls = _relabel(lab)
    for _ in range(n):
        lab = [(cls[i], tuple(sorted((m["order"][k], cls[j]) for j, k in m["nbr"][i] if j < n))) for i in range(n)]
        new = _relabel(lab)
        if len(set(new)) == len(set(cls)):
            break
        cls = new
    return cls


def _relabel(lab):
    keys = sorted(set(lab))
    idx = {k: r for r, k in enumerate(keys)}
    return [idx[x] for x in lab]


def _stereo_centres(m):
    """Possible tetrahedral centres: C/Si/Ge/Sn with four single bonds, N+/P/As with four neighbours, S/Se with
    three, at most one H, and all substituents in distinct symmetry classes."""
    cls = _ranks(m)
    out = []
    for i in range(m["heavy"]):
        z = m["z"][i]
        nb = [(j, k) for j, k in m["nbr"][i] if j < m["heavy"]]
        conn = len(nb) + m["h"][i]
        if m["h"][i] > 1:
            continue
        if z in (6, 14, 32, 50):
            ok = conn == 4 and all(m["order"][k] == 1 for _, k in nb)
        elif z in (7, 15, 33):
            ok = conn == 4 and (z != 7 or m["chg"][i] == 1)
        elif z in (16, 34):
            ok = conn == 3
        else:
            ok = False
        if not ok:
            continue
        sub = [cls[j] for j, _ in nb] + ([-1] if m["h"][i] else [])
        if len(set(sub)) == len(sub):
            out.append(i)
    return out


def sa_score(smiles, fragment_scores=None) -> RichResult:
    r"""Synthetic accessibility score (Ertl and Schuffenhauer 2009), 1 (easy) to 10 (hard), as RDKit's sascorer.

    ``fragmentScore`` is the count-weighted mean contribution of the
    molecule's radius-2 Morgan environments (:func:`morgan_environments`;
    unknown environments score -4), taken from ``fragment_scores``
    (``{environment id: contribution}``, e.g. RDKit's ``fpscores.pkl.gz``,
    derived from PubChem fragment frequencies; the data are not bundled).
    ``complexity = -(n^1.005 - n) - log10(stereo + 1) - log10(spiro + 1) -
    log10(bridgehead + 1) - (log10 2 if a ring has more than 8 atoms)``,
    a symmetry correction ``0.5 ln(n / distinct ids)`` when positive, and
    the sum ``s`` is mapped to ``11 - (s + 5) / 6.5 * 9``, smoothed above 8
    by ``8 + ln(x - 8)`` and clipped to [1, 10]. Spiro and bridgehead atoms
    follow RDKit's SSSR definitions; possible stereocentres are counted
    from symmetry classes (an approximation of RDKit's legacy CIP ranking).

    With ``fragment_scores=None`` every environment is unknown (-4), so only
    the relative complexity terms are informative.

    References
    ----------
    Ertl, P. and Schuffenhauer, A. (2009). Estimation of synthetic
    accessibility score of drug-like molecules based on molecular complexity
    and fragment contributions. *Journal of Cheminformatics*, 1, 8.
    Ertl, P. and Landrum, G. (2013). RDKit ``Contrib/SA_Score/sascorer.py``.

    Examples
    --------
    >>> r = sa_score("c1ccccc1", {3218693969: 1.0, 98513984: 0.5, 2763854213: 0.0})
    >>> round(r.fragment_score, 6), r.n_stereo, round(r.symmetry, 6), round(r.score, 6)
    (0.5, 0, 0.346574, 2.979506)
    """
    fs = fragment_scores or {}
    me = morgan_environments(smiles, radius=2)
    m = _molecule(smiles)
    nf = 0
    s1 = 0.0
    for code, cnt in me.counts.items():
        nf += cnt
        s1 += fs.get(code, -4.0) * cnt
    s1 /= nf
    n = sum(1 for i in range(m["heavy"]) if m["z"][i] != 1)
    ring_atoms = []
    for r in m["ring_bonds"]:
        at = set()
        for k in r:
            at.update(m["bonds"][k])
        ring_atoms.append(at)
    spiro = set()
    bridge = set()
    for a in range(len(ring_atoms)):
        for b in range(a + 1, len(ring_atoms)):
            inter = ring_atoms[a] & ring_atoms[b]
            if len(inter) == 1:
                spiro |= inter
            shared = set(m["ring_bonds"][a]) & set(m["ring_bonds"][b])
            if len(shared) > 1:
                cnt = {}
                for k in shared:
                    for v in m["bonds"][k]:
                        cnt[v] = cnt.get(v, 0) + 1
                bridge |= {v for v, c in cnt.items() if c == 1}
    stereo = len(_stereo_centres(m))
    macro = any(len(at) > 8 for at in ring_atoms)
    s2 = (
        0.0
        - (n**1.005 - n)
        - math.log10(stereo + 1)
        - math.log10(len(spiro) + 1)
        - math.log10(len(bridge) + 1)
        - (math.log10(2) if macro else 0.0)
    )
    s3 = 0.5 * math.log(n / len(me.counts)) if n > len(me.counts) else 0.0
    raw = s1 + s2 + s3
    sa = 11.0 - (raw + 4.0 + 1.0) / 6.5 * 9.0
    if sa > 8.0:
        sa = 8.0 + math.log(sa + 1.0 - 9.0)
    sa = min(max(sa, 1.0), 10.0)
    return RichResult(
        payload={
            "score": sa,
            "fragment_score": s1,
            "complexity": s2,
            "symmetry": s3,
            "n_stereo": stereo,
            "n_spiro": len(spiro),
            "n_bridgehead": len(bridge),
            "macrocycle": macro,
        }
    )


def cheatsheet() -> str:
    return (
        "maccs_fingerprint(smiles) -> 166 MACCS keys; reos_filter(smiles) -> rd_filters REOS verdict; "
        "molecular_properties(smiles); morgan_environments(smiles); sa_score(smiles, fragment_scores)."
    )


# alias kept from the retired placeholder of the same name
maccs_keys = maccs_fingerprint

# alias kept from the retired placeholder of the same name
synthetic_accessibility = sa_score
