# morie.fn -- function file (rootcoder007/morie)
"""DSSP secondary-structure assignment (Kabsch and Sander 1983) from backbone coordinates: the
electrostatic hydrogen-bond energy, n-turns, alpha/3-10/pi helices, beta bridges and ladders with
bulges, bends, and the 8-state alphabet H B E G I T S -."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["dssp_assign", "dssp_hbond_energy"]

_COUPLING = 27.888  # 332 kcal A / mol * 0.42 e * 0.20 e
_MIN_ENERGY = -9.9
_MAX_ENERGY = -0.5
_MIN_CA_DIST = 9.0
_MAX_PEPTIDE = 2.5
_MIN_DIST = 0.5


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dist(a, b):
    d = _sub(a, b)
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def dssp_hbond_energy(n, h, c, o) -> float:
    r"""Kabsch-Sander electrostatic energy (kcal/mol) of the hydrogen bond N-H ... O=C.

    ``E = 0.084 * 332 (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)``, floored at
    -9.9; any of the four distances below 0.5 A gives -9.9, as in DSSP.

    Examples
    --------
    >>> round(dssp_hbond_energy([0, 0, 0], [1, 0, 0], [4.2, 0, 0], [3.0, 0, 0]), 6)
    -2.573
    """
    r_on = _dist(o, n)
    r_ch = _dist(c, h)
    r_oh = _dist(o, h)
    r_cn = _dist(c, n)
    if min(r_on, r_ch, r_oh, r_cn) < _MIN_DIST:
        return _MIN_ENERGY
    e = _COUPLING * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
    return _MIN_ENERGY if e < _MIN_ENERGY else e


def dssp_assign(coords, sequence=None, chain=None, prefer_pi: bool = True) -> RichResult:
    r"""DSSP secondary structure from backbone atoms (Kabsch and Sander 1983; DSSP 2.2 rules).

    Each residue is ``[N, CA, C, O]`` (x, y, z in angstrom). The amide H is
    placed 1 A from N along the preceding C=O bond direction (at N itself
    for a chain start); prolines (``P`` in ``sequence``) donate no H-bond.
    Energies are computed for residue pairs with CA-CA < 9 A; each donor
    keeps its two lowest-energy acceptors and a bond needs E < -0.5
    kcal/mol. A chain break is a C(i-1)-N(i) distance above 2.5 A or a
    change of ``chain``.

    - n-turn at i: H-bond O(i) ... H-N(i+n), n = 3, 4, 5;
    - H: two consecutive 4-turns at i-1 and i give H at i..i+3; G (3-10)
      and I (pi) likewise for 3- and 5-turns where the residues are still
      free (``prefer_pi`` lets I overwrite H, as DSSP >= 2.2 does; False
      gives the 1983 priority);
    - bridges: parallel if [O(i-1)-N(j) and O(j)-N(i+1)] or [O(j-1)-N(i)
      and O(i)-N(j+1)], antiparallel if [O(i)-N(j) and O(j)-N(i)] or
      [O(i-1)-N(j+1) and O(j-1)-N(i+1)]; consecutive bridges form ladders,
      ladders joined across a bulge (gaps < 6 on one strand and < 3 on the
      other); a lone bridge is B, a ladder residue E;
    - T: residue inside an n-turn; S: CA(i-2), CA(i), CA(i+2) angle > 70
      degrees; everything else is loop ``-``.

    Parameters
    ----------
    coords : list
        Per residue ``[N, CA, C, O]`` coordinate triples.
    sequence : str, optional
        One-letter residue codes (only prolines matter).
    chain : list, optional
        Chain identifier per residue.
    prefer_pi : bool
        Let pi helices override alpha helices (DSSP 2.2 and later).

    Returns
    -------
    RichResult
        ``ss`` (string over H B E G I T S -), ``hbonds`` (per donor, up to
        two ``(acceptor, energy)`` pairs below -0.5), ``bridges`` (type,
        i-residues, j-residues), ``bend`` (per-residue flags).

    References
    ----------
    Kabsch, W. and Sander, C. (1983). Dictionary of protein secondary
    structure: pattern recognition of hydrogen-bonded and geometrical
    features. *Biopolymers*, 22(12), 2577-2637.
    Hekkelman, M. L. DSSP 2.2 (structure.cpp); McGibbon, R. T. et al.
    (2015). MDTraj. *Biophysical Journal*, 109(8), 1528-1532.

    Examples
    --------
    >>> import math
    >>> res = []
    >>> for i in range(12):
    ...     t = math.radians(100.0 * i)
    ...     ca = [2.3 * math.cos(t), 2.3 * math.sin(t), 1.5 * i]
    ...     n = [1.55 * math.cos(t - 0.45), 1.55 * math.sin(t - 0.45), 1.5 * i - 0.85]
    ...     c = [1.65 * math.cos(t + 0.42), 1.65 * math.sin(t + 0.42), 1.5 * i + 0.55]
    ...     o = [1.95 * math.cos(t + 0.62), 1.95 * math.sin(t + 0.62), 1.5 * i + 1.7]
    ...     res.append([n, ca, c, o])
    >>> len(dssp_assign(res).ss)
    12
    """
    n = len(coords)
    N = [[float(v) for v in r[0]] for r in coords]
    CA = [[float(v) for v in r[1]] for r in coords]
    C = [[float(v) for v in r[2]] for r in coords]
    OX = [[float(v) for v in r[3]] for r in coords]
    chain = list(chain) if chain is not None else [0] * n
    pro = [sequence is not None and sequence[i].upper() == "P" for i in range(n)]
    brk = [False] * n  # brk[i]: break between i-1 and i
    for i in range(1, n):
        brk[i] = chain[i] != chain[i - 1] or _dist(C[i - 1], N[i]) > _MAX_PEPTIDE

    def nobreak(a, b):
        return not any(brk[a + 1 : b + 1])

    H = []
    for i in range(n):
        if i > 0 and not brk[i]:
            u = _sub(C[i - 1], OX[i - 1])
            d = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
            H.append([N[i][0] + u[0] / d, N[i][1] + u[1] / d, N[i][2] + u[2] / d])
        else:
            H.append(N[i][:])
    acc = [[(-1, 0.0), (-1, 0.0)] for _ in range(n)]

    def store(donor, acceptor):
        if pro[donor]:
            return
        e = dssp_hbond_energy(N[donor], H[donor], C[acceptor], OX[acceptor])
        if e >= _MAX_ENERGY:
            return
        if acc[donor][0][0] < 0 or e < acc[donor][0][1]:
            acc[donor][1] = acc[donor][0]
            acc[donor][0] = (acceptor, e)
        elif acc[donor][1][0] < 0 or e < acc[donor][1][1]:
            acc[donor][1] = (acceptor, e)

    for i in range(n):
        for j in range(i + 1, n):
            if _dist(CA[i], CA[j]) < _MIN_CA_DIST:
                store(i, j)
                if j != i + 1:
                    store(j, i)

    def bond(donor, acceptor):
        # N-H of donor bonded to O=C of acceptor
        return acc[donor][0][0] == acceptor or acc[donor][1][0] == acceptor

    ss = ["-"] * n
    # bridges
    bridges = []
    for i in range(1, n - 4):
        for j in range(i + 3, n - 1):
            if not (nobreak(i - 1, i + 1) and nobreak(j - 1, j + 1)):
                continue
            a, b, c = i - 1, i, i + 1
            d, e, f = j - 1, j, j + 1
            if (bond(c, e) and bond(e, a)) or (bond(f, b) and bond(b, d)):
                typ = "parallel"
            elif (bond(c, d) and bond(f, a)) or (bond(e, b) and bond(b, e)):
                typ = "antiparallel"
            else:
                continue
            found = False
            for br in bridges:
                if typ != br["type"] or i != br["i"][-1] + 1:
                    continue
                if typ == "parallel" and br["j"][-1] + 1 == j:
                    br["i"].append(i)
                    br["j"].append(j)
                    found = True
                    break
                if typ == "antiparallel" and br["j"][0] - 1 == j:
                    br["i"].append(i)
                    br["j"].insert(0, j)
                    found = True
                    break
            if not found:
                bridges.append({"type": typ, "i": [i], "j": [j], "ci": chain[i], "cj": chain[j]})
    bridges.sort(key=lambda br: br["i"][0])
    k = 0
    while k < len(bridges):
        m = k + 1
        while m < len(bridges):
            bi, bj = bridges[k], bridges[m]
            ibi, iei, jbi, jei = bi["i"][0], bi["i"][-1], bi["j"][0], bi["j"][-1]
            ibj, iej, jbj, jej = bj["i"][0], bj["i"][-1], bj["j"][0], bj["j"][-1]
            # DSSP does this arithmetic on unsigned integers: a negative difference never passes a "<" test
            if (
                bi["type"] != bj["type"]
                or bi["ci"] != bj["ci"]
                or bi["cj"] != bj["cj"]
                or ibj < iei
                or ibj - iei >= 6
                or (iei >= ibj and ibi <= iej)
            ):
                m += 1
                continue
            if bi["type"] == "parallel":
                g = jbj - jei
                bulge = g >= 0 and ((g < 6 and ibj - iei < 3) or g < 3)
            else:
                g = jbi - jej
                bulge = g >= 0 and ((g < 6 and ibj - iei < 3) or g < 3)
            if bulge:
                bi["i"] = bi["i"] + bj["i"]
                bi["j"] = bi["j"] + bj["j"] if bi["type"] == "parallel" else bj["j"] + bi["j"]
                del bridges[m]
            else:
                m += 1
        k += 1
    for br in bridges:
        code = "E" if len(br["i"]) > 1 else "B"
        for lo, hi in ((br["i"][0], br["i"][-1]), (br["j"][0], br["j"][-1])):
            for r in range(lo, hi + 1):
                if ss[r] != "E":
                    ss[r] = code
    # turns and helices
    flag = {s: ["none"] * n for s in (3, 4, 5)}
    for s in (3, 4, 5):
        for i in range(n - s):
            if nobreak(i, i + s) and bond(i + s, i):
                flag[s][i + s] = "end"
                for j in range(i + 1, i + s):
                    if flag[s][j] == "none":
                        flag[s][j] = "middle"
                flag[s][i] = "startend" if flag[s][i] == "end" else "start"

    def start(s, i):
        return flag[s][i] in ("start", "startend")

    for i in range(1, n - 4):
        if start(4, i) and start(4, i - 1):
            for j in range(i, i + 4):
                ss[j] = "H"
    for i in range(1, n - 3):
        if start(3, i) and start(3, i - 1) and all(ss[j] in ("-", "G") for j in range(i, i + 3)):
            for j in range(i, i + 3):
                ss[j] = "G"
    free5 = ("-", "I", "H") if prefer_pi else ("-", "I")
    for i in range(1, n - 5):
        if start(5, i) and start(5, i - 1) and all(ss[j] in free5 for j in range(i, i + 5)):
            for j in range(i, i + 5):
                ss[j] = "I"
    bend = [False] * n
    for i in range(2, n - 2):
        if nobreak(i - 2, i + 2):
            u = _sub(CA[i], CA[i - 2])
            v = _sub(CA[i + 2], CA[i])
            cs = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / math.sqrt(
                (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            )
            bend[i] = math.degrees(math.acos(max(-1.0, min(1.0, cs)))) > 70.0
    for i in range(1, n - 1):
        if ss[i] == "-":
            turn = any(i >= k and start(s, i - k) for s in (3, 4, 5) for k in range(1, s))
            if turn:
                ss[i] = "T"
            elif bend[i]:
                ss[i] = "S"
    hb = [[p for p in acc[i] if p[0] >= 0] for i in range(n)]
    return RichResult(
        payload={
            "ss": "".join(ss),
            "hbonds": hb,
            "bridges": [(br["type"], br["i"], br["j"]) for br in bridges],
            "bend": bend,
        }
    )


def cheatsheet() -> str:
    return "dssp_assign(coords, sequence=None) -> DSSP 8-state secondary structure; dssp_hbond_energy(n, h, c, o) -> kcal/mol."


# alias kept from the retired placeholder of the same name
dssp_secondary = dssp_assign
