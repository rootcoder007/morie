# morie.fn -- function file (rootcoder007/morie)
"""Molecular descriptors from SMILES: average molecular weight, Lipinski hydrogen-bond acceptors and donors,
Veber rotatable bonds and Ertl's topological polar surface area (TPSA)."""

from __future__ import annotations

from ._qpcore import ssum
from ._richresult import RichResult
from .avalon import implicit_h, parse_smiles

__all__ = [
    "smiles_molecular_weight",
    "smiles_hba",
    "smiles_hbd",
    "smiles_rotatable_bonds",
    "smiles_tpsa",
    "lipinski_descriptors",
]

# standard atomic weights (IUPAC; the values of RDKit's periodic table)
_MASS = {
    "H": 1.008,
    "B": 10.812,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.067,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
}


def _mol(smiles):
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    hs = implicit_h(el, arom, chg, hexp, bonds)
    # the parser gives ring-closure bonds between aromatic atoms order 1; a
    # ring bond joining two aromatic atoms is aromatic (order 4)
    bonds = [
        (a, b, 4 if (o == 1 and arom[a] and arom[b] and _ring_bond(len(el), bonds, k)) else o)
        for k, (a, b, o) in enumerate(bonds)
    ]
    return el, arom, chg, hs, bonds


def smiles_molecular_weight(smiles: str) -> float:
    r"""Average molecular weight (g/mol) of a SMILES structure, implicit hydrogens included.

    Sum of standard atomic weights over heavy atoms and the hydrogens filled
    to the default valence of the organic subset (as
    :func:`morie.fn.avalon.implicit_h`).

    References
    ----------
    Meija, J. et al. (2016). Atomic weights of the elements 2013 (IUPAC
    Technical Report). *Pure and Applied Chemistry*, 88, 265-291.
    Weininger, D. (1988). SMILES, a chemical language and information system.
    *Journal of Chemical Information and Computer Sciences*, 28, 31-36.

    Examples
    --------
    >>> round(smiles_molecular_weight("CC(=O)Oc1ccccc1C(=O)O"), 3)
    180.159
    """
    el, _, _, hs, _ = _mol(smiles)
    return ssum(_MASS[e] for e in el) + ssum(hs) * _MASS["H"]


def smiles_hba(smiles: str) -> int:
    r"""Lipinski hydrogen-bond acceptor count: the number of nitrogen and oxygen atoms.

    References
    ----------
    Lipinski, C. A., Lombardo, F., Dominy, B. W. and Feeney, P. J. (1997).
    Experimental and computational approaches to estimate solubility and
    permeability in drug discovery and development settings. *Advanced Drug
    Delivery Reviews*, 23, 3-25.

    Examples
    --------
    >>> smiles_hba("CC(=O)Oc1ccccc1C(=O)O")
    4
    """
    el = _mol(smiles)[0]
    return sum(1 for e in el if e in ("N", "O"))


def smiles_hbd(smiles: str) -> int:
    r"""Lipinski hydrogen-bond donor count: hydrogens on nitrogen and oxygen (the NH and OH count).

    References
    ----------
    Lipinski, C. A. et al. (1997). Experimental and computational approaches
    to estimate solubility and permeability in drug discovery and
    development settings. *Advanced Drug Delivery Reviews*, 23, 3-25.

    Examples
    --------
    >>> smiles_hbd("NCC(=O)O"), smiles_hbd("CC(=O)Oc1ccccc1C(=O)O")
    (3, 1)
    """
    el, _, _, hs, _ = _mol(smiles)
    return sum(hs[i] for i in range(len(el)) if el[i] in ("N", "O"))


def _ring_bond(n, bonds, k):
    a, b, _ = bonds[k]
    adj = [[] for _ in range(n)]
    for j, (u, v, _o) in enumerate(bonds):
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
    return b in seen


def smiles_rotatable_bonds(smiles: str, *, exclude_amide: bool = True) -> int:
    r"""Veber rotatable-bond count.

    A rotatable bond is any single, non-ring bond between two non-terminal
    heavy atoms; amide C-N bonds are excluded for their high rotation
    barrier (Veber et al. 2002).

    References
    ----------
    Veber, D. F., Johnson, S. R., Cheng, H.-Y., Smith, B. R., Ward, K. W. and
    Kopple, K. D. (2002). Molecular properties that influence the oral
    bioavailability of drug candidates. *Journal of Medicinal Chemistry*, 45, 2615-2623.

    Examples
    --------
    >>> smiles_rotatable_bonds("CC(=O)Oc1ccccc1C(=O)O"), smiles_rotatable_bonds("CC(=O)NCCC")
    (3, 2)
    """
    el, arom, _, _, bonds = _mol(smiles)
    n = len(el)
    deg = [0] * n
    for a, b, _ in bonds:
        deg[a] += 1
        deg[b] += 1

    def carbonyl(c):
        return el[c] == "C" and any(o == 2 and el[b if a == c else a] == "O" for a, b, o in bonds if c in (a, b))

    cnt = 0
    for k, (a, b, o) in enumerate(bonds):
        if o != 1 or deg[a] < 2 or deg[b] < 2 or _ring_bond(n, bonds, k):
            continue
        if exclude_amide and ((el[a] == "N" and carbonyl(b)) or (el[b] == "N" and carbonyl(a))):
            continue
        cnt += 1
    return cnt


def smiles_tpsa(smiles: str) -> RichResult:
    r"""Ertl's topological polar surface area (square angstroms) from nitrogen and oxygen fragment contributions.

    Each N and O atom contributes a tabulated area determined by its number
    of heavy neighbours, attached hydrogens, formal charge, single, double,
    triple and aromatic bonds and membership of a three-membered ring
    (Ertl, Rohde and Selzer 2000, Table 1, as implemented in RDKit);
    unlisted environments use ``30.5 - 8.2 nbrs + 1.5 H`` (N) and
    ``28.5 - 8.6 nbrs + 1.5 H`` (O), floored at zero.

    References
    ----------
    Ertl, P., Rohde, B. and Selzer, P. (2000). Fast calculation of molecular
    polar surface area as a sum of fragment-based contributions and its
    application to the prediction of drug transport properties. *Journal of
    Medicinal Chemistry*, 43, 3714-3717.

    Examples
    --------
    >>> round(smiles_tpsa("CC(=O)Oc1ccccc1C(=O)O").tpsa, 2)
    63.6
    """
    el, arom, chg, hs, bonds = _mol(smiles)
    n = len(el)
    ns, nd, nt, na, nb = [0] * n, [0] * n, [0] * n, [0] * n, [0] * n
    nbr = [set() for _ in range(n)]
    for a, b, o in bonds:
        nb[a] += 1
        nb[b] += 1
        nbr[a].add(b)
        nbr[b].add(a)
        tgt = na if o == 4 else ns if o == 1 else nd if o == 2 else nt
        tgt[a] += 1
        tgt[b] += 1
    contrib = [0.0] * n
    for i in range(n):
        e, h, c = el[i], hs[i], chg[i]
        if e not in ("N", "O"):
            continue
        r3 = any(k in nbr[j] for j in nbr[i] for k in nbr[i] if j < k)
        k_ = nb[i]
        t = -1.0
        if e == "N":
            if k_ == 1:
                t = (
                    23.79
                    if h == 0 and c == 0 and nt[i] == 1
                    else 23.85
                    if h == 1 and c == 0 and nd[i] == 1
                    else 26.02
                    if h == 2 and c == 0 and ns[i] == 1
                    else 25.59
                    if h == 2 and c == 1 and nd[i] == 1
                    else 27.64
                    if h == 3 and c == 1 and ns[i] == 1
                    else -1.0
                )
            elif k_ == 2:
                t = (
                    12.36
                    if h == 0 and c == 0 and ns[i] == 1 and nd[i] == 1
                    else 13.60
                    if h == 0 and c == 0 and nt[i] == 1 and nd[i] == 1
                    else 21.94
                    if h == 1 and c == 0 and ns[i] == 2 and r3
                    else 12.03
                    if h == 1 and c == 0 and ns[i] == 2 and not r3
                    else 4.36
                    if h == 0 and c == 1 and nt[i] == 1 and ns[i] == 1
                    else 13.97
                    if h == 1 and c == 1 and nd[i] == 1 and ns[i] == 1
                    else 16.61
                    if h == 2 and c == 1 and ns[i] == 2
                    else 12.89
                    if h == 0 and c == 0 and na[i] == 2
                    else 15.79
                    if h == 1 and c == 0 and na[i] == 2
                    else 14.14
                    if h == 1 and c == 1 and na[i] == 2
                    else -1.0
                )
            elif k_ == 3:
                t = (
                    3.01
                    if h == 0 and c == 0 and ns[i] == 3 and r3
                    else 3.24
                    if h == 0 and c == 0 and ns[i] == 3 and not r3
                    else 11.68
                    if h == 0 and c == 0 and ns[i] == 1 and nd[i] == 2
                    else 3.01
                    if h == 0 and c == 1 and ns[i] == 2 and nd[i] == 1
                    else 4.44
                    if h == 1 and c == 1 and ns[i] == 3
                    else 4.41
                    if h == 0 and c == 0 and na[i] == 3
                    else 4.93
                    if h == 0 and c == 0 and ns[i] == 1 and na[i] == 2
                    else 8.39
                    if h == 0 and c == 0 and nd[i] == 1 and na[i] == 2
                    else 4.10
                    if h == 0 and c == 1 and na[i] == 3
                    else 3.88
                    if h == 0 and c == 1 and ns[i] == 1 and na[i] == 2
                    else -1.0
                )
            elif k_ == 4 and h == 0 and ns[i] == 4 and c == 1:
                t = 0.0
            if t < 0:
                t = max(30.5 - k_ * 8.2 + h * 1.5, 0.0)
        else:
            if k_ == 1:
                t = (
                    17.07
                    if h == 0 and c == 0 and nd[i] == 1
                    else 20.23
                    if h == 1 and c == 0 and ns[i] == 1
                    else 23.06
                    if h == 0 and c == -1 and ns[i] == 1
                    else -1.0
                )
            elif k_ == 2:
                t = (
                    12.53
                    if h == 0 and c == 0 and ns[i] == 2 and r3
                    else 9.23
                    if h == 0 and c == 0 and ns[i] == 2 and not r3
                    else 13.14
                    if h == 0 and c == 0 and na[i] == 2
                    else -1.0
                )
            if t < 0:
                t = max(28.5 - k_ * 8.6 + h * 1.5, 0.0)
        contrib[i] = t
    return RichResult(payload={"tpsa": ssum(contrib), "contributions": contrib})


def lipinski_descriptors(smiles: str) -> RichResult:
    r"""Lipinski/Veber descriptor set of a SMILES structure: MW, HBA, HBD, rotatable bonds and TPSA.

    Examples
    --------
    >>> r = lipinski_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    >>> r.hba, r.hbd, r.rotatable_bonds
    (4, 1, 3)
    """
    return RichResult(
        payload={
            "molecular_weight": smiles_molecular_weight(smiles),
            "hba": smiles_hba(smiles),
            "hbd": smiles_hbd(smiles),
            "rotatable_bonds": smiles_rotatable_bonds(smiles),
            "tpsa": smiles_tpsa(smiles).tpsa,
        }
    )


def cheatsheet() -> str:
    return "smiles_molecular_weight / smiles_hba / smiles_hbd / smiles_rotatable_bonds / smiles_tpsa -> descriptors."


# alias kept from the retired placeholder of the same name
hbond_acceptor_count = smiles_hba

# alias kept from the retired placeholder of the same name
hbond_donor_count = smiles_hbd

# alias kept from the retired placeholder of the same name
molecular_weight = smiles_molecular_weight

# alias kept from the retired placeholder of the same name
polar_surface_area = smiles_tpsa

# alias kept from the retired placeholder of the same name
rotatable_bond_count = smiles_rotatable_bonds
