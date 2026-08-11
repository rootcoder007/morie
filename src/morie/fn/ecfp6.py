# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extended-connectivity fingerprint, radius 3 (ECFP6)."""

from ._richresult import RichResult

from .ecfp4 import _bonds, _connectivity_invariants, _defaults, _morgan

__all__ = ["ecfp6", "ecfp_6_fingerprint"]


def ecfp6(adjacency, atomnum, numhs=None, charge=None, inring=None,
          isotope_delta=None, nbits=2048, radius=3):
    """Extended-connectivity fingerprint of radius 3 (ECFP6).

    ECFP diameter 6 is Morgan radius 3.  Identical machinery to
    :func:`morie.fn.ecfp4.ecfp4` -- the same round-0 Daylight invariants
    and the same environment-relabelling and duplicate-retirement rules
    -- run for one more round, so the identifier set of ECFP6 contains
    that of ECFP4 on the same molecule.  That containment is exact and is
    one of the anchors used for this module.

    Molecules are supplied as a pre-parsed graph (square bond-order
    matrix plus per-atom property vectors), not as SMILES; see
    :func:`morie.fn.ecfp4.ecfp4` for the encoding and for the full
    statement of which parts are published specification and which parts
    (the folding hash) are this implementation's own stated choice.

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
        Morgan radius; 3 for ECFP6.

    Returns
    -------
    RichResult
        ``bits``, ``count``, ``nset``, ``identifiers``, ``nenv``, ``a``,
        ``nbits``, ``radius``, ``method``.

    References
    ----------
    Rogers, D. and Hahn, M. (2010), "Extended-connectivity fingerprints",
    Journal of Chemical Information and Modeling 50(5), 742-754,
    doi:10.1021/ci100050t.  Paywalled at ACS, NOT read for this
    implementation and recorded in ledger/wave3/NEEDED_SOURCES.md.  The
    specification followed is the RDKit reference implementation, files
    Code/GraphMol/Fingerprints/MorganGenerator.cpp (rounds and
    duplicate retirement, lines 395-495) and
    Code/GraphMol/Fingerprints/FingerprintUtil.cpp
    (getConnectivityInvariants, lines 242-265), master revision fetched
    2026-08-09, stored at library/pdf/fetched-wave3/
    rdkit-reference-source/.  RDKit: Open-Source Cheminformatics,
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
        "method": "ECFP6 (Morgan radius 3), Rogers-Hahn / RDKit"})


ecfp_6_fingerprint = ecfp6


def cheatsheet():
    return "ecfp6: extended-connectivity fingerprint, Morgan radius 3 (ECFP6)."
