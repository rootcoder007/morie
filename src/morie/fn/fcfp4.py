# SPDX-License-Identifier: AGPL-3.0-or-later
"""Functional-class fingerprint, radius 2 (FCFP4)."""

from ._richresult import RichResult

from .ecfp4 import _bonds, _morgan

__all__ = ["fcfp4", "fcfp_4_fingerprint"]

# Order of the six pharmacophoric feature classes, and therefore of the
# bit positions in the round-0 invariant.  This order is fixed by the
# RDKit reference implementation and must not be permuted.
FEATURE_CLASSES = ("donor", "acceptor", "aromatic", "halogen", "basic", "acidic")


def fcfp4(adjacency, features, nbits=2048, radius=2):
    """Functional-class fingerprint of radius 2 (FCFP4).

    FCFP differs from ECFP only in the round-0 atom invariant.  Instead of
    the Daylight connectivity components, each atom carries a six-bit
    pharmacophoric feature code: bit 0 hydrogen-bond donor, bit 1
    hydrogen-bond acceptor, bit 2 aromatic, bit 3 halogen, bit 4 basic,
    bit 5 acidic.  Everything after round 0 -- the (layer, own identifier,
    sorted (bond order, neighbour identifier)) relabelling and the
    retirement of duplicate environments -- is shared with
    :func:`morie.fn.ecfp4.ecfp4`.

    The feature flags are an *input*.  Assigning them requires SMARTS
    substructure matching against the six Gobbi-Poppinger patterns, and a
    SMARTS engine is out of scope here; this function starts from the
    per-atom flags such an engine would produce.  The flag order above is
    the order of ``smartsPatterns`` in the RDKit source and is what makes
    the resulting codes comparable with an RDKit-derived feature table.

    Parameters
    ----------
    adjacency : array-like, shape (a, a)
        Symmetric bond-order matrix; 0 no bond, 1 single, 2 double,
        3 triple, 4 aromatic.
    features : array-like, shape (a, 6) or (a,)
        Per-atom feature flags in the order donor, acceptor, aromatic,
        halogen, basic, acidic; or, if one-dimensional, the already
        packed integer code sum(flag_i * 2^i) per atom.
    nbits : int
        Width of the folded fingerprint.
    radius : int
        Morgan radius; 2 for FCFP4.

    Returns
    -------
    RichResult
        ``bits``, ``count``, ``nset``, ``identifiers``, ``nenv``,
        ``featurecode``, ``a``, ``nbits``, ``radius``, ``method``.

    References
    ----------
    Rogers, D. and Hahn, M. (2010), "Extended-connectivity fingerprints",
    Journal of Chemical Information and Modeling 50(5), 742-754,
    doi:10.1021/ci100050t, section on functional-class variants.
    Paywalled at ACS, NOT read for this implementation; recorded in
    ledger/wave3/NEEDED_SOURCES.md.  The feature-class definitions and
    their order are from Gobbi, A. and Poppinger, D. (1998), "Genetic
    optimization of combinatorial libraries", Biotechnology and
    Bioengineering 61(1), 47-54, as transcribed in the RDKit source,
    Code/GraphMol/Fingerprints/FingerprintUtil.cpp lines 172-192
    (``smartsPatterns``: Donor, Acceptor, Aromatic, Halogen, Basic,
    Acidic) and applied by ``getFeatureInvariants`` at lines 211-240 with
    mask 1 << i.  The relabelling rounds are MorganGenerator.cpp lines
    395-495.  RDKit master revision fetched 2026-08-09, stored at
    library/pdf/fetched-wave3/rdkit-reference-source/.  RDKit:
    Open-Source Cheminformatics, https://www.rdkit.org.
    """
    a, bonds = _bonds(adjacency)
    rows = [r for r in features]
    if len(rows) != a:
        raise ValueError("features must have one entry per atom")
    code = []
    for r in rows:
        try:
            flags = [int(z) for z in r]
        except TypeError:
            code.append(int(r))
            continue
        if len(flags) != 6:
            raise ValueError("features rows must have 6 flags")
        v = 0
        for k in range(6):
            if flags[k]:
                v += 1 << k
        code.append(v)
    bits, cnt, ident = _morgan(a, bonds, list(code), int(radius), int(nbits))
    uniq = sorted(set(ident))
    return RichResult(payload={
        "bits": bits, "count": cnt, "nset": sum(bits),
        "identifiers": uniq, "nenv": len(ident), "featurecode": code,
        "a": a, "nbits": int(nbits), "radius": int(radius),
        "method": "FCFP4 (functional-class Morgan radius 2)"})


fcfp_4_fingerprint = fcfp4


def cheatsheet():
    return "fcfp4: functional-class fingerprint, Morgan radius 2 (FCFP4)."
