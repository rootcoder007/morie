# morie.fn -- function file (rootcoder007/morie)
r"""Binary fingerprint similarity, and why the coefficient matters.

For two fingerprints with :math:`a` and :math:`b` bits set and
:math:`c` bits set in both,

.. math:: T = \frac{c}{a + b - c}, \qquad
          D = \frac{2c}{a + b}, \qquad
          C = \frac{c}{\sqrt{ab}}, \qquad
          Tv_{\alpha\beta} = \frac{c}{\alpha(a-c) + \beta(b-c) + c}.

Tanimoto (the Jaccard coefficient on sets) is the default in
chemical searching. The others are not cosmetic variants:

*Tversky is the general case.* :math:`\alpha = \beta = 1` **is**
Tanimoto and :math:`\alpha = \beta = \tfrac{1}{2}` **is** Dice --
exact identities, and the anchor checks them rather than trusting the
algebra. Unequal weights make the coefficient asymmetric on purpose:
:math:`\alpha = 0.9, \beta = 0.1` asks "is A largely contained in B",
which is the right question for substructure-style searching and the
wrong one for symmetric clustering.

*Dice and cosine always exceed Tanimoto.* They are monotone
transformations of it for a fixed pair, so they rank identically --
but they are not interchangeable as thresholds. A 0.85 Dice cut is a
0.74 Tanimoto cut.

*Size bias.* Tanimoto systematically favours molecules with few bits
set: the denominator grows with the union, so a large molecule needs
proportionally more shared bits to reach the same score. Reporting
the raw bit counts alongside the coefficient makes that visible
rather than leaving it to be rediscovered.

**The distance is a metric.** :math:`1 - T` satisfies the triangle
inequality on binary fingerprints, which is what lets
neighbour-search and clustering be reasoned about at all; the anchor
checks it exhaustively on small fingerprints instead of citing it.

References
----------
Jaccard, P. (1912) "The distribution of the flora in the alpine
zone", *New Phytologist* 11(2), 37-50,
doi:10.1111/j.1469-8137.1912.tb05611.x, for the coefficient
:math:`c/(a+b-c)` itself.

Willett, P., Barnard, J. M. & Downs, G. M. (1998) "Chemical
similarity searching", *Journal of Chemical Information and Computer
Sciences* 38(6), 983-996, doi:10.1021/ci9800211. The survey these
formulae are taken from: the Tanimoto, Dice, cosine and Tversky
coefficients on binary fingerprints, the relations between them, and
the size bias of Tanimoto.
"""

import math

from ._richresult import RichResult

__all__ = ["fingerprint", "counts", "tanimoto", "dice", "cosine",
           "tversky", "distance", "similarity_matrix",
           "nearest_neighbours", "COEFFICIENTS",
           "tanimoto_similarity"]

COEFFICIENTS = ("tanimoto", "dice", "cosine")


def fingerprint(bits, n_bits=None):
    r"""Normalise a fingerprint to a frozen set of on-bit indices.

    Accepts a set of indices, or a 0/1 sequence of length ``n_bits``.
    """
    if isinstance(bits, (set, frozenset)):
        idx = {int(b) for b in bits}
    else:
        seq = list(bits)
        if seq and all(v in (0, 1, True, False) for v in seq) \
                and (n_bits is None or len(seq) == int(n_bits)):
            idx = {i for i, v in enumerate(seq) if v}
        else:
            idx = {int(b) for b in seq}
    if any(i < 0 for i in idx):
        raise ValueError("sasimi: a bit index cannot be negative")
    if n_bits is not None and idx and max(idx) >= int(n_bits):
        raise ValueError("sasimi: bit %d is outside a %d-bit "
                         "fingerprint" % (max(idx), int(n_bits)))
    return frozenset(idx)


def counts(fp_a, fp_b):
    r"""The three numbers every coefficient here is a function of."""
    A, B = fingerprint(fp_a), fingerprint(fp_b)
    return {"a": len(A), "b": len(B), "c": len(A & B),
            "union": len(A | B)}


def _guard(n):
    if n["a"] == 0 and n["b"] == 0:
        raise ValueError("sasimi: both fingerprints are empty, so no "
                         "similarity is defined")


def tanimoto(fp_a, fp_b):
    r"""c / (a + b - c)."""
    n = counts(fp_a, fp_b)
    _guard(n)
    return n["c"] / float(n["a"] + n["b"] - n["c"])


def dice(fp_a, fp_b):
    r"""2c / (a + b)."""
    n = counts(fp_a, fp_b)
    _guard(n)
    return 2.0 * n["c"] / float(n["a"] + n["b"])


def cosine(fp_a, fp_b):
    r"""c / sqrt(ab)."""
    n = counts(fp_a, fp_b)
    _guard(n)
    if n["a"] == 0 or n["b"] == 0:
        return 0.0
    return n["c"] / math.sqrt(float(n["a"]) * float(n["b"]))


def tversky(fp_a, fp_b, alpha=1.0, beta=1.0):
    r"""The asymmetric general case; see the module docstring."""
    al, be = float(alpha), float(beta)
    if al < 0.0 or be < 0.0:
        raise ValueError("sasimi: the Tversky weights cannot be "
                         "negative")
    n = counts(fp_a, fp_b)
    _guard(n)
    den = al * (n["a"] - n["c"]) + be * (n["b"] - n["c"]) + n["c"]
    if den == 0:
        raise ValueError("sasimi: the Tversky denominator vanishes "
                         "for alpha=%g, beta=%g on these "
                         "fingerprints" % (al, be))
    return n["c"] / float(den)


def _coef(name):
    if name == "tanimoto":
        return tanimoto
    if name == "dice":
        return dice
    if name == "cosine":
        return cosine
    raise ValueError("sasimi: coefficient must be one of %s, got %r"
                     % (", ".join(COEFFICIENTS), name))


def distance(fp_a, fp_b, coefficient="tanimoto"):
    r"""1 - similarity. A metric for Tanimoto."""
    return 1.0 - _coef(coefficient)(fp_a, fp_b)


def similarity_matrix(fps, coefficient="tanimoto"):
    r"""The full symmetric matrix of pairwise similarities."""
    f = _coef(coefficient)
    F = [fingerprint(x) for x in fps]
    if len(F) < 2:
        raise ValueError("sasimi: need at least two fingerprints")
    n = len(F)
    M = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            M[i][j] = M[j][i] = f(F[i], F[j])
    return M


def nearest_neighbours(query, fps, k=5, coefficient="tanimoto"):
    r"""The ``k`` most similar fingerprints, most similar first."""
    f = _coef(coefficient)
    q = fingerprint(query)
    scored = [(f(q, fingerprint(x)), i) for i, x in enumerate(fps)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    if int(k) < 1:
        raise ValueError("sasimi: k must be at least 1")
    return [{"index": i, "similarity": s}
            for s, i in scored[:int(k)]]


def tanimoto_similarity(fp_a, fp_b, coefficient="tanimoto",
                        alpha=None, beta=None):
    r"""Entry point: the similarity of two fingerprints, reported
    with the bit counts it was computed from.
    """
    n = counts(fp_a, fp_b)
    if alpha is not None or beta is not None:
        s = tversky(fp_a, fp_b,
                    1.0 if alpha is None else alpha,
                    1.0 if beta is None else beta)
        how = "Tversky(alpha=%g, beta=%g)" % (
            1.0 if alpha is None else alpha,
            1.0 if beta is None else beta)
    else:
        s = _coef(coefficient)(fp_a, fp_b)
        how = coefficient
    return RichResult(payload={
        "estimate": s, "similarity": s, "distance": 1.0 - s,
        "bits_a": n["a"], "bits_b": n["b"], "bits_shared": n["c"],
        "coefficient": how,
        "method": "Willett, Barnard & Downs (1998) binary "
                  "fingerprint coefficients",
    })
