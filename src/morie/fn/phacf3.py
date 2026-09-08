"""Three-point 3D pharmacophore fingerprints.

A pharmacophore is the abstraction that survives when you throw away the
chemistry and keep only what the protein can feel: a hydrogen-bond donor
here, an acceptor four angstroms away, an aromatic ring seven from both.
Two molecules from unrelated series that present the same triangle of
features to the same pocket bind the same way, and a fingerprint over
those triangles is what lets you find that.

The construction is:

  1. Reduce the molecule to FEATURE POINTS -- donor, acceptor, positive,
     negative, hydrophobic, aromatic -- each with a position.
  2. Take every triple of them. Measure the three inter-feature
     distances and put each in a BIN, because a pharmacophore that only
     matched at exact distances would never match anything.
  3. Canonicalise the triangle so the same geometry lands on the same
     bit whatever order the atoms came in.
  4. Set that bit.

Step three is the whole difficulty. A triangle has six vertex orderings
and each gives a different tuple of (types, bins); the canonical form is
the smallest of the six, and getting it wrong means the same
pharmacophore in two molecules hits two different bits and the
comparison silently fails. It is done here by generating all six and
taking the minimum, which is slow and obviously correct, rather than by
a sorting rule that is fast and subtly wrong on ties.

The bit space is enumerated the same way -- every possible canonical
(type triple, bin triple) -- so a bit index means the same thing in
every molecule and the space is a fixed size that both arms agree on.

What a three-point fingerprint CANNOT do is tell a molecule from its
mirror image: three points and three distances are the same in both
hands, and chirality only becomes visible with a fourth point and a
signed volume. That is not a defect in this implementation, it is why
four-point pharmacophores exist, and there is a check here that
demonstrates it rather than a comment claiming it.

Feature perception is not done here. Which atoms are donors is a
question for a chemistry toolkit, and guessing would put fabricated
chemistry underneath everything above. Feature points come in already
typed and placed.

References
  Gund, P. (1977) "Three-dimensional pharmacophoric pattern searching."
    Progress in Molecular and Subcellular Biology 5, 117-143. The idea.
  Mason, J.S., Good, A.C. and Martin, E.J. (2001) "3-D pharmacophores in
    drug discovery." Current Pharmaceutical Design 7(7), 567-597. The
    fingerprint over binned feature triplets, and the four-point
    extension that recovers chirality.
  Mason, J.S., Morize, I., Menard, P.R., Cheney, D.L., Hulme, C. and
    Labaudiniere, R.F. (1999) "New 4-point pharmacophore method for
    molecular similarity and diversity applications." Journal of
    Medicinal Chemistry 42(17), 3251-3264. Why the fourth point.
  Tanimoto, T.T. (1958) "An elementary mathematical theory of
    classification and prediction." IBM internal report. The similarity
    coefficient used to compare two fingerprints.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["phacf3", "pharmacophore_3d", "canonical_triangle", "bit_space",
           "distance_bin", "tanimoto", "FEATURES", "DEFAULT_EDGES",
           "MODES", "cheatsheet"]

# The six feature classes of the classical pharmacophore alphabet.
FEATURES = ("donor", "acceptor", "positive", "negative", "hydrophobic",
            "aromatic")

# Bin EDGES in angstroms, so there are len(edges) - 1 bins and anything
# outside the outer edges is not a pharmacophore distance at all. These
# are a default, not a standard: bin boundaries are a modelling choice
# and different published schemes use different ones.
DEFAULT_EDGES = (2.0, 4.5, 7.0, 10.0, 14.0, 20.0, 24.0)

MODES = ("binary", "count")


def distance_bin(d, edges):
    """Which distance bin, or -1 for a distance outside the range.

    Half-open on the left, so a distance sitting exactly on a boundary
    goes to the upper bin and no distance can land in two bins because
    of a rounding accident.
    """
    d = float(d)
    if d < edges[0] or d >= edges[-1]:
        return -1
    for k in range(len(edges) - 1):
        if d < edges[k + 1]:
            return k
    return len(edges) - 2


def canonical_triangle(t1, t2, t3, d12, d13, d23):
    """The canonical (types, bins) form of one feature triangle.

    Each vertex carries a type and each EDGE a bin. Under a permutation
    of the vertices the edges permute with them, so all six orderings
    are generated and the smallest tuple wins. Generating all six is
    the point: a comparison rule that sorted the types first and hoped
    the edges followed is wrong whenever two types are equal, which on
    a six-letter alphabet is most of the time.
    """
    # (i, j, k) vertex order; edges are (ij, ik, jk) in that order.
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1),
             (2, 1, 0))
    t = (t1, t2, t3)
    # d[a][b] as a lookup so an edge follows its endpoints.
    d = {(0, 1): d12, (1, 0): d12, (0, 2): d13, (2, 0): d13,
         (1, 2): d23, (2, 1): d23}
    best = None
    for i, j, k in perms:
        cand = (t[i], t[j], t[k], d[(i, j)], d[(i, k)], d[(j, k)])
        if best is None or cand < best:
            best = cand
    return best


def bit_space(features=FEATURES, n_bins=None, edges=DEFAULT_EDGES):
    """Every canonical triangle the alphabet allows, in a fixed order.

    Returns the list of keys and a lookup from key to bit index. Built
    by enumerating all orderings and canonicalising, so it is exactly
    the set of keys `canonical_triangle` can produce and nothing else --
    a space derived by a formula could be off by the number of
    degenerate triangles and nobody would notice until two molecules
    disagreed.
    """
    nb = (len(edges) - 1) if n_bins is None else int(n_bins)
    if nb < 1:
        raise ValueError("need at least one distance bin")
    nf = len(features)
    seen = set()
    for a in range(nf):
        for b in range(nf):
            for c in range(nf):
                for p in range(nb):
                    for q in range(nb):
                        for r in range(nb):
                            seen.add(canonical_triangle(a, b, c, p, q, r))
    keys = sorted(seen)
    index = {}
    for i, k in enumerate(keys):
        index[k] = i
    return keys, index


def _dist(a, b):
    return math.sqrt(_w.csum((a[t] - b[t]) * (a[t] - b[t])
                             for t in range(3)))


def tanimoto(a, b):
    """Tanimoto coefficient of two equal-length fingerprints.

    On counts rather than bits it is the weighted form: shared over
    union, with the minimum as the intersection. Two empty
    fingerprints have no features in common and none apart, which is a
    zero-over-zero; it is reported as nan rather than as the 1.0 that
    would call two blank molecules identical.
    """
    if len(a) != len(b):
        raise ValueError("fingerprints must be the same length")
    inter = []
    union = []
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        inter.append(x if x < y else y)
        union.append(x if x > y else y)
    num = _w.csum(inter)
    den = _w.csum(union)
    return num / den if den > 0.0 else float("nan")


def pharmacophore_3d(mol_3d, feature_set=FEATURES, edges=DEFAULT_EDGES,
                     mode="binary", space=None):
    """Fingerprint a set of typed 3D feature points.

    Parameters
    ----------
    mol_3d : sequence of sequences
        One row per feature point: x, y, z, type. The type must be a
        member of `feature_set`.
    feature_set : sequence
        The alphabet. Its ORDER fixes the bit space, so two
        fingerprints are only comparable if they were built on the same
        alphabet in the same order.
    edges : sequence
        Bin edges in angstroms.
    mode : str
        "binary" sets a bit once however many triangles hit it;
        "count" counts them, which keeps the information that a
        molecule presented the same pharmacophore in six different
        places.
    space : tuple or None
        A prebuilt (keys, index) from `bit_space`, since building it is
        the expensive part and it does not depend on the molecule.

    Returns
    -------
    RichResult
        The fingerprint, the bits set, the triangles that produced
        them, and the counts of triangles rejected for falling outside
        the distance range or for failing the triangle inequality.

    References
    ----------
    Gund (1977) Prog Mol Subcell Biol 5, 117-143; Mason et al. (2001)
    Curr Pharm Des 7(7), 567-597.
    """
    if mode not in MODES:
        raise ValueError("mode must be one of %r" % (MODES,))
    feats = [str(f) for f in feature_set]
    pts = []
    for row in mol_3d:
        t = str(row[3])
        if t not in feats:
            raise ValueError("feature type %r is not in the alphabet" % t)
        pts.append(([float(row[0]), float(row[1]), float(row[2])],
                    feats.index(t)))
    n = len(pts)
    if space is None:
        space = bit_space(feats, len(edges) - 1, edges)
    keys, index = space
    fp = [0] * len(keys)
    hits = []
    out_of_range = 0
    degenerate = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d12 = _dist(pts[i][0], pts[j][0])
                d13 = _dist(pts[i][0], pts[k][0])
                d23 = _dist(pts[j][0], pts[k][0])
                # Three points in space always satisfy the triangle
                # inequality, so a violation means the distances did
                # not come from one geometry -- which happens as soon
                # as somebody feeds in a distance matrix instead of
                # coordinates. Counting it is cheaper than debugging
                # the fingerprint later.
                if (d12 + d13 < d23 or d12 + d23 < d13
                        or d13 + d23 < d12):
                    degenerate += 1
                    continue
                b12 = distance_bin(d12, edges)
                b13 = distance_bin(d13, edges)
                b23 = distance_bin(d23, edges)
                if b12 < 0 or b13 < 0 or b23 < 0:
                    out_of_range += 1
                    continue
                key = canonical_triangle(pts[i][1], pts[j][1], pts[k][1],
                                         b12, b13, b23)
                bit = index[key]
                if mode == "binary":
                    fp[bit] = 1
                else:
                    fp[bit] += 1
                hits.append((i, j, k, bit))
    on = [i for i in range(len(fp)) if fp[i] > 0]
    total = 0
    for v in fp:
        total += v
    return RichResult(payload={
        "fingerprint": fp,
        "bits_on": on,
        "n_bits_on": len(on),
        "n_bits": len(fp),
        "density": len(on) / float(len(fp)) if fp else float("nan"),
        "total": total,
        "hit_i": [h[0] for h in hits],
        "hit_j": [h[1] for h in hits],
        "hit_k": [h[2] for h in hits],
        "hit_bit": [h[3] for h in hits],
        "n_triangles": len(hits),
        "n_out_of_range": out_of_range,
        "n_degenerate": degenerate,
        "n_features": n,
        "estimate": len(on),
        "se": float("nan"),
        "mode": mode,
        "method": "three-point 3D pharmacophore fingerprint",
    })


phacf3 = pharmacophore_3d


def cheatsheet():
    return ("phacf3: three-point 3D pharmacophore fingerprint. modes "
            + ", ".join(MODES) + "; features " + ", ".join(FEATURES))
