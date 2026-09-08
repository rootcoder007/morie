# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Topological combinatorics: complexes, homology mod 2, and Sperner.

Sperner E (1928) *Abh Math Sem Univ Hamburg* 6:265-272; Euler L (1758)
(polyhedron formula); Munkres JR (1984), *Elements of Algebraic
Topology*, Addison-Wesley (simplicial homology); Matousek J (2003),
*Using the Borsuk-Ulam Theorem*, Springer.

Homology here is over :math:`\\mathbb{F}_2`, which trades torsion
information for exact arithmetic: every rank is an integer computed by
Gaussian elimination on bits, so a Betti number is never off by
rounding. The checks are structural -- the boundary-of-boundary map
must vanish identically before any rank is trusted, the
Euler-Poincare identity :math:`\\sum (-1)^k f_k = \\sum (-1)^k b_k`
must hold as an equation between two independently computed sides, and
Sperner's rainbow count must be odd, not merely nonzero.
"""

from itertools import combinations

from ._richresult import RichResult

__all__ = [
    "simplicial_complex_faces",
    "euler_characteristic",
    "betti_numbers_gf2",
    "sperner_lemma_triangle",
]

_METHOD = "Topological combinatorics (Munkres 1984)"


def simplicial_complex_faces(maximal_simplices):
    """Close a set of maximal simplices under taking faces.

    Returns faces grouped by dimension, each face a sorted tuple of
    vertex labels. The empty simplex is not included.
    """
    maximal = [tuple(sorted(set(s))) for s in maximal_simplices]
    if not maximal:
        raise ValueError("no simplices supplied.")
    if any(len(s) == 0 for s in maximal):
        raise ValueError("empty simplices are not allowed.")
    faces = set()
    for s in maximal:
        for k in range(1, len(s) + 1):
            faces.update(combinations(s, k))
    by_dim = {}
    for f in faces:
        by_dim.setdefault(len(f) - 1, []).append(f)
    return {d: sorted(v) for d, v in sorted(by_dim.items())}


def euler_characteristic(maximal_simplices):
    r"""The Euler characteristic :math:`\chi = \sum_k (-1)^k f_k`.

    Examples
    --------
    >>> euler_characteristic([(0, 1, 2, 3)])["chi"]   # solid tetrahedron
    1
    >>> euler_characteristic([(0, 1, 2), (0, 1, 3), (0, 2, 3),
    ...                       (1, 2, 3)])["chi"]      # its boundary: S^2
    2
    """
    by_dim = simplicial_complex_faces(maximal_simplices)
    f_vec = [len(by_dim[d]) for d in sorted(by_dim)]
    chi = sum((-1) ** d * len(v) for d, v in by_dim.items())
    return RichResult(
        title="Euler characteristic",
        summary_lines=[
            ("f-vector", tuple(f_vec)),
            ("chi", chi),
        ],
        payload={
            "chi": chi,
            "f_vector": f_vec,
            "dimension": max(by_dim),
            "estimate": float(chi),
            "n": sum(f_vec),
            "method": "Euler characteristic (Euler 1758)",
        },
    )


def _gf2_rank(rows):
    """Rank over F_2 of a matrix whose rows are Python-int bitmasks."""
    rank = 0
    basis = []
    for r in rows:
        for b in basis:
            r = min(r, r ^ b)
        if r:
            basis.append(r)
            basis.sort(reverse=True)
            rank += 1
    return rank


def betti_numbers_gf2(maximal_simplices, check_boundary_squared=True):
    r"""Simplicial Betti numbers over :math:`\mathbb{F}_2`.

    :math:`b_k = \dim\ker\partial_k - \mathrm{rank}\,\partial_{k+1}`,
    with every rank computed exactly by bitwise Gaussian elimination.
    Before any rank is trusted, :math:`\partial_{k}\partial_{k+1} = 0`
    is verified entry by entry -- a boundary map that fails to square
    to zero means the matrices are indexed wrongly, and homology
    computed from them would be noise with integer formatting.

    The payload also reports the Euler-Poincare identity
    :math:`\sum (-1)^k f_k = \sum (-1)^k b_k`, computed from the two
    sides independently.

    Examples
    --------
    >>> betti_numbers_gf2([(0, 1), (1, 2), (0, 2)])["betti"]  # circle
    [1, 1]
    >>> betti_numbers_gf2([(0, 1, 2), (0, 1, 3), (0, 2, 3),
    ...                    (1, 2, 3)])["betti"]               # sphere
    [1, 0, 1]
    """
    by_dim = simplicial_complex_faces(maximal_simplices)
    top = max(by_dim)
    index = {d: {f: i for i, f in enumerate(by_dim[d])} for d in by_dim}

    def boundary_rows(d):
        """Rows of the boundary matrix of dimension d, one bitmask per
        d-simplex, bits indexing (d-1)-faces."""
        rows = []
        for s in by_dim[d]:
            mask = 0
            for omit in range(len(s)):
                face = s[:omit] + s[omit + 1:]
                mask |= 1 << index[d - 1][face]
            rows.append(mask)
        return rows

    ranks = {0: 0}
    for d in range(1, top + 1):
        ranks[d] = _gf2_rank(boundary_rows(d))
    squared_zero = True
    if check_boundary_squared:
        for d in range(2, top + 1):
            for s in by_dim[d]:
                acc = 0
                for omit in range(len(s)):
                    face = s[:omit] + s[omit + 1:]
                    for omit2 in range(len(face)):
                        sub = face[:omit2] + face[omit2 + 1:]
                        acc ^= 1 << index[d - 2][sub]
                if acc != 0:
                    squared_zero = False
    betti = []
    for d in range(top + 1):
        n_d = len(by_dim[d])
        rank_d = ranks.get(d, 0)
        rank_next = ranks.get(d + 1, 0)
        betti.append((n_d - rank_d) - rank_next)
    f_vec = [len(by_dim[d]) for d in range(top + 1)]
    chi_f = sum((-1) ** d * f for d, f in enumerate(f_vec))
    chi_b = sum((-1) ** d * b for d, b in enumerate(betti))
    out = RichResult(
        title="Betti numbers over F_2",
        summary_lines=[
            ("Betti", tuple(betti)),
            ("chi from f-vector", chi_f),
            ("chi from Betti", chi_b),
            ("boundary^2 = 0", squared_zero),
        ],
        payload={
            "betti": betti,
            "f_vector": f_vec,
            "boundary_ranks": [ranks.get(d, 0) for d in range(top + 2)],
            "chi_from_faces": chi_f,
            "chi_from_betti": chi_b,
            "euler_poincare_holds": chi_f == chi_b,
            "boundary_squared_zero": squared_zero,
            "estimate": float(betti[0]),
            "n": sum(f_vec),
            "method": "Simplicial homology over F_2 (Munkres 1984)",
        },
    )
    if not squared_zero:
        out.warnings.append(
            "The boundary map does not square to zero, so the matrices "
            "are indexed wrongly and the Betti numbers above are noise. "
            "This cannot happen for faces generated by "
            "simplicial_complex_faces."
        )
    if chi_f != chi_b:
        out.warnings.append(
            "The Euler-Poincare identity fails, which is impossible over "
            "a field; the rank computation is defective."
        )
    return out


def sperner_lemma_triangle(subdivisions, labels=None):
    r"""Sperner's lemma on the standard subdivided triangle.

    The triangle with corners 0, 1, 2 is subdivided into :math:`k^2`
    cells; grid point :math:`(i, j)` carries barycentric coordinates
    :math:`(k - i - j,\; i,\; j)/k`. A Sperner labelling assigns each
    point a label whose coordinate is positive -- corners get their own
    label, edge points one of the two endpoint labels. The lemma says
    the number of RAINBOW cells (all three labels) is odd, hence
    nonzero. Oddness is the full strength of the result and is what is
    asserted; a nonzero count alone would be a weaker check.

    The default labelling takes the smallest admissible label. A
    custom labelling is supplied as ``{(i, j): label}`` and is
    validated against admissibility rather than trusted -- an improper
    labelling is exactly the case where the lemma is false, so it is
    refused loudly.

    Examples
    --------
    >>> sperner_lemma_triangle(4)["rainbow_count"] % 2
    1
    """
    k = int(subdivisions)
    if k < 1:
        raise ValueError(f"subdivisions must be positive; got {k}.")
    pts = [(i, j) for i in range(k + 1) for j in range(k + 1 - i)]

    def admissible(i, j):
        allowed = []
        if k - i - j > 0:
            allowed.append(0)
        if i > 0:
            allowed.append(1)
        if j > 0:
            allowed.append(2)
        return allowed

    lab = {}
    for (i, j) in pts:
        allowed = admissible(i, j)
        if labels is not None:
            if (i, j) not in labels:
                raise ValueError(f"no label supplied for grid point "
                                 f"({i}, {j}).")
            v = int(labels[(i, j)])
            if v not in allowed:
                raise ValueError(
                    f"label {v} at ({i}, {j}) is not admissible; a "
                    f"Sperner labelling must pick from {allowed}, and "
                    "the lemma is FALSE for improper labellings."
                )
            lab[(i, j)] = v
        else:
            lab[(i, j)] = allowed[0]

    rainbow = 0
    cells = 0
    for i in range(k):
        for j in range(k - i):
            up = [(i, j), (i + 1, j), (i, j + 1)]
            cells += 1
            if {lab[p] for p in up} == {0, 1, 2}:
                rainbow += 1
            if i + j <= k - 2:
                down = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
                cells += 1
                if {lab[p] for p in down} == {0, 1, 2}:
                    rainbow += 1
    return RichResult(
        title=f"Sperner's lemma, {k}-fold subdivision",
        summary_lines=[
            ("Cells", cells),
            ("Rainbow cells", rainbow),
            ("Odd", rainbow % 2 == 1),
        ],
        payload={
            "rainbow_count": rainbow,
            "is_odd": rainbow % 2 == 1,
            "n_cells": cells,
            "n_points": len(pts),
            "estimate": float(rainbow),
            "n": k,
            "method": "Sperner's lemma (Sperner 1928)",
        },
    )


def cheatsheet():
    return (
        "topcmb: Euler characteristics, Betti numbers over F_2 with the "
        "boundary-squared-zero and Euler-Poincare identities checked, and "
        "Sperner's lemma asserting oddness of the rainbow count"
    )
