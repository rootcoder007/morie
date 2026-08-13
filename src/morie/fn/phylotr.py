# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neighbor-joining phylogenetic tree construction (Saitou-Nei 1987)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["phylotr", "phylogenetic_tree"]


def _sij(D, m, i, j):
    # Saitou & Nei (1987) eq. (4): sum of branch lengths when i and j
    # are joined,
    #   S_ij = [sum_{k != i,j} (D_ik + D_jk)] / (2 (m - 2))
    #        + D_ij / 2
    #        + [sum_{k < l; k,l != i,j} D_kl] / (m - 2).
    t1 = 0.0
    for k in range(m):
        if k != i and k != j:
            t1 += D[i][k] + D[j][k]
    t3 = 0.0
    for k in range(m):
        for l in range(k + 1, m):
            if k != i and k != j and l != i and l != j:
                t3 += D[k][l]
    return t1 / (2.0 * (m - 2)) + D[i][j] / 2.0 + t3 / (m - 2.0)


def phylotr(distance, labels=None):
    """
    Neighbor-joining tree from a distance matrix, as published in
    Saitou and Nei (1987).

    Starting from a star tree, every pair (i, j) of current OTUs is
    scored by the total branch length S_ij of the tree in which i and
    j are joined (eq. 4); the pair with smallest S_ij is joined.
    Branch lengths for the joined pair follow eqs. (6a)-(6b):
    L_i = (D_ij + D_iZ - D_jZ)/2 with D_iZ the average distance from
    i to all remaining OTUs, and the combined OTU's distances are the
    averages D_(ij)k = (D_ik + D_jk)/2 of eq. (5). The cycle repeats
    until three OTUs remain, whose three branch lengths follow from
    the pairwise distances. Ties in S_ij are broken by the first pair
    in row-major scan order (pinned identically in the R mirror).

    Note: this is the 1987 algorithm as printed, including the
    averaged distance update of eq. (5). It selects the same
    topology as the Studier-Keppler (1988) O(n^3) criterion used by
    modern NJ, which selects the same pair; this module uses the
    Saitou & Nei eq. (4) form
    software on additive data, but intermediate distances and the
    printed S values follow the original paper.

    Parameters
    ----------
    distance : array-like, shape (n, n)
        Symmetric distance matrix.
    labels : sequence of str, optional
        OTU names (default "1".."n" as in the paper's example).

    Returns
    -------
    result : RichResult
        Keys: joins (per cycle: label_i, label_j, new label, L_i,
        L_j, S_ij), s0 (eq. 1 star-tree total length), final_labels,
        final_lengths (the last three branches), n, method.

    References
    ----------
    Studier, J. A. and Keppler, K. J. (1988), "A note on the
    neighbor-joining algorithm of Saitou and Nei", Molecular
    Biology and Evolution 5(6), 729-731. Letter to the Editor,
    p. 729: S_ij = (N - 2) D_ij - R_i - R_j with R_i = sum_k D_ik.
    This module does NOT compute that; :func:`_sij` implements
    Saitou & Nei eq. (4) directly. The two criteria are minimised by
    the same pair, so the topology is identical -- that equivalence
    is asserted in the tests -- and Studier & Keppler is named here
    because it is the O(n^3) form modern implementations use, not
    because it is the form used below.

    Saitou, N. and Nei, M. (1987), "The neighbor-joining method: a
    new method for reconstructing phylogenetic trees", Molecular
    Biology and Evolution 4(4), 406-425. Equations (1), (4), (5),
    (6a), (6b), pp. 408-409; worked example Table 1/Table 2 and
    Figure 3, pp. 410-411. Local source:
    library/pdf/fetched-wave3/Saitou-Nei-1987-NeighborJoining-MBE.pdf.
    """
    Dm = np.asarray(distance, dtype=float)
    if Dm.ndim != 2 or Dm.shape[0] != Dm.shape[1]:
        raise ValueError("distance must be a square matrix")
    n = Dm.shape[0]
    if n < 4:
        raise ValueError("need at least 4 OTUs")
    D = [[float(Dm[i, j]) for j in range(n)] for i in range(n)]
    labs = ([str(x) for x in labels] if labels is not None
            else [str(i + 1) for i in range(n)])
    if len(labs) != n:
        raise ValueError("labels length must match matrix size")
    # eq. (1): S_0 = sum_{i<j} D_ij / (n - 1)
    tot = sum(D[i][j] for i in range(n) for j in range(i + 1, n))
    s0 = tot / (n - 1.0)
    joins = []
    m = n
    while m > 3:
        best = None
        bi = bj = -1
        for i in range(m):
            for j in range(i + 1, m):
                s = _sij(D, m, i, j)
                if best is None or s < best:
                    best, bi, bj = s, i, j
        # eqs. (6a)-(6b)
        diz = sum(D[bi][k] for k in range(m)
                  if k != bi and k != bj) / (m - 2.0)
        djz = sum(D[bj][k] for k in range(m)
                  if k != bi and k != bj) / (m - 2.0)
        li = (D[bi][bj] + diz - djz) / 2.0
        lj = (D[bi][bj] + djz - diz) / 2.0
        new_lab = "(" + labs[bi] + "-" + labs[bj] + ")"
        joins.append({"a": labs[bi], "b": labs[bj], "new": new_lab,
                      "La": li, "Lb": lj, "S": best})
        # eq. (5) averaged update; build the reduced matrix
        keep = [k for k in range(m) if k != bi and k != bj]
        Dn = [[0.0] * (m - 1) for _ in range(m - 1)]
        for a2, ka in enumerate(keep):
            for b2, kb in enumerate(keep):
                Dn[a2][b2] = D[ka][kb]
        for a2, ka in enumerate(keep):
            v = (D[bi][ka] + D[bj][ka]) / 2.0
            Dn[a2][m - 2] = v
            Dn[m - 2][a2] = v
        D = Dn
        labs = [labs[k] for k in keep] + [new_lab]
        m -= 1
    la = (D[0][1] + D[0][2] - D[1][2]) / 2.0
    lb = (D[0][1] + D[1][2] - D[0][2]) / 2.0
    lc = (D[0][2] + D[1][2] - D[0][1]) / 2.0
    return RichResult(payload={
        "joins": joins,
        "s0": s0,
        "final_labels": list(labs),
        "final_lengths": np.asarray([la, lb, lc]),
        "n": n,
        "method": "neighbor joining, original 1987 algorithm (Saitou-Nei)",
    })


phylogenetic_tree = phylotr


def cheatsheet():
    return ("phylotr(D, labels=None) -> Saitou-Nei (1987) neighbor "
            "joining: S_ij criterion, eq 6a/6b branch lengths.")
