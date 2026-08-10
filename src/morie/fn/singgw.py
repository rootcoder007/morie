# morie.fn -- function file (rootcoder007/morie)
"""Single-step GBLUP relationship matrix H and its inverse."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["single_step_h"]


def single_step_h(A, G, genotyped, w=0.0):
    """Single-step relationship matrix H combining pedigree and genomics.

    Given the pedigree numerator relationship matrix A over all
    individuals, a genomic relationship matrix G over the genotyped
    subset (block "1"), and the non-genotyped remainder (block "2"),
    the joint matrix of Christensen and Lund (2010), eq. (4) (equal to
    the H of Legarra, Aguilar and Misztal 2009 and Aguilar et al.
    2010, with genotyped individuals as block 1), is

        H_11 = Gw
        H_12 = Gw A11^-1 A12
        H_21 = A21 A11^-1 Gw
        H_22 = A22 + A21 A11^-1 (Gw - A11) A11^-1 A12

    where Gw = (1 - w) G + w A11 blends in a polygenic fraction w
    (Christensen and Lund 2010, "The combined genetic effect",
    G_w = (1 - w) G* + w A11; w = 0 gives the raw eq. 4).  Its inverse
    is the sparse update of their eq. (8) (eq. 6 at w = 0):

        H^-1 = A^-1 + [ Gw^-1 - A11^-1   0 ]
                      [ 0                0 ].

    Limiting cases (both exercised in the tests): G = A11 gives
    H = A exactly, and with every individual genotyped H = Gw.

    Parameters
    ----------
    A : (n, n) array-like
        Pedigree numerator relationship matrix over all individuals.
    G : (q, q) array-like
        Genomic relationship matrix over the genotyped individuals, in
        the order given by ``genotyped``.
    genotyped : sequence of int
        Indices (0-based) of the genotyped individuals within A.
    w : float in [0, 1)
        Polygenic blending weight; eq. (8) requires w > 0 for
        guaranteed invertibility when G is singular.

    Returns
    -------
    RichResult
        Keys ``estimate`` (H, full n x n), ``Hinv``, ``Gw``,
        ``genotyped``, ``w``, ``n``, ``n_genotyped``, ``method``.

    References
    ----------
    Christensen, O. F. and Lund, M. S. (2010). Genomic prediction when
    some animals are not genotyped. Genetics Selection Evolution 42,
    2; eqs. (4), (6), (8) and sec. "The combined genetic effect", p. 3
    (fetched-wave3 PDF ChristensenLund-2010).
    Legarra, A., Aguilar, I. and Misztal, I. (2009). A relationship
    matrix including full pedigree and genomic information. Journal of
    Dairy Science 92(9), 4656-4663.
    Aguilar, I., Misztal, I., Johnson, D. L., Legarra, A., Tsuruta, S.
    and Lawlor, T. J. (2010). Hot topic: A unified approach to utilize
    phenotypic, full pedigree, and genomic information for genetic
    evaluation of Holstein final score. Journal of Dairy Science
    93(2), 743-752.
    """
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    gset = [int(i) for i in genotyped]
    q = len(gset)
    if q == 0:
        raise ValueError("need at least one genotyped individual")
    if len(set(gset)) != q or min(gset) < 0 or max(gset) >= n:
        raise ValueError("genotyped indices must be unique and within A")
    if G.shape != (q, q):
        raise ValueError("G must be q x q in the order of genotyped")
    w = float(w)
    if not (0.0 <= w < 1.0):
        raise ValueError("w must be in [0, 1)")
    others = [i for i in range(n) if i not in set(gset)]
    if not others:
        # every individual genotyped: H = Gw exactly (limiting case of
        # eq. 4, Christensen and Lund 2010 p. 3)
        inv_idx = [0] * n
        for pos, i in enumerate(gset):
            inv_idx[i] = pos
        Gw = (1.0 - w) * G + w * A[np.ix_(gset, gset)]
        H = Gw[np.ix_(inv_idx, inv_idx)]
        Hinv = np.linalg.inv(Gw)[np.ix_(inv_idx, inv_idx)]
        return RichResult(payload={
            "estimate": H, "Hinv": Hinv, "Gw": Gw,
            "genotyped": gset, "w": w, "n": int(n), "n_genotyped": int(q),
            "method": "Single-step H (Christensen-Lund 2010 eq. 4/8; ssGBLUP)",
        })
    idx = gset + others
    # permuted A with genotyped first
    Ap = A[np.ix_(idx, idx)]
    A11 = Ap[:q, :q]
    A12 = Ap[:q, q:]
    A21 = Ap[q:, :q]
    A22 = Ap[q:, q:]
    Gw = (1.0 - w) * G + w * A11
    A11inv = np.linalg.inv(A11)
    B = A11inv @ A12          # q x (n-q)
    H11 = Gw
    H12 = Gw @ B
    H21 = B.T @ Gw
    H22 = A22 + B.T @ (Gw - A11) @ B
    Hp = np.concatenate([
        np.concatenate([H11, H12], axis=1),
        np.concatenate([H21, H22], axis=1),
    ], axis=0)
    # undo the permutation
    inv_idx = [0] * n
    for pos, i in enumerate(idx):
        inv_idx[i] = pos
    H = Hp[np.ix_(inv_idx, inv_idx)]
    # inverse via eq. (8)
    Ainv = np.linalg.inv(Ap)
    Gwinv = np.linalg.inv(Gw)
    Hinvp = Ainv.copy()
    Hinvp[:q, :q] = Hinvp[:q, :q] + Gwinv - A11inv
    Hinv = Hinvp[np.ix_(inv_idx, inv_idx)]
    return RichResult(payload={
        "estimate": H, "Hinv": Hinv, "Gw": Gw,
        "genotyped": gset, "w": w, "n": int(n), "n_genotyped": int(q),
        "method": "Single-step H (Christensen-Lund 2010 eq. 4/8; ssGBLUP)",
    })


def cheatsheet():
    return "singgw: ssGBLUP H from A and G (Christensen-Lund 2010 eq 4); Hinv = Ainv + [Gw^-1 - A11^-1]."


# compact alias per ledger/NAMING.md; single_step_gblup is the
# pre-existing exported name, kept as an alias
singlestepH = single_step_h
singgw = single_step_h
single_step_gblup = single_step_h
