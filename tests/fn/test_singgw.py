"""Anchored tests for singgw.single_step_h (Christensen-Lund 2010)."""

from morie.fn import _array_core as np

from morie.fn.singgw import single_step_h

# Pedigree: 1,2 founders; 3 = 1x2; 4 = 1x3.  Tabular-method A,
# computed by hand: a13 = 0.5, a23 = 0.5, a33 = 1, a14 = 0.75,
# a24 = 0.25, a34 = 0.75, a44 = 1 + 0.5*a13 = 1.25.
A = [
    [1.00, 0.00, 0.50, 0.75],
    [0.00, 1.00, 0.50, 0.25],
    [0.50, 0.50, 1.00, 0.75],
    [0.75, 0.25, 0.75, 1.25],
]


def test_limiting_case_G_equals_A11_gives_A():
    """CL2010 p.3: 'when G = A11 ... the extension in (4) [gives] A'.

    (Their statement is for no genotyping / all genotyping; G = A11
    reproducing H = A is the direct algebraic reduction of eq. 4.)
    """
    gset = [2, 3]
    A11 = [[A[i][j] for j in gset] for i in gset]
    res = single_step_h(A, A11, gset, w=0.0)
    H = np.asarray(res["estimate"])
    assert float(np.max(np.abs(H - np.asarray(A)))) < 1e-12
    Hinv = np.asarray(res["Hinv"])
    assert float(np.max(np.abs(Hinv - np.linalg.inv(np.asarray(A))))) < 1e-10


def test_limiting_case_all_genotyped():
    """Every individual genotyped: H = G exactly (CL2010 p.3)."""
    G = [
        [1.10, 0.05, 0.52, 0.70],
        [0.05, 0.95, 0.48, 0.22],
        [0.52, 0.48, 1.02, 0.71],
        [0.70, 0.22, 0.71, 1.20],
    ]
    res = single_step_h(A, G, [0, 1, 2, 3], w=0.0)
    assert float(np.max(np.abs(np.asarray(res["estimate"]) - np.asarray(G)))) < 1e-12
    # and in permuted order the blocks land back in A's order
    Gp = [[G[i][j] for j in (3, 1, 0, 2)] for i in (3, 1, 0, 2)]
    resp = single_step_h(A, Gp, [3, 1, 0, 2], w=0.0)
    assert float(np.max(np.abs(np.asarray(resp["estimate"]) - np.asarray(G)))) < 1e-12


def test_eq4_vs_eq8_inverse_identity():
    """Eqs. (4) and (8) are independent expressions; their product
    must be the identity (w > 0 guarantees invertibility)."""
    G = [[1.05, 0.70], [0.70, 1.30]]
    res = single_step_h(A, G, [2, 3], w=0.05)
    H = np.asarray(res["estimate"])
    Hinv = np.asarray(res["Hinv"])
    P = H @ Hinv
    assert float(np.max(np.abs(P - np.eye(4)))) < 1e-10


def test_w_blending_affine_identity():
    """H is affine in Gw, so H(w) = (1-w) H(w=0) + w A exactly."""
    G = [[1.05, 0.70], [0.70, 1.30]]
    w = 0.3
    H0 = np.asarray(single_step_h(A, G, [2, 3], w=0.0)["estimate"])
    Hw = np.asarray(single_step_h(A, G, [2, 3], w=w)["estimate"])
    blend = (1.0 - w) * H0 + w * np.asarray(A)
    assert float(np.max(np.abs(Hw - blend))) < 1e-12
