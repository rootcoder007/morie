"""Anchored tests for snpblr.snp_blup (Meuwissen 2001 BLUP / MME)."""

from morie.fn import _array_core as np

from morie.fn.snpblr import snp_blup


def test_hand_anchor_n2_m1():
    """n=2, m=1, y=(1,3), M=(0,2), lam=1.

    p = 0.5, Z = (-1, 1).  MME: [[2,0],[0,3]] [mu;u] = [4;2]
    -> mu = 2, u = 2/3, GEBV = (-2/3, 2/3).  Hand-solved.
    """
    res = snp_blup([1.0, 3.0], [[0.0], [2.0]], lam=1.0)
    assert abs(res["mu"] - 2.0) < 1e-12
    assert abs(res["u"][0] - 2.0 / 3.0) < 1e-12
    assert abs(res["estimate"][0] + 2.0 / 3.0) < 1e-12
    assert abs(res["estimate"][1] - 2.0 / 3.0) < 1e-12


def test_h2_lambda_derivation():
    """h2 = 0.5 with p = 0.5: lam = (0.5/0.5) * 2*0.5*0.5 = 0.5,
    so u = Z'y / (Z'Z + lam) = 2 / 2.5 = 0.8 (mu = 2 unchanged)."""
    res = snp_blup([1.0, 3.0], [[0.0], [2.0]], h2=0.5)
    assert abs(res["lam"] - 0.5) < 1e-12
    assert abs(res["u"][0] - 0.8) < 1e-12


def test_gls_route_equivalence():
    """Independent algebra route: with V = Z Z' + lam I,
    beta = (1'V^-1 1)^-1 1'V^-1 y (GLS) and u = Z' V^-1 (y - 1 beta)
    must reproduce the MME solution (Henderson 1975 equivalence)."""
    n, m = 8, 3
    M = np.asarray([
        [0, 1, 2], [2, 0, 1], [1, 1, 0], [2, 2, 2],
        [0, 0, 1], [1, 2, 0], [2, 1, 1], [0, 2, 2],
    ], dtype=float)
    y = np.asarray([0.3, -1.2, 0.7, 2.1, -0.4, 0.9, 1.5, -0.8])
    lam = 2.7
    res = snp_blup(y, M, lam=lam)
    p = np.sum(M, axis=0) / (2.0 * n)
    Z = M - 2.0 * p
    V = Z @ Z.T + lam * np.eye(n)
    Vi = np.linalg.inv(V)
    one = np.ones(n)
    beta = float(one @ Vi @ y) / float(one @ Vi @ one)
    u = Z.T @ Vi @ (y - one * beta)
    assert abs(res["mu"] - beta) < 1e-9
    assert float(np.max(np.abs(res["u"] - u))) < 1e-9
    assert float(np.max(np.abs(res["estimate"] - Z @ u))) < 1e-9


def test_shrinkage_monotone():
    """Larger lam shrinks marker effects towards zero."""
    M = np.asarray([
        [0, 1, 2, 1], [2, 0, 1, 0], [1, 1, 0, 2], [2, 2, 2, 1],
        [0, 0, 1, 2], [1, 2, 0, 0], [2, 1, 1, 1], [0, 2, 2, 0],
        [1, 0, 0, 2], [2, 1, 2, 1],
    ], dtype=float)
    y = np.asarray([0.3, -1.2, 0.7, 2.1, -0.4, 0.9, 1.5, -0.8, 0.2, 1.1])
    u1 = snp_blup(y, M, lam=0.5)["u"]
    u2 = snp_blup(y, M, lam=50.0)["u"]
    assert float(np.sum(np.asarray(u2) ** 2)) < float(np.sum(np.asarray(u1) ** 2))
