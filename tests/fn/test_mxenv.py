"""Tests for mxenv.multi_env_model (GxE GBLUP, MVSML eq. 5.4)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mxenv import multi_env_model

# 2 lines x 2 environments, each cell observed twice (n = 8)
LINE = [0, 0, 1, 1, 0, 0, 1, 1]
ENV = [0, 0, 0, 0, 1, 1, 1, 1]
Y = [3.1, 2.8, 4.0, 4.4, 2.2, 2.6, 5.1, 4.7]
G = [[1.0, 0.3], [0.3, 1.0]]
SE = [[0.8, 0.2], [0.2, 0.5]]
XE = [[float(e)] for e in ENV]          # environment effect beyond mu
ZL = [[1.0 if LINE[i] == j else 0.0 for j in range(2)] for i in range(8)]
# Sigma_E (x) G orders the GxE effects environment-major: column e*J + j
ZEL = [[1.0 if (ENV[i] == e and LINE[i] == j) else 0.0 for e in range(2) for j in range(2)]
       for i in range(8)]


def _gls_blup(s2g, s2e):
    X = np.array([[1.0] + r for r in XE])
    K = [[SE[a // 2][b // 2] * G[a % 2][b % 2] for b in range(4)] for a in range(4)]
    ZLm, ZELm = np.array(ZL), np.array(ZEL)
    V = s2g * (ZLm @ np.array(G) @ ZLm.T) + ZELm @ np.array(K) @ ZELm.T + s2e * np.eye(8)
    Vi = np.linalg.inv(V)
    y = np.array(Y)
    beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    r = Vi @ (y - X @ beta)
    return (beta.tolist(), (s2g * (np.array(G) @ ZLm.T @ r)).tolist(),
            (np.array(K) @ ZELm.T @ r).tolist())


def test_mxenv_basic():
    """Mixed-model solution from V = s2g Z_L G Z_L' + Z_EL (Sigma_E (x) G)
    Z_EL' + s2e I: beta = GLS, b = Cov(b, y) V^-1 (y - X beta) --
    Henderson's equations give the same BLUE/BLUP."""
    beta, b1, b2 = _gls_blup(0.7, 0.4)
    r = multi_env_model(Y, XE, ZL, ZEL, G, 0.7, SE, sigma2_e=0.4)
    assert isinstance(r, dict)
    assert [float(v) for v in r["beta"]] == pytest.approx(beta, rel=1e-10)
    assert [float(v) for v in r["b_lines"]] == pytest.approx(b1, rel=1e-10, abs=1e-12)
    assert [float(v) for v in r["b_gxe"]] == pytest.approx(b2, rel=1e-10, abs=1e-12)
    assert r["n"] == 8


def test_mxenv_edge():
    """A huge residual variance shrinks every random effect to zero: with
    s2e = 1e8, b ~ Cov(b, y) (y - X beta) / s2e, and Cov(b, y) has
    entries <= 0.8 over two observations with residuals below 2, so
    |b| <~ 0.8 * 2 * 2 / 1e8 = 3.2e-8, inside 1e-6."""
    r = multi_env_model(Y, XE, ZL, ZEL, G, 0.7, SE, sigma2_e=1e8)
    assert max(abs(float(v)) for v in list(r["b_lines"]) + list(r["b_gxe"])) < 1e-6
