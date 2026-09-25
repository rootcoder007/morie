"""Tests for vardec.var_variance_decomp (alias of fevdc.fevd).

Reference values are statsmodels 0.15 VARResults.fevd(6).decomp for a
VAR(1) fitted without trend; the coefficient and residual covariance
matrices below are that fit's coefs[0] and sigma_u, printed with repr.
"""

import pytest

from morie.fn.vardec import var_variance_decomp

A = [[0.5023565538248529, 0.0015248950399881778, 0.21018907217040123],
     [0.1642526436976442, 0.3464596057096218, 0.00013660866549075612],
     [-0.01975532109063659, 0.3269885491453032, 0.3773260403130584]]
S = [[0.9317784078463273, 0.30427829329185896, 0.13390431790287197],
     [0.30427829329185896, 0.897614910419649, 0.16368538578348715],
     [0.13390431790287197, 0.16368538578348715, 0.45332342134193826]]


def test_vardec_basic():
    """decomposition[h][i] matches statsmodels decomp[i, h]."""
    r = var_variance_decomp(A, S, periods=6)
    d = r.extra["decomposition"]
    for got, ref in zip(d[5][2], [0.09364396618344853, 0.27197610339055983, 0.6343799304259916]):
        assert got == pytest.approx(ref, rel=1e-12)
    for got, ref in zip(d[3][1], [0.19049763736731412, 0.8083947119302327, 0.0011076507024529285]):
        assert got == pytest.approx(ref, rel=1e-12)
    assert d[0][0] == [1.0, 0.0, 0.0]
    for h in range(7):
        for i in range(3):
            assert sum(d[h][i]) == pytest.approx(1.0, abs=1e-15)


def test_vardec_edge():
    """A non-square coefficient matrix is rejected."""
    with pytest.raises(ValueError, match="var_coefficients must be square"):
        var_variance_decomp([[0.5, 0.1, 0.0]], S)
