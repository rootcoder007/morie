"""Tests for facov.factor_analytic_covariance."""

from morie.fn import _array_core as np

from morie.fn.facov import factor_analytic_covariance


def test_facov_basic():
    """Test basic functionality with a small known example."""
    # n_env: number of environments (order of Sigma); n_factors: k factors
    n_env = 3
    n_factors = 2

    # Build Lambda as an n_env-by-k loading matrix using the lower-triangular
    # Smith et al. parameterisation: L[i][j] = vdc(i*k + j, 2 + j) + 0.5 for j<=i.
    # For n_env=3, k=2, compute L row by row using the documented vdc formula.
    # We replicate the vdc sequence j=0..5 with bases 2,2 then 2,3,3 then 2,3,4:
    #   vdc(0, 2) = 0.0  -> L[0][0] = 0.5
    #   vdc(1, 2) = 0.5  -> L[0][1] = 1.0
    #   vdc(2, 2) = 0.25 -> L[1][0] = 0.75
    #   vdc(3, 3) = 1/3  -> L[1][1] = 5/3
    #   vdc(4, 2) = 0.75 -> L[2][0] = 1.25
    #   vdc(5, 4) = 1/5  -> L[2][1] = 7/5
    L = [
        [0.5,                    1.0],
        [0.75,                   1.0 + 1.0 / 3.0],
        [1.25,                   1.0 + 1.0 / 5.0],
    ]
    psi = [0.1, 0.2, 0.3]

    result = factor_analytic_covariance(n_env, n_factors, loadings=L, psi=psi)

    # The returned object exposes a dict-like payload.
    assert isinstance(result, dict)

    # Sigma = Lambda Lambda' + diag(Psi). Compute it independently.
    Lm = [[L[i][j] for j in range(n_factors)] for i in range(n_env)]
    Sigma_expected = [[0.0] * n_env for _ in range(n_env)]
    for a in range(n_env):
        for b in range(n_env):
            s = 0.0
            for j in range(n_factors):
                s += Lm[a][j] * Lm[b][j]
            Sigma_expected[a][b] = s + (psi[a] if a == b else 0.0)

    # estimate is documented as Sigma[0][0] = L[0] . L[0] + psi[0]
    expected_estimate = Lm[0][0] * Lm[0][0] + Lm[0][1] * Lm[0][1] + psi[0]
    assert result["estimate"] == expected_estimate

    # Sigma must equal the independently computed matrix.
    for a in range(n_env):
        for b in range(n_env):
            assert result["Sigma"][a][b] == Sigma_expected[a][b]

    # Lambda and Psi are the inputs actually used.
    assert result["Lambda"] == L
    assert result["Psi"] == psi

    # n_params = n_env*k - k(k-1)/2 + n_env
    expected_n_params = n_env * n_factors - n_factors * (n_factors - 1) // 2 + n_env
    assert result["n_params"] == expected_n_params


def test_facov_edge():
    """Test edge cases: n_factors == 0 and defaults."""
    # k = 0 -> Sigma = diag(Psi); with psi=None this is the identity.
    result0 = factor_analytic_covariance(4, 0)
    assert isinstance(result0, dict)
    # Sigma is the identity matrix of size n_env
    for i in range(4):
        for j in range(4):
            expected = 1.0 if i == j else 0.0
            assert result0["Sigma"][i][j] == expected
    # estimate = Sigma[0][0] = 1.0 when psi is the default unit variances
    assert result0["estimate"] == 1.0
    # n_params = 0*4 - 0 + 4 = 4
    assert result0["n_params"] == 4
    # Lambda is an n_env-by-0 matrix; Psi defaults to all ones.
    assert result0["Lambda"] == [[] for _ in range(4)]
    assert result0["Psi"] == [1.0, 1.0, 1.0, 1.0]
