"""Tests for gb10711.gibbons_ctrl_normal_asymp."""

from morie.fn import _array_core as np

from morie.fn.gb10711 import gibbons_ctrl_normal_asymp


def test_gb10711_basic():
    """Test basic functionality."""
    # Choose k = 3 populations, with documented lambda_i in (0, 1),
    # strictly positive densities, and CDF values p, F_2, F_3.
    k = 3
    lam = [0.4, 0.35, 0.25]   # strictly inside (0, 1)
    dens = [0.5, 0.8, 1.2]    # strictly positive
    pval = [0.6, 0.55, 0.7]   # p = F_1(theta_1), then F_2, F_3

    result = gibbons_ctrl_normal_asymp(lam, dens, pval)

    # The function returns a RichResult (mapping-like), not a hypothesis-test dict.
    assert hasattr(result, "keys") or isinstance(result, dict)
    assert "sigma" in result
    assert "q" in result
    assert "p" in result
    assert "k" in result
    assert "method" in result

    # sigma must be a (k-1) x (k-1) matrix of floats.
    sigma = result["sigma"]
    assert len(sigma) == k - 1
    for row in sigma:
        assert len(row) == k - 1
        for v in row:
            assert isinstance(v, float)

    # k and p are echoed back.
    assert result["k"] == k
    assert result["p"] == float(pval[0])

    # q_i = dens[i+1] / dens[0] for i = 0, ..., k-2.
    expected_q = [dens[i + 1] / dens[0] for i in range(k - 1)]
    assert list(result["q"]) == expected_q

    # Independently recompute sigma from the documented formula:
    #   sigma_{ij} = q_i * q_j * p * (1-p) / lambda_1
    #               + (i == j) * F_{i+1}(theta_1) * (1 - F_{i+1}(theta_1)) / lambda_{i+1}
    p = float(pval[0])
    lam0 = float(lam[0])
    qs = [dens[i + 1] / dens[0] for i in range(k - 1)]
    expected_sigma = []
    for i in range(k - 1):
        row = []
        for j in range(k - 1):
            v = qs[i] * qs[j] * p * (1.0 - p) / lam0
            if i == j:
                v += float(pval[i + 1]) * (1.0 - float(pval[i + 1])) / float(lam[i + 1])
            row.append(float(v))
        expected_sigma.append(row)

    for i in range(k - 1):
        for j in range(k - 1):
            assert sigma[i][j] == expected_sigma[i][j]


def test_gb10711_edge():
    """Test edge cases."""
    # Minimum valid k = 2; build a (k-1) = 1 by 1 covariance.
    lam = [0.6, 0.4]
    dens = [1.0, 2.0]
    pval = [0.3, 0.5]

    result = gibbons_ctrl_normal_asymp(lam, dens, pval)
    assert hasattr(result, "keys") or isinstance(result, dict)
    assert result["k"] == 2
    assert len(result["sigma"]) == 1
    assert len(result["sigma"][0]) == 1

    # Independent computation of the single entry.
    p = float(pval[0])
    q0 = dens[1] / dens[0]
    expected_00 = (
        q0 * q0 * p * (1.0 - p) / float(lam[0])
        + float(pval[1]) * (1.0 - float(pval[1])) / float(lam[1])
    )
    assert result["sigma"][0][0] == float(expected_00)
    assert result["q"] == [q0]
