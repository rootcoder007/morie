"""Tests for effsiz.effective_sample_size."""

from morie.fn import _array_core as np

from morie.fn.effsiz import effective_sample_size


def test_effsiz_basic():
    """Test basic functionality with a directly supplied deff."""
    n = 100
    deff = 2.5
    result = effective_sample_size(n, deff)

    # Documented return keys
    assert "n_effective" in result
    assert "deff" in result
    assert "se_inflation" in result
    assert "information_lost" in result

    # Formula: n_eff = n / deff
    expected_n_eff = n / deff
    assert abs(result["n_effective"] - expected_n_eff) < 1e-9

    # deff should be returned as supplied
    assert abs(result["deff"] - deff) < 1e-9

    # se_inflation = sqrt(deff)
    assert abs(result["se_inflation"] - np.sqrt(deff)) < 1e-9

    # information_lost = 1 - 1/deff
    assert abs(result["information_lost"] - (1.0 - 1.0 / deff)) < 1e-9


def test_effsiz_clustered():
    """Test clustered design using icc and cluster_size (Kish 1965 formula)."""
    n = 5000
    icc = 0.05
    cluster_size = 50

    result = effective_sample_size(n, icc=icc, cluster_size=cluster_size)

    # deff = 1 + (m_bar - 1) * rho
    expected_deff = 1.0 + (cluster_size - 1.0) * icc
    assert abs(result["deff"] - expected_deff) < 1e-9

    expected_n_eff = n / expected_deff
    assert abs(result["n_effective"] - expected_n_eff) < 1e-9

    assert abs(result["se_inflation"] - np.sqrt(expected_deff)) < 1e-9


def test_effsiz_edge():
    """Test edge case: no clustering means deff=1 and n_eff=n."""
    n = 1000
    icc = 0.0
    cluster_size = 20

    result = effective_sample_size(n, icc=icc, cluster_size=cluster_size)

    assert abs(result["deff"] - 1.0) < 1e-9
    assert abs(result["n_effective"] - n) < 1e-9
    assert abs(result["se_inflation"] - 1.0) < 1e-9
    assert abs(result["information_lost"] - 0.0) < 1e-9
