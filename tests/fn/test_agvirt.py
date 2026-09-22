"""Tests for agvirt.alphazero_virtual_loss."""

from morie.fn import _array_core as np

from morie.fn.agvirt import alphazero_virtual_loss


def test_agvirt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1, 100)
    N = rng.integers(0, 10, 100).astype(float)
    pending = rng.integers(0, 5, 100).astype(float)
    nvl = 1.0
    result = alphazero_virtual_loss(W, N, pending, nvl=nvl)
    assert isinstance(result, dict)
    assert "Q" in result
    assert "N" in result
    assert "W" in result
    assert "Qclean" in result
    assert "k" in result
    assert "nvl" in result
    assert result["k"] == 100
    assert result["nvl"] == nvl
    # Independent computation of the documented formula
    expected_Nv = [N[i] + nvl * pending[i] for i in range(100)]
    expected_Wv = [W[i] - nvl * pending[i] for i in range(100)]
    expected_Q = [
        0.0 if expected_Nv[i] == 0.0 else expected_Wv[i] / expected_Nv[i]
        for i in range(100)
    ]
    expected_Qc = [
        0.0 if N[i] == 0.0 else W[i] / N[i] for i in range(100)
    ]
    for i in range(100):
        assert result["N"][i] == expected_Nv[i]
        assert result["W"][i] == expected_Wv[i]
        assert result["Q"][i] == expected_Q[i]
        assert result["Qclean"][i] == expected_Qc[i]


def test_agvirt_edge():
    """Test edge cases."""
    W = [0.0, 1.0, 2.0, -1.0]
    N = [0.0, 0.0, 3.0, 4.0]
    pending = [0.0, 2.0, 1.0, 0.0]
    nvl = 1.0
    result = alphazero_virtual_loss(W, N, pending, nvl=nvl)
    assert isinstance(result, dict)
    assert result["k"] == 4
    assert result["nvl"] == nvl
    # N' = N + nvl*pending
    assert result["N"] == [0.0, 2.0, 4.0, 4.0]
    # W' = W - nvl*pending
    assert result["W"] == [0.0, -1.0, 1.0, -1.0]
    # Qclean = 0 where N==0, else W/N
    assert result["Qclean"] == [0.0, 0.0, 2.0 / 3.0, -1.0 / 4.0]
    # Q = 0 where N'==0, else W'/N'
    assert result["Q"] == [0.0, -1.0 / 2.0, 1.0 / 4.0, -1.0 / 4.0]
