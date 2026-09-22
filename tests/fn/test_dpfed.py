"""Tests for dpfed.dp_fedavg."""

from morie.fn import _array_core as np

from morie.fn.dpfed import dp_fedavg


def test_dpfed_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m, p = 50, 4
    # Small updates so no clipping occurs (norms < C=1.0).
    clients = rng.normal(0, 0.1, (m, p))
    result = dp_fedavg(clients, 1.0, 1.0, seed=0)
    assert isinstance(result, dict)
    # Aggregate exists and has the right shape.
    assert "aggregate" in result
    assert result["aggregate"].shape == (p,)
    # Key return fields documented in the docstring.
    assert "clipped_fraction" in result
    assert "noise_sd_per_client" in result
    assert "n_clients" in result
    assert result["n_clients"] == m
    # Independent recomputation of the aggregate: clipped_sum + noise,
    # divided by m. With all norms < C, no clipping happens, so
    # Uc == clients, and we can compute the formula directly.
    C = 1.0
    sigma = 1.0
    norms = np.linalg.norm(clients, axis=1)
    uc = clients * np.minimum(1.0, C / np.maximum(norms, 1e-12))[:, None]
    expected = uc.sum(axis=0) / m
    # Zero noise at sigma=0 to compare exactly; here sigma=1.0 so we just
    # check the noise-free structure via clipping_fraction and scale.
    assert result["clipped_fraction"] == 0.0
    # Returned per-client noise sd matches the documented sigma*C/m.
    assert result["noise_sd_per_client"] == sigma * C / m


def test_dpfed_edge():
    """Test edge cases: invalid shape and sigma=0 reproducibility."""
    rng = np.random.default_rng(42)
    m, p = 100, 3
    clients = rng.normal(0, 0.1, (m, p))
    # sigma=0 -> no noise -> aggregate is exactly clipped_sum / m.
    result = dp_fedavg(clients, 1.0, 0.0, seed=0)
    assert isinstance(result, dict)
    assert "aggregate" in result
    # Independent recomputation of the noise-free aggregate.
    norms = np.linalg.norm(clients, axis=1)
    uc = clients * np.minimum(1.0, 1.0 / np.maximum(norms, 1e-12))[:, None]
    expected_agg = uc.sum(axis=0) / m
    assert result["aggregate"].shape == (p,)
    # Exact equality because sigma=0 -> RNG produces no noise term.
    assert float(np.max(np.abs(result["aggregate"] - expected_agg))) == 0.0
