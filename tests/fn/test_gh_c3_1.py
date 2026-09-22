"""Tests for gh_c3_1.ghosal_random_measure_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_1 import ghosal_random_measure_def


def test_gh_c3_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_atoms = 50
    seed = 42
    result = ghosal_random_measure_def(x, n_atoms=n_atoms, seed=seed)

    assert "estimate" in result
    assert "additivity_gap" in result
    assert "total_mass" in result
    assert "method" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(est)
    assert 0.0 <= est <= 1.0

    total_mass = float(np.asarray(result["total_mass"], dtype=float))
    assert np.isclose(total_mass, 1.0)

    additivity_gap = float(np.asarray(result["additivity_gap"], dtype=float))
    assert additivity_gap >= 0.0
    assert np.isclose(additivity_gap, 0.0)

    # Independent recomputation using the documented procedure:
    # atoms ~ Uniform(0,1), weights ~ Gamma(1,1) then normalized.
    rng = np.random.default_rng(seed)
    atoms = [float(v) for v in rng.uniform(0.0, 1.0, n_atoms)]
    w_raw = [float(v) for v in rng.gamma(1.0, 1.0, n_atoms)]
    s = sum(w_raw)
    w = [wi / s for wi in w_raw]

    def P(a, b):
        return sum(wi for wi, t in zip(w, atoms) if a <= t < b)

    expected_est = P(0.0, 0.5)
    expected_gap = abs(P(0.0, 1.0) - (P(0.0, 0.4) + P(0.4, 1.0)))

    assert np.isclose(est, expected_est)
    assert np.isclose(additivity_gap, expected_gap)
    assert np.isclose(total_mass, P(0.0, 1.0))


def test_gh_c3_1_edge():
    """Test edge cases: single input still works and returns valid measure."""
    x = np.array([42.0])
    n_atoms = 50
    seed = 42
    result = ghosal_random_measure_def(x, n_atoms=n_atoms, seed=seed)

    assert "estimate" in result
    assert "additivity_gap" in result
    assert "total_mass" in result

    total_mass = float(np.asarray(result["total_mass"], dtype=float))
    assert np.isclose(total_mass, 1.0)

    est = float(np.asarray(result["estimate"], dtype=float))
    assert 0.0 <= est <= 1.0

    additivity_gap = float(np.asarray(result["additivity_gap"], dtype=float))
    assert np.isclose(additivity_gap, 0.0)

    # Verify total_mass equals sum of weights (which is 1 by normalization).
    rng = np.random.default_rng(seed)
    atoms = [float(v) for v in rng.uniform(0.0, 1.0, n_atoms)]
    w_raw = [float(v) for v in rng.gamma(1.0, 1.0, n_atoms)]
    s = sum(w_raw)
    w = [wi / s for wi in w_raw]

    def P(a, b):
        return sum(wi for wi, t in zip(w, atoms) if a <= t < b)

    assert np.isclose(total_mass, P(0.0, 1.0))
