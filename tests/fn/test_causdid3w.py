"""Tests for causdid3w.causal_did_three_way."""

from morie.fn import _array_core as np

from morie.fn.causdid3w import causal_did_three_way


def test_causdid3w_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    y = rng.normal(0, 1, n)
    tr = rng.integers(0, 2, n).astype(float)
    po = rng.integers(0, 2, n).astype(float)
    gr = rng.integers(0, 2, n).astype(float)
    result = causal_did_three_way(y, tr, po, gr)

    # Cell means computed independently from the formula.
    cells = {}
    for g in (0, 1):
        for t in (0, 1):
            for p in (0, 1):
                mask = (gr == g) & (tr == t) & (po == p)
                cells[(g, t, p)] = float(y[mask].mean())

    def did(g):
        return ((cells[(g, 1, 1)] - cells[(g, 1, 0)])
                - (cells[(g, 0, 1)] - cells[(g, 0, 0)]))

    expected_did_eligible = did(1)
    expected_did_placebo = did(0)
    expected_ddd = expected_did_eligible - expected_did_placebo

    assert "ddd" in result
    assert "did_eligible" in result
    assert "did_placebo" in result
    assert "cell_means" in result
    assert "se" in result
    assert "p_value" in result
    assert abs(result["ddd"] - expected_ddd) < 1e-12
    assert abs(result["did_eligible"] - expected_did_eligible) < 1e-12
    assert abs(result["did_placebo"] - expected_did_placebo) < 1e-12
    for key in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        assert abs(result["cell_means"][key] - cells[key]) < 1e-12
    assert result["se"] >= 0.0


def test_causdid3w_edge():
    """Test edge cases."""
    # Use a deterministic, fully populated design so every (g, t, post) cell
    # is non-empty, which is required by the function.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    treated = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    group = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

    result = causal_did_three_way(y, treated, post, group)

    cells = {}
    for g in (0, 1):
        for t in (0, 1):
            for p in (0, 1):
                mask = (group == g) & (treated == t) & (post == p)
                cells[(g, t, p)] = float(y[mask].mean())

    def did(g):
        return ((cells[(g, 1, 1)] - cells[(g, 1, 0)])
                - (cells[(g, 0, 1)] - cells[(g, 0, 0)]))

    expected_did_eligible = did(1)
    expected_did_placebo = did(0)
    expected_ddd = expected_did_eligible - expected_did_placebo

    assert "ddd" in result
    assert "did_eligible" in result
    assert "did_placebo" in result
    assert "cell_means" in result
    assert "se" in result
    assert "p_value" in result
    assert abs(result["ddd"] - expected_ddd) < 1e-12
    assert abs(result["did_eligible"] - expected_did_eligible) < 1e-12
    assert abs(result["did_placebo"] - expected_did_placebo) < 1e-12
    for key in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        assert abs(result["cell_means"][key] - cells[key]) < 1e-12
