"""Tests for blip2v.blip2_qformer."""

from morie.fn import _array_core as np

from morie.fn.blip2v import blip2_qformer


def _flat_to_2d(flat, rows):
    n = len(flat)
    cols = n // rows
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def test_blip2v_basic():
    """Test basic functionality with documented argument shapes."""
    rng = np.random.default_rng(42)

    q_dim = 6
    f_dim = 4
    dk = 8
    dv = 5
    n_queries = 3
    n_patches = 5

    queries_flat = list(rng.normal(0, 1, n_queries * q_dim))
    queries = _flat_to_2d(queries_flat, n_queries)

    image_features_flat = list(rng.normal(0, 1, n_patches * f_dim))
    image_features = _flat_to_2d(image_features_flat, n_patches)

    WQ = _flat_to_2d(list(rng.normal(0, 1, dk * q_dim)), dk)
    WK = _flat_to_2d(list(rng.normal(0, 1, dk * f_dim)), dk)
    WV = _flat_to_2d(list(rng.normal(0, 1, dv * f_dim)), dv)

    result = blip2_qformer(queries, image_features, WQ, WK, WV)

    assert isinstance(result, dict)
    assert "output" in result
    assert "weights" in result
    assert "n_queries" in result
    assert "n_patches" in result
    assert "compression" in result
    assert "note" in result

    out = result["output"]
    weights = result["weights"]

    assert len(out) == n_queries
    assert len(out[0]) == dv

    assert len(weights) == n_queries
    for w in weights:
        assert len(w) == n_patches
        s = sum(w)
        assert abs(s - 1.0) < 1e-6
        assert min(w) >= 0.0

    assert result["n_queries"] == n_queries
    assert result["n_patches"] == n_patches
    assert abs(result["compression"] - n_patches / n_queries) < 1e-12


def test_blip2v_edge():
    """Test edge cases with documented argument shapes."""
    rng = np.random.default_rng(7)

    q_dim = 4
    f_dim = 3
    dk = 4
    dv = 6
    n_queries = 1
    n_patches = 1

    queries = _flat_to_2d(list(rng.normal(0, 1, n_queries * q_dim)), n_queries)
    image_features = _flat_to_2d(list(rng.normal(0, 1, n_patches * f_dim)), n_patches)
    WQ = _flat_to_2d(list(rng.normal(0, 1, dk * q_dim)), dk)
    WK = _flat_to_2d(list(rng.normal(0, 1, dk * f_dim)), dk)
    WV = _flat_to_2d(list(rng.normal(0, 1, dv * f_dim)), dv)

    result = blip2_qformer(queries, image_features, WQ, WK, WV)

    assert isinstance(result, dict)
    assert len(result["output"]) == 1
    assert len(result["output"][0]) == dv
    assert result["weights"] == [[1.0]]
    assert result["n_queries"] == 1
    assert result["n_patches"] == 1
    assert abs(result["compression"] - 1.0) < 1e-12
