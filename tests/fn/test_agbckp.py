"""Tests for agbckp.alphazero_backup."""

from morie.fn import _array_core as np

from morie.fn.agbckp import alphazero_backup


def test_agbckp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    leaf = rng.normal(0, 1, 100)
    value = float(rng.normal())
    path = [0, 1, 2]  # 3 edges from root to leaf
    result = alphazero_backup(leaf, value, path)

    # The function returns a RichResult whose payload is the documented dict.
    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert "n" in payload and "w" in payload and "q" in payload and "g" in payload

    # Independent recomputation of the AlphaZero backup along a 3-edge path.
    # Walk leaf-ward first; at each step negate (alternate=True), then
    # acc = r[i] + gamma * acc. With rewards all zero and gamma=1.0,
    # we get acc alternates sign each step starting from `value`.
    L = 3
    gamma = 1.0
    r = [0.0, 0.0, 0.0]
    acc = float(value)
    g_expected = [0.0] * L
    for i in range(L - 1, -1, -1):
        acc = -acc
        acc = r[i] + gamma * acc
        g_expected[i] = acc
    # All N, W default to zero -> q = 0/1 after a single visit on every edge.
    n_expected = [1.0, 1.0, 1.0]
    w_expected = list(g_expected)
    q_expected = [w_expected[i] / n_expected[i] for i in range(L)]
    estimate_expected = q_expected[0]

    assert payload["g"] == g_expected
    assert payload["n"] == n_expected
    assert payload["w"] == w_expected
    assert payload["q"] == q_expected
    assert payload["estimate"] == estimate_expected
    assert payload["leaf"] is leaf


def test_agbckp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    leaf = rng.normal(0, 1, 100)
    value = float(rng.normal())
    path = []  # empty path -> empty outputs, estimate is NaN
    result = alphazero_backup(leaf, value, path)

    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert payload["n"] == []
    assert payload["w"] == []
    assert payload["q"] == []
    assert payload["g"] == []
    # With no edges, the root estimate is undefined.
    assert payload["estimate"] != payload["estimate"]  # NaN check
    assert payload["leaf"] is leaf
