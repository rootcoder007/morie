"""Tests for aitamg.aitchison_amalgamation."""

from morie.fn import _array_core as np

from morie.fn.aitamg import aitchison_amalgamation


def test_aitamg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10.0, 100)
    idx = [1, 2, 3]
    result = aitchison_amalgamation(x, idx)
    assert isinstance(result, dict)
    assert "composition" in result
    assert "amalgamated" in result
    assert "parts" in result
    assert "kept" in result
    assert "D" in result

    # Independent computation of the expected composition.
    sel = {1, 2, 3}
    keep = [i for i in range(1, len(x) + 1) if i not in sel]
    amal = sum(x[i - 1] for i in idx)
    raw = [x[i - 1] for i in keep] + [amal]
    s = sum(raw)
    k = 1.0
    expected = [k * v / s for v in raw]

    assert len(result["composition"]) == len(expected)
    for got, exp in zip(result["composition"], expected):
        assert abs(got - exp) < 1e-12

    # The composition must sum to total (1.0 by default).
    assert abs(sum(result["composition"]) - 1.0) < 1e-12

    # All returned composition parts must be strictly positive.
    for v in result["composition"]:
        assert v > 0

    # The amalgamated part equals the sum of the selected input parts.
    assert abs(result["amalgamated"] - amal) < 1e-12

    # 'parts' must contain exactly the one-based indices passed in.
    assert sorted(result["parts"]) == sorted(idx)

    # 'kept' must contain the complementary one-based indices.
    assert sorted(result["kept"]) == sorted(keep)

    # D is the number of parts in the amalgamated composition.
    assert result["D"] == len(keep) + 1


def test_aitamg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10.0, 100)
    # Amalgamate the first two parts of a 4-part composition.
    idx = [1, 2]
    result = aitchison_amalgamation(x, idx)
    assert isinstance(result, dict)
    assert "composition" in result

    # Independent computation.
    sel = {1, 2}
    keep = [i for i in range(1, len(x) + 1) if i not in sel]
    amal = sum(x[i - 1] for i in idx)
    raw = [x[i - 1] for i in keep] + [amal]
    s = sum(raw)
    expected = [1.0 * v / s for v in raw]

    assert len(result["composition"]) == len(expected)
    for got, exp in zip(result["composition"], expected):
        assert abs(got - exp) < 1e-12
    assert abs(sum(result["composition"]) - 1.0) < 1e-12
    assert result["D"] == len(expected)
