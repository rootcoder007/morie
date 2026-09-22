"""Tests for alfpaf.alphafold_pair_repr."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.alfpaf import alphafold_pair_repr


def test_alfpaf_basic():
    """Test basic functionality."""
    s = 4
    n = 3
    cm = 2
    c = 2
    cz = 3

    rng = np.random.default_rng(44)
    m = [[[float(v) for v in rng.normal(0, 1, cm)] for _ in range(n)] for _ in range(s)]
    wa = [[float(v) for v in rng.normal(0, 1, cm)] for _ in range(c)]
    wb = [[float(v) for v in rng.normal(0, 1, cm)] for _ in range(c)]
    wo = [[float(v) for v in rng.normal(0, 1, c * c)] for _ in range(cz)]

    result = alphafold_pair_repr(m, wa, wb, wo)
    assert hasattr(result, "keys") or isinstance(result, dict)
    assert "z" in result.keys()
    assert "o" in result.keys()
    assert "estimate" in result.keys()
    assert "n" in result.keys()
    assert "method" in result.keys()
    assert result["n"] == n
    assert len(result["z"]) == n
    assert len(result["z"][0]) == n
    assert len(result["z"][0][0]) == cz
    assert len(result["o"]) == n
    assert len(result["o"][0]) == n
    assert len(result["o"][0][0]) == c * c


def test_alfpaf_edge():
    """Test edge cases."""
    s = 3
    n = 2
    cm = 2
    c = 1
    cz = 1

    m = [[[float(v) for v in (1.0, 0.0)] for _ in range(n)] for _ in range(s)]
    wa = [[1.0] * cm]
    wb = [[1.0] * cm]
    wo = [[1.0] * (c * c)]

    result = alphafold_pair_repr(m, wa, wb, wo, layernorm=False)
    assert "z" in result.keys()
    assert "o" in result.keys()
    # o[i][j] should be mean over s of m[si][i][0] * m[si][j][0]
    expected_o = [[
        sum(m[si][i][0] * m[si][j][0] for si in range(s)) / s
        for j in range(n)
    ] for i in range(n)]
    for i in range(n):
        for j in range(n):
            assert abs(result["o"][i][j][0] - expected_o[i][j]) < 1e-9
    assert result["n"] == n
