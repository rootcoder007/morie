"""Tests for aitprm.compositional_permanova."""

from morie.fn import _array_core as np

from morie.fn.aitprm import compositional_permanova


def test_aitprm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Compositions must be strictly positive when aitchison=True (the default).
    X = rng.uniform(0.1, 5.0, (30, 4))
    groups = ["g0", "g1", "g2"] * 10
    result = compositional_permanova(X, groups, True)
    assert isinstance(result, dict)
    assert "F" in result
    assert "SSA" in result
    assert "SSW" in result
    assert "SST" in result
    assert "df1" in result
    assert "df2" in result

    # Basic shape / type checks of the documented outputs.
    assert result["a"] == 3
    assert result["N"] == 30
    assert result["df1"] == 2
    assert result["df2"] == 27
    assert result["sizes"] == [10, 10, 10]
    assert isinstance(result["F"], float)

    # Independent recomputation of SST, SSW and the pseudo-F from the
    # Aitchison (clr -> Euclidean) distances, to verify the closed-form
    # statistic in the docstring rather than trust the function's number.
    import math
    N, D = 30, 4
    clr = []
    for r in X:
        lg = [math.log(v) for v in r]
        gm = sum(lg) / D
        clr.append([v - gm for v in lg])
    size = {"g0": 10, "g1": 10, "g2": 10}
    sst = 0.0
    ssw = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d2 = sum((clr[i][k] - clr[j][k]) ** 2 for k in range(D))
            sst += d2
            if groups[i] == groups[j]:
                ssw += d2 / size[groups[i]]
    sst /= N
    ssa = sst - ssw
    df1, df2 = 2, 27
    expected_F = (ssa / df1) / (ssw / df2)
    assert result["SSA"] == ssa
    assert result["SSW"] == ssw
    assert result["SST"] == sst
    assert result["F"] == expected_F


def test_aitprm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 5.0, (30, 4))
    groups = ["g0", "g1", "g2"] * 10
    # Default aitchison=True, so pass only the two required positional args.
    result = compositional_permanova(X, groups)
    assert isinstance(result, dict)
    assert "F" in result
    assert "SSA" in result
    assert "SSW" in result
    assert "SST" in result
    assert "df1" in result
    assert "df2" in result
