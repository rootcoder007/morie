"""Tests for gb_jcd.gibbons_jt_cd_form."""

from morie.fn import _array_core as np

from morie.fn.gb_jcd import gibbons_jt_cd_form


def test_gb_jcd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_per_group = 20
    k = 3
    groups = [rng.normal(0.0, 1.0, n_per_group) for _ in range(k)]
    result = gibbons_jt_cd_form(groups)

    assert isinstance(result, dict)
    # The documented return keys per the function source.
    assert "statistic" in result
    assert "u" in result
    assert "k" in result
    assert "n" in result
    assert "npairs" in result
    assert "method" in result

    # Scalar metadata.
    assert int(result["k"]) == k
    assert int(result["n"]) == n_per_group * k
    assert int(result["npairs"]) == k * (k - 1) // 2

    # U matrix shape and upper-triangle structure.
    u = result["u"]
    assert len(u) == k
    for row in u:
        assert len(row) == k
    for i in range(k):
        for j in range(k):
            assert u[i][j] == 0.0 or j > i

    # Independent recomputation of U_ij and the total statistic
    # directly from the documented Mann-Whitney pairwise count rule.
    def _u_ij(a, b):
        c = 0.0
        for x in a:
            for y in b:
                if x < y:
                    c += 1.0
                elif x == y:
                    c += 0.5
        return c

    expected_u = [[0.0] * k for _ in range(k)]
    expected_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            v = _u_ij(groups[i], groups[j])
            expected_u[i][j] = v
            expected_stat += v

    assert u == expected_u
    assert float(result["statistic"]) == expected_stat


def test_gb_jcd_edge():
    """Test edge cases: k=2 reduces to a single Mann-Whitney count."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 15)
    b = rng.normal(0.0, 1.0, 12)
    result = gibbons_jt_cd_form([a, b])

    assert isinstance(result, dict)
    assert int(result["k"]) == 2
    assert int(result["n"]) == 15 + 12
    assert int(result["npairs"]) == 1

    # With k=2 there is exactly one pairwise count U_01.
    expected = 0.0
    for x in a:
        for y in b:
            if x < y:
                expected += 1.0
            elif x == y:
                expected += 0.5

    assert float(result["statistic"]) == expected
    assert result["u"][0][1] == expected
    assert result["u"][0][0] == 0.0
    assert result["u"][1][0] == 0.0
    assert result["u"][1][1] == 0.0
