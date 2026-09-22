"""Tests for alfsts.alphafold_structure_transition."""

from morie.fn import _array_core as np

from morie.fn.alfsts import alphafold_structure_transition


def _relu(x):
    return x if x > 0 else 0


def _lin(x_row, w):
    cs_in = len(x_row)
    cs_out = len(w)
    return [sum(w[o][i] * x_row[i] for i in range(cs_in)) for o in range(cs_out)]


def _vadd(a, b):
    return [a[t] + b[t] for t in range(len(a))]


def _lnorm(row):
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    return [(x - mean) / (var + 1e-5) ** 0.5 for x in row]


def _to_list(a):
    if hasattr(a, "tolist"):
        return a.tolist()
    return a


def test_alfsts_basic():
    """Test basic functionality with documented argument shapes."""
    rng_s = np.random.default_rng(0)
    rng_w1 = np.random.default_rng(1)
    rng_w2 = np.random.default_rng(2)
    rng_w3 = np.random.default_rng(3)

    n, cs = 3, 4
    s = _to_list(rng_s.normal(0, 1, n * cs).reshape(n, cs))
    w1 = _to_list(rng_w1.normal(0, 1, cs * cs).reshape(cs, cs))
    w2 = _to_list(rng_w2.normal(0, 1, cs * cs).reshape(cs, cs))
    w3 = _to_list(rng_w3.normal(0, 1, cs * cs).reshape(cs, cs))

    result = alphafold_structure_transition(s, w1, w2, w3)

    # Documented return keys
    assert "s" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # n must equal number of input rows
    assert result["n"] == n
    assert len(result["s"]) == n

    # Compute the expected output independently from the documented formula.
    expected_s = []
    for i in range(n):
        h1 = _lin(s[i], w1)
        h1a = [_relu(t) for t in h1]
        h2 = _lin(h1a, w2)
        h2a = [_relu(t) for t in h2]
        h3 = _lin(h2a, w3)
        u = _vadd(s[i], h3)
        u = _lnorm(u)
        expected_s.append(u)

    expected_flat = [expected_s[i][t] for i in range(n) for t in range(cs)]
    expected_estimate = sum(expected_flat) / len(expected_flat)

    assert result["estimate"] == expected_estimate
    for i in range(n):
        for t in range(cs):
            assert result["s"][i][t] == expected_s[i][t]


def test_alfsts_edge():
    """Test edge case: with w3 zero and no layernorm the result equals the input."""
    rng_s = np.random.default_rng(0)
    rng_drop = np.random.default_rng(7)

    n, cs = 2, 3
    s = _to_list(rng_s.normal(0, 1, n * cs).reshape(n, cs))
    w1 = _to_list(np.random.default_rng(1).normal(0, 1, cs * cs).reshape(cs, cs))
    w2 = _to_list(np.random.default_rng(2).normal(0, 1, cs * cs).reshape(cs, cs))
    w3 = [[0.0] * cs for _ in range(cs)]

    # w3 zero, no layernorm: output equals input row-for-row
    result = alphafold_structure_transition(s, w1, w2, w3, layernorm=False)

    assert "s" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n

    for i in range(n):
        for t in range(cs):
            assert result["s"][i][t] == s[i][t]

    expected_estimate = sum(s[i][t] for i in range(n) for t in range(cs)) / (n * cs)
    assert result["estimate"] == expected_estimate

    # With dropout mask applied, verify it multiplies through element-wise
    drop = _to_list(rng_drop.uniform(0.5, 1.0, n * cs).reshape(n, cs))
    result2 = alphafold_structure_transition(s, w1, w2, w3, layernorm=False, drop=drop)

    for i in range(n):
        for t in range(cs):
            assert result2["s"][i][t] == s[i][t] * drop[i][t]
