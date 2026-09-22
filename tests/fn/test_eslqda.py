"""Tests for eslqda.esl_qda."""

from morie.fn import _array_core as np

from morie.fn.eslqda import esl_qda


def test_eslqda_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n0, n1 = 50, 50
    p = 5
    X0 = rng_x.normal(loc=-1.0, scale=1.0, size=(n0, p))
    X1 = rng_x.normal(loc=2.0, scale=1.0, size=(n1, p))
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    result = esl_qda(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "prediction" in result
    assert "discriminants" in result
    assert "classes" in result
    assert "priors" in result
    assert "log_dets" in result
    assert "n" in result
    assert "p" in result
    assert "K" in result
    assert "method" in result
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["K"] == 2
    assert result["priors"] == [0.5, 0.5]
    expected_est = result["prediction"][0]
    assert result["estimate"] == expected_est
    assert len(result["prediction"]) == 100
    assert len(result["discriminants"]) == 100 * 2
    assert all(isinstance(v, float) for v in result["discriminants"])
    for k in range(2):
        col = result["discriminants"][k::2]
        for i, val in enumerate(col):
            assert val == col[i]
    cls_arr = list(result["classes"])
    assert sorted(cls_arr) == sorted([0, 1]) or sorted(cls_arr) == sorted(["0", "1"])


def test_eslqda_edge():
    """Test edge cases: one class with too few observations."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n0, n1 = 50, 50
    p = 5
    X0 = rng_x.normal(loc=-1.0, scale=1.0, size=(n0, p))
    X1 = rng_x.normal(loc=2.0, scale=1.0, size=(n1, p))
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    result = esl_qda(X, y)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["K"] == 2
