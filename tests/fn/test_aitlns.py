"""Tests for aitlns.logistic_normal_sample."""

from morie.fn import _array_core as np

from morie.fn.aitlns import logistic_normal_sample


def test_aitlns_basic():
    """Test basic functionality."""
    mu = np.array([0.0, 0.0, 0.0])
    Sigma = np.array([[1.0, 0.1, 0.0],
                      [0.1, 1.0, 0.1],
                      [0.0, 0.1, 1.0]])
    n = 100
    result = logistic_normal_sample(mu, Sigma, n)
    assert isinstance(result, dict)

    # Documented returned keys
    assert "sample" in result
    assert "alr" in result
    assert "center" in result
    assert "mean_alr" in result
    assert "n" in result
    assert "D" in result

    # Shapes derived from documented behaviour (D-1 alr coords -> D parts)
    D = int(result["D"])
    assert D == 4
    assert len(result["sample"]) == n
    assert len(result["sample"][0]) == D
    assert len(result["alr"]) == n
    assert len(result["alr"][0]) == D - 1
    assert len(result["center"]) == D
    assert len(result["mean_alr"]) == D - 1

    # Compositional invariants: each composition sums to total (default 1.0)
    for row in result["sample"]:
        s = sum(row)
        assert abs(s - 1.0) < 1e-9
        for v in row:
            assert v > 0.0

    # 'center' is alr^-1(mu): exponentiate mu and a trailing 1, then close.
    e = [float(np.exp(mu[i])) for i in range(3)] + [1.0]
    s = sum(e)
    expected_center = [v / s for v in e]
    for a, b in zip(result["center"], expected_center):
        assert abs(a - b) < 1e-9

    # mean_alr is the column mean of the alr draws
    p = D - 1
    expected_mean_alr = [sum(result["alr"][t][i] for t in range(n)) / n
                         for i in range(p)]
    for a, b in zip(result["mean_alr"], expected_mean_alr):
        assert abs(a - b) < 1e-9


def test_aitlns_edge():
    """Test edge cases."""
    mu = np.array([0.0])
    Sigma = np.array([[1.0]])
    n = 5
    result = logistic_normal_sample(mu, Sigma, n, seed=7)
    assert isinstance(result, dict)
    assert len(result["sample"]) == n
    assert len(result["sample"][0]) == 2
    for row in result["sample"]:
        assert abs(sum(row) - 1.0) < 1e-9
        for v in row:
            assert v > 0.0
