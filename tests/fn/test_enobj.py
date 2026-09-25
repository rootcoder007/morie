"""Tests for enobj.elastic_net_objective."""

from morie.fn.enobj import elastic_net_objective


def test_enobj_basic():
    """Penalized RSS on a design whose fit is exact, so PRSS is the penalty."""
    X = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]]
    y = [1.0, 2.0, 3.0, 6.0]
    beta = [0.0, 1.0, 2.0, 3.0]  # intercept 0, then slopes 1, 2, 3
    lam = 0.1
    alpha = 0.5
    result = elastic_net_objective(X, y, beta, lam, alpha)
    # Fit is exact: every fitted value equals its observation.
    assert result["rss"] == 0.0
    # Intercept is unpenalized, so only the three slopes enter the penalties.
    assert result["l2"] == 1.0 ** 2 + 2.0 ** 2 + 3.0 ** 2
    assert result["l1"] == 1.0 + 2.0 + 3.0
    expected_pen = lam * (0.5 * (1.0 - alpha) * 14.0 + alpha * 6.0)
    assert abs(result["penalty"] - expected_pen) < 1e-12
    assert abs(result["prss"] - expected_pen) < 1e-12
    assert result["lambda"] == lam
    assert result["alpha"] == alpha
    assert result["n"] == 4
    assert result["p"] == 4  # three columns plus the prepended intercept


def test_enobj_edge():
    """alpha endpoints, no intercept, and the documented input checks."""
    X = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 1.0]]
    y = [1.0, 1.0, 1.0]
    beta = [2.0, -3.0]

    # add_intercept=False: design is X itself, both coefficients penalized.
    ridge = elastic_net_objective(X, y, beta, 1.0, 0.0, add_intercept=False)
    rss = (1.0 - 2.0) ** 2 + (1.0 - (-3.0)) ** 2 + (1.0 - (2.0 - 3.0)) ** 2
    assert abs(ridge["rss"] - rss) < 1e-12
    assert ridge["l2"] == 4.0 + 9.0
    assert ridge["l1"] == 5.0
    # alpha = 0 is the ridge penalty: lambda/2 * sum b_j^2.
    assert abs(ridge["penalty"] - 0.5 * 13.0) < 1e-12
    assert abs(ridge["prss"] - (rss + 0.5 * 13.0)) < 1e-12

    # alpha = 1 is the lasso penalty: lambda * sum |b_j|.
    lasso = elastic_net_objective(X, y, beta, 2.0, 1.0, add_intercept=False)
    assert abs(lasso["penalty"] - 2.0 * 5.0) < 1e-12

    # lambda = 0 leaves the unpenalized RSS.
    plain = elastic_net_objective(X, y, beta, 0.0, 0.5, add_intercept=False)
    assert plain["penalty"] == 0.0
    assert abs(plain["prss"] - rss) < 1e-12

    for bad in (
        lambda: elastic_net_objective(X, [1.0, 1.0], beta, 0.1, 0.5, add_intercept=False),
        lambda: elastic_net_objective(X, y, [1.0], 0.1, 0.5, add_intercept=False),
        lambda: elastic_net_objective(X, y, beta, -1.0, 0.5, add_intercept=False),
        lambda: elastic_net_objective(X, y, beta, 0.1, 1.5, add_intercept=False),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
