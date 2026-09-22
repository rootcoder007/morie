"""Tests for bayhier.hierarchical_pooling."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.bayhier import hierarchical_pooling


def test_bayhier_basic():
    """Test basic functionality."""
    # Build well-formed inputs: y is numeric observations; group is a
    # label per observation. The function compares group values for
    # equality, so use integer group labels (not continuous random
    # noise) and choose enough groups and observations to leave
    # positive residual degrees of freedom for sigma2.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    result = hierarchical_pooling(y, group)

    # The function returns a RichResult; its payload is what we inspect.
    assert isinstance(result, dict)

    # Required documented outputs.
    assert "theta" in result
    assert "lambda_g" in result
    assert "theta_nopool" in result
    assert "theta_pool" in result
    assert "mu" in result
    assert "sigma2" in result
    assert "tau2" in result

    # Three groups, four observations each.
    assert len(result["theta"]) == 3
    assert len(result["lambda_g"]) == 3
    assert len(result["theta_nopool"]) == 3

    # lambda_g is a shrinkage factor in [0, 1].
    for lg in result["lambda_g"]:
        assert 0.0 <= lg <= 1.0

    # Independent recomputation: within-group sum of squares,
    # pooled within-group MS (with df = n - G), between-group MS,
    # n_tilde, tau2, lambda_g, theta, and theta_pool.
    n = len(y)
    G = 3
    y0 = y[group == 0]
    y1 = y[group == 1]
    y2 = y[group == 2]
    n_g = [len(y0), len(y1), len(y2)]
    ybar = [y0.mean(), y1.mean(), y2.mean()]
    grand = y.mean()

    ssw = 0.0
    for yg, ybg in zip([y0, y1, y2], ybar):
        for v in yg:
            d = v - ybg
            ssw += d * d
    sigma2_expected = ssw / (n - G)

    ssb = 0.0
    for ng, ybg in zip(n_g, ybar):
        d = ybg - grand
        ssb += ng * d * d
    msb = ssb / (G - 1)

    sq = 0.0
    for ng in n_g:
        sq += ng * ng
    ntil = (n - sq / n) / (G - 1)
    tau2_expected = (msb - sigma2_expected) / ntil
    if tau2_expected < 0.0:
        tau2_expected = 0.0

    lam_expected = []
    for ng, ybg in zip(n_g, ybar):
        sgj = sigma2_expected / ng
        if (tau2_expected + sgj) > 0.0:
            lam_expected.append(tau2_expected / (tau2_expected + sgj))
        else:
            lam_expected.append(0.0)

    num = 0.0
    den = 0.0
    for ng, ybg in zip(n_g, ybar):
        denom = tau2_expected + sigma2_expected / ng
        if denom > 0.0:
            w = 1.0 / denom
        else:
            w = 0.0
        num += w * ybg
        den += w
    mu_expected = num / den if den > 0.0 else grand

    theta_expected = [
        lam_expected[j] * ybar[j] + (1.0 - lam_expected[j]) * mu_expected
        for j in range(G)
    ]

    # Variance hyperparameters agree with the independent formula.
    assert abs(result["sigma2"] - sigma2_expected) < 1e-10
    assert abs(result["tau2"] - tau2_expected) < 1e-10
    assert abs(result["mu"] - mu_expected) < 1e-10
    assert abs(result["theta_pool"] - mu_expected) < 1e-10
    assert abs(result["grand_mean"] - grand) < 1e-10

    # Per-group outputs agree with the independent formula.
    for j in range(G):
        assert abs(result["theta_nopool"][j] - ybar[j]) < 1e-10
        assert abs(result["lambda_g"][j] - lam_expected[j]) < 1e-10
        assert abs(result["theta"][j] - theta_expected[j]) < 1e-10

    # The provided sigma2 and tau2 must override the estimated ones.
    result2 = hierarchical_pooling(y, group, sigma2=0.5, tau2=0.25)
    assert result2["sigma2"] == 0.5
    assert result2["tau2"] == 0.25


def test_bayhier_edge():
    """Test edge cases."""
    # Use a valid configuration with positive residual df.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    result = hierarchical_pooling(y, group)
    assert isinstance(result, dict)
    assert "theta" in result
    assert "lambda_g" in result
    assert "theta_nopool" in result
    assert "theta_pool" in result
