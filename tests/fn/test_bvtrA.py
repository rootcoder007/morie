"""Tests for bvtrA.bias_variance_tradeoff."""

from morie.fn import _array_core as np

from morie.fn.bvtrA import bias_variance_tradeoff


def test_bvtrA_basic():
    """Test basic functionality against the documented decomposition.

    Formula: E(y - fhat)^2 = Var(e) + Bias[fhat]^2 + Var(fhat)
    """
    rng_f = np.random.default_rng(101)
    R = 5
    n = 7
    # Predictions: R rows of n predictions each (one row per replicate).
    F = rng_f.normal(0.0, 1.0, (R, n))
    # True function values at the same n evaluation points.
    f = rng_f.normal(0.0, 1.0, n)
    sigma2 = 0.3  # scalar irreducible error

    result = bias_variance_tradeoff(F, f, sigma2)

    # Per the docstring the function returns these keys.
    assert "bias2" in result
    assert "variance" in result
    assert "irreducible" in result
    assert "total" in result
    assert "bias2_point" in result
    assert "variance_point" in result
    assert "R" in result
    assert "n" in result

    # R and n reported by the function must match the inputs.
    assert result["R"] == R
    assert result["n"] == n

    # Pointwise decomposition matches an independent computation on the same
    # inputs (plain arithmetic, no call to the function under test).
    row_means = [sum(F[r][j] for r in range(R)) / R for j in range(n)]
    exp_bias2_pt = [(row_means[j] - f[j]) ** 2 for j in range(n)]
    exp_var_pt = [
        sum((F[r][j] - row_means[j]) ** 2 for r in range(R)) / R
        for j in range(n)
    ]

    # Element-wise equality using zip on the two pointwise lists.
    for got, want in zip(result["bias2_point"], exp_bias2_pt):
        assert abs(got - want) < 1e-12
    for got, want in zip(result["variance_point"], exp_var_pt):
        assert abs(got - want) < 1e-12

    # Scalar averages must equal mean of the pointwise arrays.
    exp_bias2 = sum(exp_bias2_pt) / n
    exp_var = sum(exp_var_pt) / n
    assert abs(result["bias2"] - exp_bias2) < 1e-12
    assert abs(result["variance"] - exp_var) < 1e-12
    assert result["irreducible"] == sigma2
    # Total equals sigma2 + bias2 + variance, computed independently.
    assert abs(result["total"] - (sigma2 + exp_bias2 + exp_var)) < 1e-12


def test_bvtrA_edge():
    """Test a deterministic edge case: zero bias and zero variance with nonzero noise."""
    R = 4
    n = 3
    # Identical predictions across replicates => variance is 0.
    # Predictions equal the truth => bias is 0.
    base = [1.0, 2.0, 3.0]
    F = [list(base) for _ in range(R)]
    f = list(base)
    sigma2 = 0.5

    result = bias_variance_tradeoff(F, f, sigma2)

    assert result["R"] == R
    assert result["n"] == n
    assert abs(result["bias2"]) < 1e-12
    assert abs(result["variance"]) < 1e-12
    assert result["irreducible"] == sigma2
    assert abs(result["total"] - sigma2) < 1e-12
    assert all(abs(v) < 1e-12 for v in result["bias2_point"])
    assert all(abs(v) < 1e-12 for v in result["variance_point"])
