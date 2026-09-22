"""Tests for btbias.boot_bias_estimator."""

from morie.fn import _array_core as np

from morie.fn.btbias import boot_bias_estimator


def test_btbias_basic():
    """Test basic functionality.

    The function expects theta_hat to be a scalar statistic on the
    original data and theta_b to be an array-like of bootstrap
    replicates. The MLE variance with known sigma^2 = 1 has a known
    bias of -sigma^2 / n on samples of size n. We construct replicates
    so the bias is predictable and check the documented direction
    (corrected = 2*theta_hat - mean(replicates), NOT mean(replicates)).
    """
    rng = np.random.default_rng(42)
    n = 100
    sigma2 = 1.0

    # Original sample variance (MLE, biased by -sigma^2/n).
    x = rng.normal(0, 1, n)
    theta_hat = float(np.sum((x - x.mean()) ** 2) / n)

    # Bootstrap replicates of the same statistic on resamples of x.
    B = 500
    idx = rng.integers(0, n, size=(B, n))
    samples = x[idx]
    means = samples.mean(axis=1)
    devs = samples - means[:, None]
    theta_b = np.sum(devs ** 2, axis=1) / n

    result = boot_bias_estimator(theta_hat, theta_b)

    # Result should be a mapping with the documented keys.
    assert isinstance(result, dict)
    assert "bias" in result
    assert "corrected" in result
    assert "estimate" in result
    assert "mean_replicate" in result
    assert "relative_bias" in result
    assert "B" in result

    # Documented formula: bias = mean(replicates) - theta_hat.
    expected_bias = float(theta_b.mean()) - theta_hat
    assert abs(result["bias"] - expected_bias) < 1e-12

    # Documented correction direction: corrected = 2*theta_hat - mean(reps).
    expected_corrected = 2.0 * theta_hat - float(theta_b.mean())
    assert abs(result["corrected"] - expected_corrected) < 1e-12

    # The correction must lie on the OPPOSITE side of theta_hat from the
    # replicate mean (this is what trips people up per the docstring).
    mean_rep = float(theta_b.mean())
    if mean_rep != theta_hat:
        assert (result["corrected"] - theta_hat) * (mean_rep - theta_hat) < 0

    # Reported estimate equals the input statistic.
    assert result["estimate"] == theta_hat
    assert result["mean_replicate"] == mean_rep
    assert result["B"] == int(theta_b.size)
    assert result["relative_bias"] == result["bias"] / theta_hat


def test_btbias_edge():
    """Test edge cases."""
    theta_hat = 3.0
    rng = np.random.default_rng(42)
    # A small but valid set of replicates (need at least 2 per the source).
    theta_b = rng.normal(3.0, 0.1, size=10)
    result = boot_bias_estimator(theta_hat, theta_b)
    assert isinstance(result, dict)
    assert "bias" in result
    assert result["B"] == 10
