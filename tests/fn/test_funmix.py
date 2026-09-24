"""Tests for funmix.functional_mixture."""

import math

from morie.fn import _array_core as np

from morie.fn.funmix import functional_mixture


def test_funmix_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n, m = 40, 20
    Y = rng.normal(0, 1, (n, m))
    K = 3
    result = functional_mixture(Y, K)

    labels = result["labels"]
    posterior = result["posterior"]
    proportions = result["proportions"]
    mean_curves = result["mean_curves"]
    loglik = result["loglik"]
    bic = result["bic"]

    assert len(labels) == n
    assert len(posterior) == n
    assert len(proportions) == K
    assert len(mean_curves) == K
    assert all(len(mc) == m for mc in mean_curves)
    assert math.isfinite(loglik)
    assert math.isfinite(bic)
    # Labels must be valid component indices
    for lab in labels:
        assert 0 <= lab < K


def test_funmix_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n, m = 10, 5
    Y = rng.normal(0, 1, (n, m))
    K = 1  # minimum valid K
    result = functional_mixture(Y, K)

    labels = result["labels"]
    proportions = result["proportions"]

    assert len(labels) == n
    assert len(proportions) == K
    # With K=1, every curve must be assigned to the single component
    assert all(lab == 0 for lab in labels)
