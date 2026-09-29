"""Tests for morie.fn.gelrb -- Gelman-Rubin R-hat."""

from morie.fn import _array_core as np
from morie.fn.gelrb import gelman_rubin_rhat, gelrb


def test_alias():
    assert gelrb is gelman_rubin_rhat


def test_converged():
    rng = np.random.default_rng(42)
    chains = [rng.standard_normal(200) for _ in range(4)]
    r = gelman_rubin_rhat(chains)
    assert r.name == "gelman_rubin_rhat"
    assert r.value < 1.2


def test_diverged():
    chains = [np.full(100, 0.0), np.full(100, 10.0)]
    r = gelman_rubin_rhat(chains)
    assert r.value > 1.5


def test_rhat_recomputed():
    import math

    import pytest

    chains = [[1.0, 1.2, 0.8, 1.1, 0.9], [1.5, 1.7, 1.3, 1.6, 1.4], [0.9, 1.0, 1.1, 0.7, 1.2]]
    m, n = 3, 5
    cm = [sum(c) / n for c in chains]
    g = sum(cm) / m
    B = n * sum((v - g) ** 2 for v in cm) / (m - 1)
    W = sum(sum((x - cm[j]) ** 2 for x in c) / (n - 1) for j, c in enumerate(chains)) / m
    vh = (1 - 1 / n) * W + B / n
    r = gelman_rubin_rhat(chains)
    assert r.value == pytest.approx(math.sqrt(vh / W), rel=1e-12)
    assert (r.extra["B"], r.extra["W"]) == pytest.approx((B, W), rel=1e-12)
