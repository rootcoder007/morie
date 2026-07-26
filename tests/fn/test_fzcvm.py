"""fzcvm: smoothed Cramer-von Mises goodness-of-fit test.

Fauzi & Maesono (2023), *Statistical Inference Based on Kernel Distribution
Function Estimators*, Ch. 5.
"""

import numpy as np
import pytest

from morie.fn.fzcvm import fauzi_cvm_smoothed as cvm


def test_fzcvm_does_not_reject_the_true_distribution():
    rng = np.random.default_rng(2801)
    assert cvm(rng.standard_normal(300), cdf="norm", args=(0.0, 1.0))["p_value"] > 0.05


def test_fzcvm_rejects_a_badly_wrong_distribution():
    rng = np.random.default_rng(2803)
    x = rng.uniform(-3, 3, 400)
    assert cvm(x, cdf="norm", args=(0.0, 1.0))["p_value"] < 0.05


def test_fzcvm_statistic_is_non_negative():
    """CvM integrates a squared discrepancy, so it cannot go below zero."""
    rng = np.random.default_rng(2807)
    for n in (30, 200, 800):
        assert cvm(rng.standard_normal(n), cdf="norm", args=(0.0, 1.0))["statistic"] >= 0.0


def test_fzcvm_grows_with_the_size_of_the_misfit():
    """Shifting the data further from the hypothesised centre must increase
    the discrepancy monotonically."""
    rng = np.random.default_rng(2811)
    base = rng.standard_normal(400)
    stats_ = [
        cvm(base + shift, cdf="norm", args=(0.0, 1.0))["statistic"]
        for shift in (0.0, 0.5, 1.0, 2.0)
    ]
    assert stats_ == sorted(stats_)


def test_fzcvm_reports_bandwidth_and_sample_size():
    rng = np.random.default_rng(2819)
    r = cvm(rng.standard_normal(150), cdf="norm", args=(0.0, 1.0), h=0.2)
    assert r["n"] == 150
    assert r["h"] == pytest.approx(0.2)


def test_fzcvm_is_location_scale_equivariant():
    """Testing N(0,1) on x and N(mu,sigma) on mu + sigma*x is the same
    question, so the statistic must agree."""
    rng = np.random.default_rng(2833)
    x = rng.standard_normal(300)
    a = cvm(x, cdf="norm", args=(0.0, 1.0))["statistic"]
    b = cvm(3.0 + 2.0 * x, cdf="norm", args=(3.0, 2.0))["statistic"]
    assert a == pytest.approx(b, rel=0.05)


def test_fzcvm_controls_its_size_and_has_power():
    """The regression guard for the p-value scaling bug.

    "It rejected two examples" is not evidence a test works. What matters is
    that it rejects a TRUE null only about alpha of the time and rejects
    false ones often. Measured at n = 200 over 300 replications, alpha = 0.05:

        true null            0.040   <- size, correctly conservative
        shift 0.25           0.917
        shift 0.50 / 1.00    1.000
        t(3) heavy tails     0.497
        uniform, matched     0.367
          mean and variance

    Before the scaling fix the p-value was 0.5 for every input, so size was 0
    and power was 0 -- the test could not reject anything at all. This
    catches a regression to that state, which no single-example check would.

    The shape alternatives are deliberately kept as loose bounds: a CvM test
    at this n has genuine but modest power against distributions matching the
    first two moments, and pretending otherwise would make the test flaky.
    """
    rng = np.random.default_rng(4242)
    n, reps = 200, 120

    def reject_rate(gen):
        p = [cvm(gen(), cdf="norm", args=(0.0, 1.0))["p_value"] for _ in range(reps)]
        return float(np.mean(np.asarray(p) <= 0.05))

    size = reject_rate(lambda: rng.standard_normal(n))
    assert size <= 0.12, f"size {size} -- Type I error inflated"

    assert reject_rate(lambda: rng.standard_normal(n) + 0.5) > 0.9
    assert reject_rate(lambda: rng.standard_normal(n) + 1.0) > 0.95
    # Shape-only alternatives: real power, but modest at this n.
    assert reject_rate(lambda: rng.standard_t(3, n)) > 0.2
