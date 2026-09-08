"""Tests for km069.kamath_ch5_rlhf_objective."""

from morie.fn import _array_core as np

from morie.fn.km069 import kamath_ch5_rlhf_objective


def test_km069_basic():
    """Test basic functionality."""
    # pi_theta / pi_ref are probability distributions over the same
    # response set; the generator fed raw normal draws, which the
    # module correctly refuses.
    raw = [abs(float(v)) + 1e-6
           for v in np.random.default_rng(42).normal(0, 1, 100)._flat()]
    tot = sum(raw)
    pi_theta = [v / tot for v in raw]
    raw2 = [abs(float(v)) + 1e-6
            for v in np.random.default_rng(7).normal(0, 1, 100)._flat()]
    tot2 = sum(raw2)
    pi_ref = [v / tot2 for v in raw2]
    r_phi = [float(v)
             for v in np.random.default_rng(9).normal(0, 1, 100)._flat()]
    beta = 0.8
    result = kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km069_edge():
    """Test edge cases."""
    # pi_theta / pi_ref are probability distributions over the same
    # response set; the generator fed raw normal draws, which the
    # module correctly refuses.
    raw = [abs(float(v)) + 1e-6
           for v in np.random.default_rng(42).normal(0, 1, 100)._flat()]
    tot = sum(raw)
    pi_theta = [v / tot for v in raw]
    raw2 = [abs(float(v)) + 1e-6
            for v in np.random.default_rng(7).normal(0, 1, 100)._flat()]
    tot2 = sum(raw2)
    pi_ref = [v / tot2 for v in raw2]
    r_phi = [float(v)
             for v in np.random.default_rng(9).normal(0, 1, 100)._flat()]
    beta = 0.8
    result = kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta)
    assert isinstance(result, dict)
