"""Tests for jntmed.joint_significance_mediation."""

import numpy as np
import pytest

from morie.fn.jntmed import joint_significance_mediation


def test_jntmed_detects_true_mediation():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    m = 0.7 * x + rng.normal(size=n)
    y = 0.6 * m + 0.2 * x + rng.normal(size=n)
    r = joint_significance_mediation(x, m, y)
    assert r["significant"] is True
    assert float(r["p_value"]) == max(float(r["p_a"]), float(r["p_b"]))
    assert float(r["a"]) == pytest.approx(0.7, abs=0.15)
    assert float(r["b"]) == pytest.approx(0.6, abs=0.15)


def test_jntmed_no_mediation_when_either_path_is_dead():
    """Both a = 0 and b = 0 cases must fail the joint test -- the whole
    point of requiring BOTH paths (MacKinnon et al. 2002)."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    # a-path dead: M independent of X, but M drives Y.
    m0 = rng.normal(size=n)
    y0 = 0.8 * m0 + rng.normal(size=n)
    assert joint_significance_mediation(x, m0, y0)["significant"] is False
    # b-path dead: X drives M, M irrelevant to Y.
    m1 = 0.8 * x + rng.normal(size=n)
    y1 = 0.5 * x + rng.normal(size=n)
    assert joint_significance_mediation(x, m1, y1)["significant"] is False


def test_jntmed_size_under_the_complete_null():
    """No mediation anywhere: measured 0/30 rejections at alpha = 0.05
    (the max-p test is conservative under the complete null -- the
    known trade-off from the paper)."""
    rej = 0
    for s in range(30):
        rng = np.random.default_rng(s)
        x, m, y = rng.normal(size=(3, 150))
        rej += joint_significance_mediation(x, m, y)["significant"]
    assert rej <= 4


def test_jntmed_validates_input():
    with pytest.raises(ValueError, match="share a length"):
        joint_significance_mediation([1, 2, 3], [1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="alpha"):
        joint_significance_mediation([1.0] * 5, [1.0] * 5, [1.0] * 5, alpha=2)
