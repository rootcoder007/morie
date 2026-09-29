"""Tests for jntmed.joint_significance_mediation."""

import pytest

from morie.fn import _array_core as np
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


def test_jntmed_paths_recomputed_from_closed_forms():
    """a = Sxm / Sxx; b by Frisch-Waugh (residualise m and y on x);
    t_a = a / sqrt(s^2 / Sxx) with n - 2 df."""
    import math

    from morie.fn import _stats_core as stats

    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    m = [0.3, 1.1, 1.6, 3.4, 3.9, 5.2, 6.4, 6.8]
    y = [1.0, 1.9, 3.1, 4.4, 4.8, 6.9, 7.2, 8.1]
    n = len(x)
    mx, mm, my = sum(x) / n, sum(m) / n, sum(y) / n
    sxx = math.fsum((u - mx) ** 2 for u in x)
    a = math.fsum((u - mx) * (v - mm) for u, v in zip(x, m)) / sxx
    ea = [v - mm - a * (u - mx) for u, v in zip(x, m)]
    t_a = a / math.sqrt(math.fsum(e * e for e in ea) / (n - 2) / sxx)
    gy = math.fsum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx
    ey = [v - my - gy * (u - mx) for u, v in zip(x, y)]
    b = math.fsum(p * q for p, q in zip(ea, ey)) / math.fsum(e * e for e in ea)
    r = joint_significance_mediation(x, m, y)
    assert float(r["a"]) == pytest.approx(a, rel=1e-12)
    assert float(r["b"]) == pytest.approx(b, rel=1e-11)
    assert float(r["p_a"]) == pytest.approx(2 * float(stats.t.sf(abs(t_a), n - 2)), rel=1e-9)
    assert float(r["indirect"]) == pytest.approx(a * b, rel=1e-11)
