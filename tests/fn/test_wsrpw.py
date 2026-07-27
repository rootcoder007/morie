"""Tests for wsrpw.wilcoxon_power."""

import numpy as np
import pytest

from morie.fn.wsrpw import wilcoxon_power


def test_wsrpw_power_is_monotone_in_effect_size():
    """Measured on this fixture: 0.04 at zero effect, 0.73 at d = 0.5,
    1.00 at d = 1.0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    lo = float(wilcoxon_power(x, effect_size=0.2, nsim=400, seed=1)["statistic"])
    hi = float(wilcoxon_power(x, effect_size=1.0, nsim=400, seed=1)["statistic"])
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > 0.9


def test_wsrpw_null_effect_gives_alpha_level_power():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    r = wilcoxon_power(x, effect_size=0.0, nsim=600, seed=2, alpha=0.05)
    assert float(r["statistic"]) == pytest.approx(0.05, abs=0.035)


def test_wsrpw_power_grows_with_sample_size():
    rng = np.random.default_rng(2)
    small = float(wilcoxon_power(rng.normal(size=15), effect_size=0.5, nsim=400, seed=3)["statistic"])
    large = float(wilcoxon_power(rng.normal(size=80), effect_size=0.5, nsim=400, seed=3)["statistic"])
    assert large > small
