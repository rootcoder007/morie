"""Tests for morie.fn.nocse: values recomputed from the definition."""

import math

from morie.fn.nocse import nominate_confidence_interval


def test_normal_half_widths():
    from morie.fn._stats_core import norm

    r = nominate_confidence_interval([0.1, 0.25], alpha=0.1)
    z = float(norm.ppf(0.95))
    assert abs(r.value - z) < 1e-14
    assert abs(r.extra["ci_half_widths"][1] - z * 0.25) < 1e-14
    # the 95 percent critical value solves Phi(z) = 0.975
    z95 = nominate_confidence_interval([1.0]).value
    assert abs(0.5 * math.erfc(-z95 / math.sqrt(2)) - 0.975) < 1e-12
