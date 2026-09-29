"""Tests for morie.fn.prob_or_exclusive: values recomputed from first principles."""

import math

from morie.fn.prob_or_exclusive import prob_or_exclusive


def test_sum_rule():
    ps = [0.1, 0.25, 0.3]
    assert abs(prob_or_exclusive(ps)["p_or"] - math.fsum(ps)) < 1e-15


def test_rejects_non_exclusive():
    import pytest

    with pytest.raises(ValueError):
        prob_or_exclusive([0.6, 0.7])
