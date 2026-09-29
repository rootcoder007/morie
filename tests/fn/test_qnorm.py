"""Tests for morie.fn.qnorm — normal quantile function."""

import pytest

from morie.fn import _array_core as np
from morie.fn.qnorm import qnorm


class TestQnorm:
    """Tests for qnorm()."""

    def test_median(self):
        """qnorm(0.5) = 0.0 for standard normal."""
        assert qnorm(0.5) == pytest.approx(0.0, abs=1e-12)

    def test_975(self):
        """qnorm(0.975) ~ 1.96."""
        assert qnorm(0.975) == pytest.approx(1.96, abs=1e-2)

    def test_025(self):
        """qnorm(0.025) ~ -1.96."""
        assert qnorm(0.025) == pytest.approx(-1.96, abs=1e-2)

    def test_type(self):
        """Scalar input returns float."""
        result = qnorm(0.5)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_sd(self):
        """Should reject sd <= 0."""
        with pytest.raises(ValueError):
            qnorm(0.5, sd=0)


def test_qnorm_inverts_the_normal_cdf():
    import math

    for p in (0.001, 0.2, 0.5, 0.8, 0.975):
        q = qnorm(p, mean=1.5, sd=2.0)
        assert 0.5 * math.erfc(-((q - 1.5) / 2.0) / math.sqrt(2)) == pytest.approx(p, rel=1e-12)
    assert qnorm(math.log(0.3), log=True) == pytest.approx(qnorm(0.3), rel=1e-14)
    assert qnorm(0.3, lower_tail=False) == pytest.approx(-qnorm(0.3), rel=1e-14)
