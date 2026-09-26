"""co2Trnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.co2Trnd import co2_trend


def test_co2Trnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        co2_trend(co2_monthly=None)
