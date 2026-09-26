"""crpsF is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.crpsF import crps


def test_crpsF_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        crps(forecast_cdf=None, y=None)
