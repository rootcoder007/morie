"""xrlms is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlms import lm_sarma


def test_xrlms_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lm_sarma(data=None)
