"""xrlml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlml import lm_lag


def test_xrlml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lm_lag(data=None)
