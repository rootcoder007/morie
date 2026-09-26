"""xrlmb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlmb import lm_robust_error


def test_xrlmb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lm_robust_error(data=None)
