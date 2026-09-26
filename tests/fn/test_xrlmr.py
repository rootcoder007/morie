"""xrlmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlmr import lm_robust_lag


def test_xrlmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lm_robust_lag(data=None)
