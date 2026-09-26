"""mgwrr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mgwrr2 import mgwrr2


def test_mgwrr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mgwrr2(y=None, y_hat=None)
