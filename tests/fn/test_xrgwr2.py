"""xrgwr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwr2 import gwr_rsquared


def test_xrgwr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_rsquared(data=None)
