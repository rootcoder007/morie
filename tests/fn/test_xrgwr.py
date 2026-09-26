"""xrgwr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwr import gwr_basic


def test_xrgwr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_basic(data=None)
