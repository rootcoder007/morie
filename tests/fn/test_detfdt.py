"""detfdt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.detfdt import detrended_fluctuation


def test_detfdt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        detrended_fluctuation(y=None, scales=None)
