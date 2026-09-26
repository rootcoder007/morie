"""zxwnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxwnd import wind_rose


def test_zxwnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wind_rose(data=None)
