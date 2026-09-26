"""zxw84 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxw84 import wgs84_to_local


def test_zxw84_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wgs84_to_local(data=None)
