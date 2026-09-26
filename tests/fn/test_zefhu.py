"""zefhu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zefhu import fay_herriot_unit


def test_zefhu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fay_herriot_unit(data=None)
