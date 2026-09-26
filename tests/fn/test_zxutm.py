"""zxutm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxutm import utm_convert


def test_zxutm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        utm_convert(data=None)
