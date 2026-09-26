"""zekex is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zekex import kernel_exposure


def test_zekex_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kernel_exposure(data=None)
