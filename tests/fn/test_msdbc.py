"""msdbc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msdbc import double_center


def test_msdbc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        double_center(data=None)
