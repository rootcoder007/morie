"""mcsrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcsrs import mcsrs


def test_mcsrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcsrs()
