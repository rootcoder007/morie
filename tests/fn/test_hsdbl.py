"""hsdbl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsdbl import hsdbl


def test_hsdbl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsdbl()
