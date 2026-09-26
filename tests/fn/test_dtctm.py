"""dtctm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtctm import dtctm


def test_dtctm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtctm()
