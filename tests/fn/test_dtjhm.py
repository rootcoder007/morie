"""dtjhm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtjhm import dtjhm


def test_dtjhm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtjhm()
