"""dtcfa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcfa import dtcfa


def test_dtcfa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcfa()
