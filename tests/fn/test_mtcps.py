"""mtcps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcps import mtcps


def test_mtcps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcps()
