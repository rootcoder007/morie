"""rbftps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbftps import rbftps


def test_rbftps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbftps()
