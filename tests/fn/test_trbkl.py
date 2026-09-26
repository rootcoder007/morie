"""trbkl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trbkl import trbkl


def test_trbkl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trbkl()
