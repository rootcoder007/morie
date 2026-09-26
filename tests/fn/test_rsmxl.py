"""rsmxl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsmxl import rsmxl


def test_rsmxl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsmxl()
