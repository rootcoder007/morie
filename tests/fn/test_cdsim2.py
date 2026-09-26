"""cdsim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdsim2 import cdsim2


def test_cdsim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdsim2()
