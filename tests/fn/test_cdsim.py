"""cdsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdsim import cdsim


def test_cdsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdsim()
