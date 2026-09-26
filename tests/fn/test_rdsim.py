"""rdsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rdsim import rdsim


def test_rdsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rdsim()
