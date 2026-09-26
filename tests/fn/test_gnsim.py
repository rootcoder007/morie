"""gnsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gnsim import gnsim


def test_gnsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gnsim()
