"""dssim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dssim import dssim


def test_dssim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dssim()
