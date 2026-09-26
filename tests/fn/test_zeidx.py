"""zeidx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeidx import idw_exposure


def test_zeidx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idw_exposure(data=None)
