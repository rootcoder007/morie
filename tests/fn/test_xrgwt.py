"""xrgwt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwt import gwr_tvalues


def test_xrgwt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_tvalues(data=None)
