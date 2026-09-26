"""xrgwb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwb import gwr_bandwidth


def test_xrgwb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_bandwidth(data=None)
