"""zsgps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgps import gp_spatial


def test_zsgps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gp_spatial(data=None)
