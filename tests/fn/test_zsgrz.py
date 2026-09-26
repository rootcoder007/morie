"""zsgrz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgrz import grid_zonal


def test_zsgrz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grid_zonal(data=None)
