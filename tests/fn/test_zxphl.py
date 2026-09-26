"""zxphl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxphl import persistence_land


def test_zxphl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        persistence_land(data=None)
