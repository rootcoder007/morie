"""zxhvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxhvr import haversine_dist


def test_zxhvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        haversine_dist(data=None)
