"""geoidh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geoidh import geoidh


def test_geoidh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geoidh()
