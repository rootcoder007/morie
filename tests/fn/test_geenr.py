"""geenr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geenr import geenr


def test_geenr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geenr()
