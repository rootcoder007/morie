"""gehlc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gehlc import gehlc


def test_gehlc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gehlc()
