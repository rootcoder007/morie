"""gemin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gemin import gemin


def test_gemin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gemin()
