"""geind is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geind import geind


def test_geind_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geind()
