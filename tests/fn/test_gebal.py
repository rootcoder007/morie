"""gebal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gebal import gebal


def test_gebal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gebal()
