"""albers is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.albers import albers


def test_albers_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        albers()
