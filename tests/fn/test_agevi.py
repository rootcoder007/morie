"""agevi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agevi import agevi


def test_agevi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agevi()
