"""agwtr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agwtr import agwtr


def test_agwtr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agwtr()
