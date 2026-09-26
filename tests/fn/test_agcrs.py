"""agcrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agcrs import agcrs


def test_agcrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agcrs()
