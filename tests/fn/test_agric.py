"""agric is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agric import agric


def test_agric_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agric()
