"""dk3co is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3co import dk3co


def test_dk3co_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3co()
