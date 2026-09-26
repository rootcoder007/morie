"""dk4co is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk4co import dk4co


def test_dk4co_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk4co()
