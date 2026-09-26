"""rfmax is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfmax import rfmax


def test_rfmax_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfmax()
