"""stch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stch import stch


def test_stch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stch()
