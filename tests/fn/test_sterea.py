"""sterea is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sterea import sterea


def test_sterea_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sterea()
