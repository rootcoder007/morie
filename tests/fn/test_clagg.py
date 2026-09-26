"""clagg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clagg import clagg


def test_clagg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clagg()
