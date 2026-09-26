"""sopot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sopot import sopot


def test_sopot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sopot()
