"""ppitv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppitv import ppitv


def test_ppitv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppitv()
