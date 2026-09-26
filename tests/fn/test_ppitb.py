"""ppitb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppitb import ppitb


def test_ppitb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppitb()
