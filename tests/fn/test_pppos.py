"""pppos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pppos import pppos


def test_pppos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pppos()
