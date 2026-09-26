"""mppos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mppos import mppos


def test_mppos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mppos()
