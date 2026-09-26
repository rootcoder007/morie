"""sbpiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbpiv import sbpiv


def test_sbpiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbpiv()
