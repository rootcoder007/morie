"""ghclr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghclr import ghclr


def test_ghclr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghclr()
