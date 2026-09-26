"""ghprb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghprb import ghprb


def test_ghprb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghprb()
