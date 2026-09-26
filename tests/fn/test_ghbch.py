"""ghbch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghbch import ghbch


def test_ghbch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghbch()
