"""enwnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enwnd import enwnd


def test_enwnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enwnd()
