"""foturn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foturn import foturn


def test_foturn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foturn()
