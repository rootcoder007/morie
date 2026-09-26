"""sfmoran is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfmoran import sfmoran


def test_sfmoran_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfmoran(W=None)
