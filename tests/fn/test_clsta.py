"""clsta is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clsta import clsta


def test_clsta_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clsta()
