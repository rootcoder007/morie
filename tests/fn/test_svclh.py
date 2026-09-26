"""svclh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svclh import coalition_heart


def test_svclh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        coalition_heart(data=None)
