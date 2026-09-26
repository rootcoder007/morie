"""wlest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlest import wlest


def test_wlest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlest()
