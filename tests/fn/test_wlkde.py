"""wlkde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlkde import wlkde


def test_wlkde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlkde()
