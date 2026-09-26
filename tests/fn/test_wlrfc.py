"""wlrfc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlrfc import wlrfc


def test_wlrfc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlrfc()
