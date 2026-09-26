"""wlbbm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlbbm import wlbbm


def test_wlbbm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlbbm()
