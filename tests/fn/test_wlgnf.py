"""wlgnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlgnf import wlgnf


def test_wlgnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlgnf()
