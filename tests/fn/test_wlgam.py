"""wlgam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlgam import wlgam


def test_wlgam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlgam()
