"""wlsdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlsdm import wlsdm


def test_wlsdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlsdm()
