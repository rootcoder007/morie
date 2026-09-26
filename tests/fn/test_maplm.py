"""maplm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maplm import maplm


def test_maplm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maplm()
