"""mcanis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcanis import mcanis


def test_mcanis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcanis()
