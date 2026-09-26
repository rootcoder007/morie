"""maosp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maosp import maosp


def test_maosp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maosp()
