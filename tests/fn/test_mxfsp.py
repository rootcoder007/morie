"""mxfsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mxfsp import mxfsp


def test_mxfsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mxfsp()
