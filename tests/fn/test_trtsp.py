"""trtsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trtsp import trtsp


def test_trtsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trtsp()
