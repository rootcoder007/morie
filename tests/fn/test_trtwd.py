"""trtwd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trtwd import trtwd


def test_trtwd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trtwd()
