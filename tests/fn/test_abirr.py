"""abirr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abirr import abirr


def test_abirr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abirr()
