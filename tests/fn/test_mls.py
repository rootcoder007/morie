"""mls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mls import mls


def test_mls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mls()
