"""masis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.masis import masis


def test_masis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        masis()
