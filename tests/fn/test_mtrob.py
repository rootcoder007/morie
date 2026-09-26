"""mtrob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtrob import mtrob


def test_mtrob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtrob()
