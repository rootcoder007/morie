"""mttrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mttrs import mttrs


def test_mttrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mttrs()
