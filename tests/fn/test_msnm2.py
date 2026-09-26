"""msnm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msnm2 import nonmetric_2d


def test_msnm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nonmetric_2d(data=None)
