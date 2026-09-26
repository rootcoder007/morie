"""icc31c is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.icc31c import icc_one_way


def test_icc31c_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        icc_one_way(X=None)
