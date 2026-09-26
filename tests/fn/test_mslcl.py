"""mslcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mslcl import lcmc


def test_mslcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lcmc(data=None)
