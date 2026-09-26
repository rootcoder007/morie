"""cstpp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cstpp import cstpp


def test_cstpp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cstpp()
