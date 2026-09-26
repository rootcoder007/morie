"""rnfocl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnfocl import rnfocl


def test_rnfocl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnfocl()
