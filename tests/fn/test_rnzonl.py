"""rnzonl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnzonl import rnzonl


def test_rnzonl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnzonl()
