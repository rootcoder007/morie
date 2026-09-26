"""isidx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isidx import isidx


def test_isidx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isidx()
