"""cldbx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cldbx import cldbx


def test_cldbx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cldbx()
