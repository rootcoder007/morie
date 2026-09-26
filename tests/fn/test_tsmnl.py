"""tsmnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsmnl import tsmnl


def test_tsmnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsmnl()
