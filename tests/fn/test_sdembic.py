"""sdembic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdembic import sdembic


def test_sdembic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdembic(ll=None, k=None, n=None)
