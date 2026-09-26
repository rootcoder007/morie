"""svmnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svmnl import multinomial_spatial


def test_svmnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        multinomial_spatial(data=None)
