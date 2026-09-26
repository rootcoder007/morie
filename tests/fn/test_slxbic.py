"""slxbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slxbic import slxbic


def test_slxbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slxbic(ll=None, k=None, n=None)
