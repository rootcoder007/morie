"""gnsbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gnsbic import gnsbic


def test_gnsbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gnsbic(ll=None, k=None, n=None)
