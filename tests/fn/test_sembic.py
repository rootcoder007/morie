"""sembic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sembic import sembic


def test_sembic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sembic(ll=None, k=None, n=None)
