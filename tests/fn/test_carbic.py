"""carbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.carbic import carbic


def test_carbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        carbic(ll=None, k=None, n=None)
