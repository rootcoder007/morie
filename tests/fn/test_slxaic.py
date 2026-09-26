"""slxaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slxaic import slxaic


def test_slxaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slxaic(ll=None, k=None, n=None)
