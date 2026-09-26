"""caraic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.caraic import caraic


def test_caraic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        caraic(ll=None, k=None, n=None)
