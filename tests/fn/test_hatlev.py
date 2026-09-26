"""hatlev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hatlev import leverage


def test_hatlev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        leverage(X=None)
