"""scpaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.scpaic import scpaic


def test_scpaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scpaic(ll=None, k=None, n=None)
