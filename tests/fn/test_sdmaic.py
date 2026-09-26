"""sdmaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdmaic import sdmaic


def test_sdmaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdmaic(ll=None, k=None, n=None)
