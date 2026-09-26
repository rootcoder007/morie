"""sfaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfaic import sfaic


def test_sfaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfaic(ll=None, k=None, n=None)
