"""saraic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saraic import saraic


def test_saraic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saraic(ll=None, k=None, n=None)
