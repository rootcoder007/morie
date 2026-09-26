"""splgaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splgaic import splgaic


def test_splgaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splgaic(ll=None, k=None, n=None)
