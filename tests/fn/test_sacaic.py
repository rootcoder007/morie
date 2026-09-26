"""sacaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sacaic import sacaic


def test_sacaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sacaic(ll=None, k=None, n=None)
