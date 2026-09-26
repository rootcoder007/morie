"""gnsaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gnsaic import gnsaic


def test_gnsaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gnsaic(ll=None, k=None, n=None)
