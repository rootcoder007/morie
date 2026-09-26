"""sdemaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdemaic import sdemaic


def test_sdemaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdemaic(ll=None, k=None, n=None)
