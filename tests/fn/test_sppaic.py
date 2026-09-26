"""sppaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sppaic import sppaic


def test_sppaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sppaic(ll=None, k=None, n=None)
