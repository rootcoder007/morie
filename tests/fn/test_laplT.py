"""laplT is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.laplT import laplace_transform


def test_laplT_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        laplace_transform(f=None, t=None, s=None)
