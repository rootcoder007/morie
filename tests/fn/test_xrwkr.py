"""xrwkr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwkr import w_kernel


def test_xrwkr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_kernel(data=None)
