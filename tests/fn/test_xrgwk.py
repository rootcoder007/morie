"""xrgwk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgwk import gwr_kernel


def test_xrgwk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwr_kernel(data=None)
