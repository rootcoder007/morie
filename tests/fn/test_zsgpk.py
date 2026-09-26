"""zsgpk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgpk import gp_kernel


def test_zsgpk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gp_kernel(data=None)
