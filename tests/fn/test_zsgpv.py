"""zsgpv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgpv import gp_variance


def test_zsgpv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gp_variance(data=None)
