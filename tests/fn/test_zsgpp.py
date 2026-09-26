"""zsgpp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgpp import gp_predict


def test_zsgpp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gp_predict(data=None)
