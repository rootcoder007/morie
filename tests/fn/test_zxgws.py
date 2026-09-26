"""zxgws is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxgws import gw_summary


def test_zxgws_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gw_summary(data=None)
