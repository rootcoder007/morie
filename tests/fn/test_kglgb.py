"""kglgb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kglgb import lk_backtransform


def test_kglgb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lk_backtransform(data=None)
