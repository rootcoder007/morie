"""kgtrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgtrn import kriging_trend_surface


def test_kgtrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_trend_surface(values=None, x=None)
