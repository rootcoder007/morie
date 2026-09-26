"""kgunt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgunt import uk_trend


def test_kgunt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uk_trend(data=None)
