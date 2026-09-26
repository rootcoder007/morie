"""hullel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hullel import hullel


def test_hullel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hullel()
