"""pt2nd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pt2nd import second_order_pp


def test_pt2nd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        second_order_pp(data=None)
