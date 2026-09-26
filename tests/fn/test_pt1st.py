"""pt1st is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pt1st import first_order_pp


def test_pt1st_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        first_order_pp(data=None)
