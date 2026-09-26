"""zxgrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxgrc import great_circle


def test_zxgrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        great_circle(data=None)
