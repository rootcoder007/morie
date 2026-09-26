"""pltrb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pltrb import pltrb


def test_pltrb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pltrb()
