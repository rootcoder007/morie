"""gdtrb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdtrb import gdtrb


def test_gdtrb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdtrb()
