"""rspca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rspca import rspca


def test_rspca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rspca()
