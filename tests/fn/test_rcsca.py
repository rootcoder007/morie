"""rcsca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcsca import rcsca


def test_rcsca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcsca()
