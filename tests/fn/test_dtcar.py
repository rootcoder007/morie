"""dtcar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcar import dtcar


def test_dtcar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcar()
