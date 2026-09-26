"""gdpty is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdpty import gdpty


def test_gdpty_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdpty()
