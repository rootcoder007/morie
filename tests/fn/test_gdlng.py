"""gdlng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdlng import gdlng


def test_gdlng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdlng()
