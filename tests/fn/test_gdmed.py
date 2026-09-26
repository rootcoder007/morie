"""gdmed is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmed import gdmed


def test_gdmed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmed()
