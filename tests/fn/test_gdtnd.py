"""gdtnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdtnd import gdtnd


def test_gdtnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdtnd()
