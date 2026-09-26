"""gdmap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmap import gdmap


def test_gdmap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmap()
