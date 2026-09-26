"""gdref is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdref import gdref


def test_gdref_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdref()
