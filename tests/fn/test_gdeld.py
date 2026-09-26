"""gdeld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdeld import gdeld


def test_gdeld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdeld()
