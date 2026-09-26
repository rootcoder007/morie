"""gdlit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdlit import gdlit


def test_gdlit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdlit()
