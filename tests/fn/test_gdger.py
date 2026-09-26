"""gdger is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdger import gdger


def test_gdger_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdger()
