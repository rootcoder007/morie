"""gdgin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdgin import gdgin


def test_gdgin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdgin()
