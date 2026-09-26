"""gdmar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmar import gdmar


def test_gdmar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmar()
