"""gdotm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdotm import gdotm


def test_gdotm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdotm()
