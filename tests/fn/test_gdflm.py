"""gdflm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdflm import gdflm


def test_gdflm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdflm()
