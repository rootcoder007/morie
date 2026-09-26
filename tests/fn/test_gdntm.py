"""gdntm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdntm import gdntm


def test_gdntm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdntm()
