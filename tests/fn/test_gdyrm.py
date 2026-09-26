"""gdyrm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdyrm import gdyrm


def test_gdyrm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdyrm()
