"""abmsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abmsm import abmsm


def test_abmsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abmsm()
