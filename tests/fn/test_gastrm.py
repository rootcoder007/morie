"""gastrm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gastrm import gastrm


def test_gastrm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gastrm()
