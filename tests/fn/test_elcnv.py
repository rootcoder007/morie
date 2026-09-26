"""elcnv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elcnv import elcnv


def test_elcnv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elcnv()
