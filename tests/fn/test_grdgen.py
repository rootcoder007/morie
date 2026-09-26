"""grdgen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.grdgen import grdgen


def test_grdgen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grdgen()
