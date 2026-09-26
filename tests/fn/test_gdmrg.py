"""gdmrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmrg import gdmrg


def test_gdmrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmrg()
