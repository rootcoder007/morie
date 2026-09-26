"""carvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.carvr import carvr


def test_carvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        carvr(var_sp=None, var_un=None)
