"""ptmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptmat import matern_process


def test_ptmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        matern_process(data=None)
