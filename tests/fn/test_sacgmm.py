"""sacgmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sacgmm import sacgmm


def test_sacgmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sacgmm(y=None, X=None, W=None)
