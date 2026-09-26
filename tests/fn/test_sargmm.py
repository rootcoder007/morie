"""sargmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sargmm import sargmm


def test_sargmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sargmm(y=None, X=None, W=None)
