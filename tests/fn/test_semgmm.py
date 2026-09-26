"""semgmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semgmm import semgmm


def test_semgmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semgmm(y=None, X=None, W=None)
