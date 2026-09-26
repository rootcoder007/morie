"""sacsig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sacsig import sacsig


def test_sacsig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sacsig(resid=None, n=None)
