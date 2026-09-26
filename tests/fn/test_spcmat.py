"""spcmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcmat import spcmat


def test_spcmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcmat()
