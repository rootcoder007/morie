"""bgsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bgsim import bgsim


def test_bgsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bgsim()
