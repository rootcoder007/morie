"""sgsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgsim import sgsim


def test_sgsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgsim()
