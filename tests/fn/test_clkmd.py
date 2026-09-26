"""clkmd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clkmd import clkmd


def test_clkmd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clkmd()
