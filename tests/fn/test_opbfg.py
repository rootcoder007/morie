"""opbfg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opbfg import opbfg


def test_opbfg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opbfg()
