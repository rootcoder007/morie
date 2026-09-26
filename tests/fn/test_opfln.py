"""opfln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opfln import opfln


def test_opfln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opfln()
