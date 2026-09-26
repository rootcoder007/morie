"""opsa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opsa import opsa


def test_opsa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opsa()
