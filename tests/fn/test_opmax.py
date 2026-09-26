"""opmax is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opmax import opmax


def test_opmax_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opmax()
