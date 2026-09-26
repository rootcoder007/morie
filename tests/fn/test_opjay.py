"""opjay is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opjay import opjay


def test_opjay_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opjay()
