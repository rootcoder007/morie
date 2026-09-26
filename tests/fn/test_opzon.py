"""opzon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opzon import opzon


def test_opzon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opzon()
