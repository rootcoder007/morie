"""opmed is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opmed import opmed


def test_opmed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opmed()
