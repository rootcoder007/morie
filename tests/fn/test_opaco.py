"""opaco is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opaco import opaco


def test_opaco_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opaco()
