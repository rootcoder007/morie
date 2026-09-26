"""enfir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enfir import enfir


def test_enfir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enfir()
