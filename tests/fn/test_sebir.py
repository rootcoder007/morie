"""sebir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebir import sebir


def test_sebir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebir()
