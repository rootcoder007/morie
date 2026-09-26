"""ghsir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghsir import ghsir


def test_ghsir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghsir()
