"""soorg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soorg import soorg


def test_soorg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soorg()
