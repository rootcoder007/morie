"""chltps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chltps import chltps


def test_chltps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chltps()
