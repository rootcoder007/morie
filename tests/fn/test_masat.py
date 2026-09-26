"""masat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.masat import masat


def test_masat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        masat()
