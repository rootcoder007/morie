"""maradr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maradr import maradr


def test_maradr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maradr()
