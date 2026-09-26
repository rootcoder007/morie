"""maklp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maklp import maklp


def test_maklp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maklp()
