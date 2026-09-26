"""cldun is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cldun import cldun


def test_cldun_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cldun()
