"""clelb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clelb import clelb


def test_clelb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clelb()
