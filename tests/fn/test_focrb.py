"""focrb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focrb import focrb


def test_focrb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focrb()
