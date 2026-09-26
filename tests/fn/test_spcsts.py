"""spcsts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcsts import spcsts


def test_spcsts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcsts()
