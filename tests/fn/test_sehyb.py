"""sehyb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sehyb import sehyb


def test_sehyb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sehyb()
