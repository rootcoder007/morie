"""spcnst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcnst import spcnst


def test_spcnst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcnst()
