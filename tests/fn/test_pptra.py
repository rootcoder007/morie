"""pptra is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pptra import pptra


def test_pptra_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pptra()
