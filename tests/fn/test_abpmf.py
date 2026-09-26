"""abpmf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abpmf import abpmf


def test_abpmf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abpmf()
