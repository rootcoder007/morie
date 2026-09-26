"""sawtf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawtf import sawtf


def test_sawtf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawtf()
