"""abemf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abemf import abemf


def test_abemf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abemf()
