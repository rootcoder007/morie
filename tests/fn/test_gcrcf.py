"""gcrcf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcrcf import gcrcf


def test_gcrcf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcrcf()
