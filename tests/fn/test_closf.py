"""closf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.closf import closf


def test_closf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        closf()
