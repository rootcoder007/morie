"""gpclf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpclf import gpclf


def test_gpclf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpclf()
