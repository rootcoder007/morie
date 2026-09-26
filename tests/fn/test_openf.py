"""openf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.openf import openf


def test_openf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        openf()
