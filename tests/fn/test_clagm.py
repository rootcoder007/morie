"""clagm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clagm import clagm


def test_clagm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clagm()
