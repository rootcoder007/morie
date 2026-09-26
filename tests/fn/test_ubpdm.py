"""ubpdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubpdm import ubpdm


def test_ubpdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubpdm()
