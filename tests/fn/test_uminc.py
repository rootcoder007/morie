"""uminc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.uminc import uminc


def test_uminc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uminc()
