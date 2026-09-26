"""vmnst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmnst import vmnst


def test_vmnst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmnst()
