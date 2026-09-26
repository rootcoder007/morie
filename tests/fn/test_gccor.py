"""gccor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gccor import gccor


def test_gccor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gccor()
