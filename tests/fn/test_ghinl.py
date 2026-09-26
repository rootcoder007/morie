"""ghinl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghinl import ghinl


def test_ghinl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghinl()
