"""grwsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.grwsp import grwsp


def test_grwsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grwsp()
