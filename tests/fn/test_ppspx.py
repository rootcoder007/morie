"""ppspx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppspx import ppspx


def test_ppspx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppspx()
