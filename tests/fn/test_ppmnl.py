"""ppmnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmnl import ppmnl


def test_ppmnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmnl()
