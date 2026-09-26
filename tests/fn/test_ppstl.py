"""ppstl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppstl import ppstl


def test_ppstl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppstl()
