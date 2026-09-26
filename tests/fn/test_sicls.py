"""sicls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sicls import sicls


def test_sicls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sicls()
