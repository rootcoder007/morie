"""opcko is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opcko import opcko


def test_opcko_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opcko()
