"""dtfsk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtfsk import dtfsk


def test_dtfsk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtfsk()
