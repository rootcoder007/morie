"""nbund is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbund import nbund


def test_nbund_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbund()
