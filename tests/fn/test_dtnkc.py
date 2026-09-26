"""dtnkc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtnkc import dtnkc


def test_dtnkc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtnkc()
