"""clred is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clred import clred


def test_clred_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clred()
