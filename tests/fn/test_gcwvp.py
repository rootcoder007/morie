"""gcwvp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcwvp import gcwvp


def test_gcwvp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcwvp()
