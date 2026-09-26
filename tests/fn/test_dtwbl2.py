"""dtwbl2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtwbl2 import dtwbl2


def test_dtwbl2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtwbl2()
