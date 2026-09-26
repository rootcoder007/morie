"""ppint2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppint2 import ppint2


def test_ppint2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppint2()
