"""ppitj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppitj import ppitj


def test_ppitj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppitj()
