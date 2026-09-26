"""ghchg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghchg import ghchg


def test_ghchg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghchg()
