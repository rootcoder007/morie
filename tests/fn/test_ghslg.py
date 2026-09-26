"""ghslg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghslg import ghslg


def test_ghslg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghslg()
