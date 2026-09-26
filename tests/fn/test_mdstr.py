"""mdstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdstr import mdstr


def test_mdstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdstr()
