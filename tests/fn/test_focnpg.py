"""focnpg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focnpg import focnpg


def test_focnpg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focnpg()
