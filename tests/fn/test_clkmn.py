"""clkmn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clkmn import clkmn


def test_clkmn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clkmn()
