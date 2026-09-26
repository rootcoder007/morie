"""tmhp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmhp import tmhp


def test_tmhp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmhp()
