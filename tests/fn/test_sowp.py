"""sowp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sowp import sowp


def test_sowp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sowp()
