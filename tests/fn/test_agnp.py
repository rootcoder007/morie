"""agnp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agnp import agnp


def test_agnp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agnp()
