"""svqrp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svqrp import svqrp


def test_svqrp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svqrp()
